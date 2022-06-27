import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.cpp_extension import load
from torch.autograd import Function
from apex import amp

convAttn = load(name="convAttn",
                extra_include_paths=["include"],
                sources=["./cpp/convAttn.cpp", "./kernel/convAttn.cu"],
                verbose=True)
convAttnBackward = load(name="convAttnBackward",
                        extra_include_paths=["include"],
                        sources=["./cpp/convAttnBackward.cpp", "./kernel/convAttnBackward.cu"],
                        verbose=True)

applyAttn = load(name="applyAttn",
                 extra_include_paths=["include"],
                 sources=["./cpp/applyAttn.cpp", "./kernel/applyAttn.cu"],
                 verbose=True)
applyAttnBackward = load(name="applyAttnBackward",
                         extra_include_paths=["include"],
                         sources=["./cpp/applyAttnBackward.cpp", "./kernel/applyAttnBackward.cu"],
                         verbose=True)
# python -m torch.distributed.launch main.py --cfg ./configs/conv_tiny_patch4_window7_224.yaml --local_rank 0
# --data-path /home/server/convTransformer/data/imagenet/ILSVRC/Data/CLS-LOC --batch-size 128 --use-checkpoint True
# --tag cudaExtendWinSize7AddQscale --amp-opt-level O1
'''
python -m torch.distributed.launch main.py --cfg ./configs/conv_tiny_patch4_window7_224.yaml --local_rank 0
--data-path /home/server/convTransformer/data/imagenet/ILSVRC/Data/CLS-LOC
--batch-size 128 --use-checkpoint True --tag cudaExtendWinSize7AddQscale --amp-opt-level O1
'''

class convAttnModelFunction(Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, q, k, shapeInfo):
        B, C, H, W, Heads, winh, winw = tuple(shapeInfo)
        ctx.save_for_backward(q, k, shapeInfo)
        attnMap = torch.empty((B, Heads, winh * winw, H, W), device="cuda:0")
        convAttn.convAttn(attnMap, q, k, B, Heads, winh, winw, C, H, W)
        return attnMap

    @staticmethod
    @amp.float_function
    def backward(ctx, grad_output):
        q, k, shapeInfo = ctx.saved_tensors
        B, C, H, W, Heads, winh, winw = tuple(shapeInfo)
        grad_q = torch.empty((B, C, H, W), device="cuda:0")
        grad_k = torch.empty((B, C, H + winh - 1, W + winw - 1), device="cuda:0")
        convAttnBackward.convAttnBackward(grad_q, grad_k, grad_output, q, k, B, Heads, winh, winw, C, H, W)

        return grad_q, grad_k, None


class applyAttnModelFunction(Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, attn, v, shapeInfo):
        B, C, H, W, Heads,  winh, winw = tuple(shapeInfo)
        ctx.save_for_backward(attn, v, shapeInfo)
        x = torch.empty((B, C, H, W), device="cuda:0")
        applyAttn.applyAttn(x, attn, v, B, Heads, winh, winw, C, H, W)

        return x

    @staticmethod
    @amp.float_function
    def backward(ctx, grad_output):
        attn, v, shapeInfo = ctx.saved_tensors
        B, C, H, W, Heads, winh, winw = tuple(shapeInfo)
        grad_attn = torch.empty((B, Heads, winh * winw, H, W), device="cuda:0")
        grad_v = torch.empty((B, C, H + winh - 1, W + winw - 1), device="cuda:0")
        applyAttnBackward.applyAttnBackward(grad_attn, grad_v, grad_output, attn, v, B, Heads, winh, winw, C, H, W)

        return grad_attn, grad_v, None


class convLayers(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = in_dim or out_dim
        hidden_dim = hidden_dim or in_dim
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.act = act_layer()
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)

        return x


class convAttention(nn.Module):
    def __init__(self, win_size=(7, 7), dim=96, num_heads=3, attn_drop=0., default_shape=(32, 24, 24)):  # B H W
        super(convAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = torch.nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.window_size = win_size
        # if not convDotMul:
        #     self.convDotMul = torch.nn.Conv2d(dim // num_heads * 2, win_size[0] * win_size[1], kernel_size=win_size,
        #                                       stride=1, padding=(win_size[0] // 2, win_size[1] // 2), bias=False)
        # else:
        #     self.convDotMul = convDotMul
        self.softmax = nn.Softmax(dim=2)
        self.scale = (dim // num_heads) ** -0.5

        # position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((win_size[0] * win_size[1], num_heads))
        )  # 2*Wh-1 * 2*Ww-1, nH

        self.padding = nn.ConstantPad2d((win_size[1] // 2, win_size[1] // 2, win_size[0] // 2, win_size[0] // 2), 0)

        # TODO: position encoding and rewrite the for loop for v mat generation

        self.proj = torch.nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_drop = torch.nn.Dropout2d(p=attn_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.trans_grid = self.get_grid(default_shape).cuda()
        self.default_shape = default_shape
        self.pattern = 'cuda'
        # self.register_buffer()

    def forward(self, x):
        B, C, H, W = x.shape
        shapeInfo = torch.tensor((B, C, H, W, self.num_heads, self.window_size[0], self.window_size[1]))
        qkv = self.qkv(x).reshape(B, 3, C, H, W).permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.pattern == 'cuda':
            q = q * self.scale
            kp = self.padding(k)
            vp = self.padding(v)
            attn_map = convAttnModelFunction.apply(q, kp, shapeInfo)
            relative_position_bias = self.relative_position_bias_table.permute(1, 0).view(
                1, self.num_heads, self.window_size[0] * self.window_size[1], 1, 1
            )
            attn_map += relative_position_bias
            attn_map = self.softmax(attn_map)
            x = applyAttnModelFunction.apply(attn_map, vp, shapeInfo)
        else:
            if (B, H, W) != self.default_shape:
                self.trans_grid = self.get_grid((B, H, W)).cuda()
            k = k.flatten(0, 1).unsqueeze(0).expand((self.window_size[0] * self.window_size[1], -1, -1, -1))
            k = F.grid_sample(k, self.trans_grid, mode='nearest', align_corners=False).transpose(0, 1).reshape(
                B, C, self.window_size[0] * self.window_size[1], H, W
            )

            v = v.flatten(0, 1).unsqueeze(0).expand((self.window_size[0] * self.window_size[1], -1, -1, -1))
            v = F.grid_sample(v, self.trans_grid, mode='nearest', align_corners=False).transpose(0, 1).reshape(
                B, C, self.window_size[0] * self.window_size[1], H, W
            )
            attn_map = torch.mul(q.unsqueeze(2) * self.scale,
                                 k).reshape(B, self.num_heads, C // self.num_heads,
                                            self.window_size[0] * self.window_size[1], H, W).sum(dim=2, keepdim=False)
            # Batch, num_heads, w*w, Hp, Wp
            relative_position_bias = self.relative_position_bias_table.permute(1, 0).view(
                1, self.num_heads, self.window_size[0] * self.window_size[1], 1, 1
            )
            attn_map += relative_position_bias
            attn_map = self.softmax(attn_map)
            x = torch.sum(v.reshape(B, self.num_heads, C // self.num_heads, self.window_size[0] * self.window_size[1],
                                    H, W) * attn_map.unsqueeze(2), dim=3).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_map(self, q, k):

        assert q.shape == k.shape and len(q.shape) == 5, "input feature has wrong size"
        B, num_heads, dim_per_head, H, W = q.shape
        out_dim = self.window_size[0] * self.window_size[1]
        k_padding = self.padding(
            k.flatten(0, 1)
        ).view(B, self.num_heads, dim_per_head,
               H + self.window_size[0] - 1, W + self.window_size[1] - 1).unsqueeze(3)
        # Batch, num_heads, dim//num_heads, 1, Hp, Wp
        k = torch.cat(
            [k_padding[:, :, :, :, i // self.window_size[0]:i // self.window_size[0] + H,
             i % self.window_size[0]:i % self.window_size[0] + W]
             for i in range(self.window_size[0] * self.window_size[1])
             ], dim=3
        )  # Batch, num_heads, dim//num_heads, w*w, Hp, Wp
        attn_map = torch.mul(q.unsqueeze(3) * self.scale, k).sum(dim=2, keepdim=False)
        # Batch, num_heads, w*w, Hp, Wp
        relative_position_bias = self.relative_position_bias_table.permute(1, 0).view(
            1, self.num_heads, self.window_size[0] * self.window_size[1], 1, 1
        )
        attn_map += relative_position_bias
        attn_map = self.softmax(attn_map)

        return attn_map

    def get_grid(self, input_shape):
        B, H, W = input_shape
        scale = torch.tensor([W, H])
        pos = torch.tensor([self.window_size[1]//2, self.window_size[0]//2])
        theta = torch.eye(2).unsqueeze(0).expand(self.window_size[0]*self.window_size[1], -1, -1)

        trans = torch.stack(
            torch.meshgrid([torch.arange(self.window_size[0]), torch.arange(self.window_size[1])])[::-1]
        )   # 2, w, w
        trans = (trans.flatten(1).permute(1, 0) - pos) / scale * 2

        theta = torch.cat([theta, trans.unsqueeze(2)], dim=2)

        return F.affine_grid(theta, [trans.shape[0], B*self.dim, H, W], align_corners=False)


class ConvTransBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=(7, 7), attn_drop=0., conv_drop=0., drop_path=0.,
                 conv_dim_ratio=4, act_layer=nn.GELU, default_shape=(32, 24, 24)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dim = dim
        self.convAttention = convAttention(dim=dim, num_heads=num_heads, win_size=win_size,
                                           attn_drop=attn_drop, default_shape=default_shape)
        self.convLayers = convLayers(dim, hidden_dim=int(dim * conv_dim_ratio), act_layer=act_layer, drop=conv_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        # print(self.dim, x.shape)
        x = self.norm1(x.transpose(1, 3)).transpose(1, 3)

        x = self.convAttention(x)

        x = short_cut + self.drop_path(x)

        x = x + self.drop_path(self.convLayers(
            self.norm2(x.transpose(1, 3)).transpose(1, 3)
        ))

        return x


class patchEmbedding(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.conv_embed = nn.Conv2d(in_chans, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):

        x_embed = self.conv_embed(x)

        return x_embed


class BasicLayer(nn.Module):
    def __init__(self, depth, dim, num_heads, win_size, conv_dim_ratio=4,
                 attn_drop=0., conv_drop=0., drop_path=[0.], downsample=None, default_shape=(32, 24, 24)):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvTransBlock(dim,
                           num_heads=num_heads,
                           win_size=win_size,
                           conv_dim_ratio=conv_dim_ratio,
                           attn_drop=attn_drop,
                           conv_drop=conv_drop,
                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                           default_shape=default_shape
                           ) for i in range(depth)
        ])
        # if downsample is not None:
        self.downsample = downsample

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, patch_merging_size=(2, 2)):
        super(PatchMerging, self).__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=patch_merging_size, stride=patch_merging_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class convTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=10, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 patch_embedding_size=(4, 4), patch_merging_size=(2, 2),
                 window_size=(7, 7), drop_path_rate=0.1, norm_layer=nn.LayerNorm, B=32, patch_resolution=(56, 56),
                 fully_conv_for_grasp=False):
        super().__init__()
        self.fully_conv_for_grasp = fully_conv_for_grasp
        self.num_classes = num_classes

        self.patch_embed = patchEmbedding(in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.BatchNorm2d,
                                          patch_size=patch_embedding_size)

        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i, i_layer in enumerate(depths):
            layer = BasicLayer(depth=i_layer,
                               dim=int(embed_dim * 2 ** i),
                               num_heads=num_heads[i],
                               win_size=window_size,
                               conv_dim_ratio=4,
                               conv_drop=0,
                               attn_drop=0,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               default_shape=(B, int(patch_resolution[0]*2**(-i)), int(patch_resolution[1]*2**(-i))),
                               downsample=PatchMerging(embed_dim * 2 ** i, patch_merging_size=patch_merging_size)
                               if i + 1 < len(depths) else None
                               )
            self.layers.append(layer)
        if fully_conv_for_grasp:
            self.avgpool = torch.nn.Identity()
            # self.norm_img = nn.BatchNorm2d(int(embed_dim * 2 ** (len(depths) - 1)))
            self.norm_img = nn.Identity()
            self.norm_pose = nn.Conv2d(int(embed_dim * 2 ** (len(depths) - 1)),
                    int(embed_dim * 2 ** (len(depths) - 1)), 1, 1, 0)
            #self.res = nn.Sequential(
            #    nn.Conv2d(int(embed_dim * 2 ** (len(depths) - 1)), int(embed_dim * 2 ** (len(depths))),
            #              kernel_size=1, stride=1
            #    ),
            #    nn.BatchNorm2d(int(embed_dim * 2 ** (len(depths)))),
            #    nn.ReLU(),
            #    # nn.Conv2d(int(embed_dim * 2 ** (len(depths))), int(embed_dim * 2 ** (len(depths))),
            #    #          kernel_size=3, stride=1, padding=1
            #    #          ),
            #    # nn.BatchNorm2d(int(embed_dim * 2 ** (len(depths)))),
            #    # nn.ReLU(),
            #    nn.Conv2d(int(embed_dim * 2 ** (len(depths))), int(embed_dim * 2 ** (len(depths) - 1)),
            #              kernel_size=1, stride=1
            #              ),
            #    nn.BatchNorm2d(int(embed_dim * 2 ** (len(depths) - 1))),
            #    nn.ReLU(),
            #)
            self.head = nn.Sequential(
                    nn.Conv2d(int(embed_dim * 2 ** (len(depths) - 1)),
                        int(embed_dim * 2 ** (len(depths) - 2)),
                        kernel_size=3, stride=1
                    ),
                    nn.BatchNorm2d(int(embed_dim * 2 ** (len(depths) - 2))),
                    nn.ReLU(),
                    nn.Conv2d(int(embed_dim * 2 ** (len(depths) - 2)),
                        int(embed_dim * 2 ** (len(depths) - 3)),
                        kernel_size=3, stride=1
                    ),
                    nn.BatchNorm2d(int(embed_dim * 2 ** (len(depths) - 3))),
                    nn.ReLU(),
                    nn.Conv2d(int(embed_dim * 2 ** (len(depths) - 3)),
                        num_classes,
                        kernel_size=96 // patch_embedding_size[0] // 2 ** (len(depths)- 1) - 4, stride=1
                    ),
                )
        else:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(int(embed_dim * 2 ** (len(depths) - 1)), num_classes)
        # print(int(embed_dim * 2 ** (len(depths) - 1)))

    def forward(self, *x):
        if self.fully_conv_for_grasp:
            if len(x) == 1:
                pose = torch.zeros(x[0].shape[0]).cuda()
                x = [x[0], pose]
            return self._grasp_forward(*x)
        x = x[0]
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        # if not self.fully_conv_for_grasp:
        x = self.avgpool(x.flatten(2))
        x = x.flatten(1)
        # print(x.shape)
        x = self.head(x)

        return x
    
    def _grasp_forward(self, img, pose):
        img = self.patch_embed(img)
        for layer in self.layers:
            img = layer(img)
        # short_cut = img
        img = self.norm_img(img)
        if img.shape[0] == 1:
            img = img.repeat(pose.shape[0], 1, 1, 1)
        pose = self.norm_pose(pose.squeeze().view(pose.shape[0], 1, 1, 1).expand_as(img))
        img -= pose
        # print(img.size())
        # img = self.res(img)
        # img += short_cut
        img = self.head(img)

        return img

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


if __name__ == '__main__':
    from torchsummary import summary

    net = convTransformer(in_chans=1, num_classes=32, embed_dim=96, depths=(2, 6), num_heads=(3, 12),
                          patch_embedding_size=(4, 4), fully_conv_for_grasp=True).cuda()
    # summary(net, (1, 96, 96))
    net.load_state_dict(torch.load('./train/vfinetuneconvTrans22_06_23_22_26/convTransgrasp0926state_epoch18_acc0.2717.pth')['model'])
    net.eval()
    a = torch.zeros((1, 1, 96, 96)).cuda()
    a += 0.55
    # for i in range(10, 70):
    #     a[:, :, i, i:i+20] = 0.49
    a[:, :, 10:-10, 40:-40] = 0.10
    sf = nn.Softmax(dim=2)
    out = net(a, torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3, 0.2]).cuda()).squeeze()
    print(out.shape)
    print(sf(out.view(6, -1, 2)))
    from matplotlib import pyplot as plt

    plt.imshow(a[0, 0, :, :].cpu().numpy())
    plt.show()

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
    #     with torch.no_grad():
    #         img = torch.randn(1, 1, 400, 400).cuda()
    #         target_pre = net(img)
    #     torch.cuda.synchronize()
    #     # print(time.time() - t)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    # at = ConvTransBlock(dim=96, num_heads=3).cuda()
    # a = torch.rand(32, 96, 24, 24).cuda()
    # # summary(at, (96, 48, 48), batch_size=8)
    # b = at(a)
    # print(b.shape)
    # c = convTransformer(num_classes=10, embed_dim=96, window_size=7, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]).cuda()
    # a = torch.rand(1, 3, 256, 256).cuda()
    # import time
    #
    # t = time.time()
    # b = c(a)
    # print(time.time() - t)
    # summary(c, (3, 224, 224), batch_size=8)
    # net = convAttention(dim=96, num_heads=3, win_size=(3, 5), default_shape=(2, 24, 24)).cuda()
    # for _ in range(100):
    #     b = torch.rand(2, 96, 24, 24).cuda()
    #     a(b)
    '''
    net = convTransformer(num_classes=10, embed_dim=96, window_size=(3, 7), depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24]).cuda()

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    # net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    # x = torch.rand(1, 3, 5, 5, requires_grad=True).cuda()
    for _ in range(10):
        x = torch.rand(1, 3, 224, 224, requires_grad=True).cuda()
        # x = torch.rand(1, 96, 56, 56, requires_grad=True).cuda()
        # x = torch.ones(1, 3, 5, 5)
        net.eval()

        # x = torch.rand(1, 96, 56, 56, requires_grad=True).cuda()
        y = net(x)
        l = y.mean()
        # with amp.scale_loss(l, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        l.backward()
        # x_g = x.grad
        # net.zero_grad()
        # x.grad.zero_()
        grad = []
        for m in net.modules():
            if hasattr(m, 'weight') and m.weight.grad is not None:
                grad.append([m.weight.grad.clone(), m.__repr__()])
        net.zero_grad()
        for m in net.modules():
            if isinstance(m, convAttention):
                m.pattern = 'af'
        yaf = net(x.detach().clone())
        laf = yaf.mean()
        # with amp.scale_loss(laf, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        laf.backward()
        gradaf = []
        for m in net.modules():
            if hasattr(m, 'weight') and m.weight.grad is not None:
                gradaf.append([m.weight.grad.clone(), m.__repr__()])
        for a, b in zip(grad, gradaf):
            print(torch.allclose(a[0], b[0]), torch.max(torch.abs(a[0] - b[0]) / torch.max(a[0])), a[1]) if a[1] == b[1] \
                else print(a[1], b[1])
        # x_gaf = x.grad
        net.zero_grad()
        print('check output:', torch.allclose(y, yaf, atol=1e-5), torch.max(torch.abs(y - yaf) / torch.abs(y)))
        # print(torch.allclose(attn, attnaf, atol=1e-5), torch.max(torch.abs(attn - attnaf)/torch.abs(attn)))
        print('\n')
        '''
