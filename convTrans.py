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


class convAttnModelFunction(Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, q, k, shapeInfo):
        B, C, H, W, Heads, win = tuple(shapeInfo)
        ctx.save_for_backward(q, k, shapeInfo)
        attnMap = torch.empty((B, Heads, win * win, H, W), device="cuda:0")
        convAttn.convAttn(attnMap, q, k, B, Heads, win, C, H, W)
        return attnMap

    @staticmethod
    @amp.float_function
    def backward(ctx, grad_output):
        q, k, shapeInfo = ctx.saved_tensors
        B, C, H, W, Heads, win = tuple(shapeInfo)
        grad_q = torch.empty((B, C, H, W), device="cuda:0")
        grad_k = torch.empty((B, C, H + win - 1, W + win - 1), device="cuda:0")
        convAttnBackward.convAttnBackward(grad_q, grad_k, grad_output, q, k, B, Heads, win, C, H, W)

        return grad_q, grad_k, None


class applyAttnModelFunction(Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, attn, v, shapeInfo):
        B, C, H, W, Heads, win = tuple(shapeInfo)
        ctx.save_for_backward(attn, v, shapeInfo)
        x = torch.empty((B, C, H, W), device="cuda:0")
        applyAttn.applyAttn(x, attn, v, B, Heads, win, C, H, W)

        return x

    @staticmethod
    @amp.float_function
    def backward(ctx, grad_output):
        attn, v, shapeInfo = ctx.saved_tensors
        B, C, H, W, Heads, win = tuple(shapeInfo)
        grad_attn = torch.empty((B, Heads, win*win, H, W), device="cuda:0")
        grad_v = torch.empty((B, C, H + win - 1, W + win - 1), device="cuda:0")
        applyAttnBackward.applyAttnBackward(grad_attn, grad_v, grad_output, attn, v, B, Heads, win, C, H, W)

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
        # self.register_buffer()

    def forward(self, x):
        B, C, H, W = x.shape
        shapeInfo = torch.tensor((B, C, H, W, self.num_heads, self.window_size[0]))
        # if (B, H, W) != self.default_shape:
        #     # print('Oops')
        #     # print(self.default_shape)
        #     # print(B, H, W)
        #     self.trans_grid = self.get_grid((B, H, W)).cuda()

        # q, kv = torch.split(self.qkv(x), [C, 2*C], dim=1)
        qkv = self.qkv(x).reshape(B, 3, C, H, W).permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        kp = self.padding(k)
        vp = self.padding(v)
        attn_map = convAttnModelFunction.apply(q, kp, shapeInfo)
        relative_position_bias = self.relative_position_bias_table.permute(1, 0).view(
            1, self.num_heads, self.window_size[0] * self.window_size[1], 1, 1
        )
        attn_map += relative_position_bias * 0
        attn_map = self.softmax(attn_map)
        x = applyAttnModelFunction.apply(attn_map, vp, shapeInfo)
        # k = k.flatten(0, 1).unsqueeze(0).expand((self.window_size[0]*self.window_size[1], -1, -1, -1))
        # k = F.grid_sample(k, self.trans_grid, mode='nearest', align_corners=False).transpose(0, 1).reshape(
        #     B, C, self.window_size[0] * self.window_size[1], H, W
        #     )
        #
        # v = v.flatten(0, 1).unsqueeze(0).expand((self.window_size[0] * self.window_size[1], -1, -1, -1))
        # v = F.grid_sample(v, self.trans_grid, mode='nearest', align_corners=False).transpose(0, 1).reshape(
        #     B, C, self.window_size[0] * self.window_size[1], H, W
        #     )


        # kv = kv.flatten(0, 1).repeat((self.window_size[0]*self.window_size[1], 1, 1, 1))
        # kv = F.grid_sample(kv, self.trans_grid, mode='nearest', align_corners=False).reshape(
        #     self.window_size[0]*self.window_size[1], B, 2*C, H, W
        # ).permute(1, 2, 0, 3, 4)


        # q: B C H W kv: B 2C H W
        # kv = self.padding(kv).unsqueeze(2)
        # # kv: B 2C 1 H W
        # kv = torch.cat(
        #     [kv[..., i // self.window_size[0]:i // self.window_size[0] + H,
        #      i % self.window_size[0]:i % self.window_size[0] + W]
        #      for i in range(self.window_size[0] * self.window_size[1])
        #      ], dim=2
        # )
        # k, v = torch.split(kv, C, dim=1)  # B C w*w H W

        # attn_map = torch.mul(q.unsqueeze(2) * self.scale,
        #                      k).reshape(B, self.num_heads, C//self.num_heads,
        #                                 self.window_size[0]*self.window_size[1], H, W).sum(dim=2, keepdim=False)
        # # Batch, num_heads, w*w, Hp, Wp
        # relative_position_bias = self.relative_position_bias_table.permute(1, 0).view(
        #     1, self.num_heads, self.window_size[0] * self.window_size[1], 1, 1
        # )
        # attn_map += relative_position_bias
        # attn_map = self.softmax(attn_map)
        # x = torch.sum(v.reshape(B, self.num_heads, C//self.num_heads, self.window_size[0]*self.window_size[1],
        #                         H, W) * attn_map.unsqueeze(2), dim=3).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_map(self, q, k):

        assert q.shape == k.shape and len(q.shape) == 5, "input feature has wrong size"
        B, num_heads, dim_per_head, H, W = q.shape
        out_dim = self.window_size[0]*self.window_size[1]
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
        attn_map = torch.mul(q.unsqueeze(3)*self.scale, k).sum(dim=2, keepdim=False)
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
                           win_size=(win_size, win_size),
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
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class convTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=10, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, drop_path_rate=0.1, norm_layer=nn.LayerNorm, B=32, patch_resolution=(56, 56)):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = patchEmbedding(in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.BatchNorm2d)

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
                               downsample=PatchMerging(embed_dim * 2 ** i)
                               if i + 1 < len(depths) else None
                               )
            self.layers.append(layer)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (len(depths) - 1)), num_classes)
        # print(int(embed_dim * 2 ** (len(depths) - 1)))

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x.flatten(2))
        x = x.flatten(1)
        # print(x.shape)
        x = self.head(x)

        return x

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


if __name__ == '__main__':
    from torchsummary import summary

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
    a = convAttention(dim=96, num_heads=3, win_size=(7, 7), default_shape=(2, 24, 24)).cuda()
    for _ in range(100):
        b = torch.rand(2, 96, 24, 24).cuda()
        a(b)




