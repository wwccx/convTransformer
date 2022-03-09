import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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
    def __init__(self, win_size=(7, 7), dim=96, num_heads=3, attn_drop=0., convDotMul=None):
        super(convAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.window_size = win_size
        if not convDotMul:
            self.convDotMul = torch.nn.Conv2d(dim // num_heads * 2, win_size[0] * win_size[1], kernel_size=win_size,
                                              stride=1, padding=(win_size[0] // 2, win_size[1] // 2))
        else:
            self.convDotMul = convDotMul
        self.softmax = nn.Softmax(dim=2)
        self.scale = (dim // num_heads) ** -0.5

        # position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((win_size[0] * win_size[1], num_heads))
        )  # 2*Wh-1 * 2*Ww-1, nH

        self.v_padding = nn.ConstantPad2d((win_size[1] // 2, win_size[1] // 2, win_size[0] // 2, win_size[0] // 2), 0)

        # TODO: position encoding and rewrite the for loop for v mat generation

        self.proj = torch.nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_drop = torch.nn.Dropout2d(p=attn_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # self.register_buffer()

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W).permute(1, 0, 2, 3, 4, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Batch, num_heads, dim//num_heads, H, W
        attentionMap = self.convDotMul(
            torch.cat(
                (torch.flatten(q * self.scale, start_dim=0, end_dim=1), torch.flatten(k, start_dim=0, end_dim=1)), dim=1
            )
        ).view(B, self.num_heads, self.window_size[0] * self.window_size[1], H, W)  # Batch, num_heads, w*w, H, W

        relative_position_bias = self.relative_position_bias_table.permute(1, 0).view(
            1, self.num_heads, self.window_size[0] * self.window_size[1], 1, 1
        )
        attentionMap += relative_position_bias
        attentionMap = self.softmax(attentionMap)
        # Batch, num_heads, w*w, H, W

        v_padding = self.v_padding(
            v.flatten(0, 1)
        )
        v_padding = v_padding.view(B, self.num_heads, C // self.num_heads,
                                   H + self.window_size[0] - 1, W + self.window_size[1] - 1).unsqueeze(3)
        # Batch, num_heads, dim//num_heads, 1, Hp, Wp

        v = torch.cat(
            [v_padding[:, :, :, :, i // self.window_size[0]:i // self.window_size[0] + H,
             i % self.window_size[0]:i % self.window_size[0] + W]
             for i in range(self.window_size[0] * self.window_size[1])
             ], dim=3
        )

        x = torch.sum(v * attentionMap.unsqueeze(2), dim=3).reshape(B, C, H, W)
        # print(torch.max(x - v.reshape(B, C, H, W)))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_vpadding_index(self, H, W, padding=2):
        Hp = H + padding * 2
        Wp = W + padding * 2


class ConvTransBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=(7, 7), attn_drop=0., conv_drop=0., drop_path=0.,
                 conv_dim_ratio=4, act_layer=nn.GELU, convDotMul=None):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.dim = dim
        self.convAttention = convAttention(dim=dim, num_heads=num_heads, win_size=win_size,
                                           attn_drop=attn_drop, convDotMul=convDotMul)
        self.convLayers = convLayers(dim, hidden_dim=int(dim * conv_dim_ratio), act_layer=act_layer, drop=conv_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        # print(self.dim, x.shape)
        x = self.norm(x)

        x = self.convAttention(x)

        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.convLayers(x))

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
                 attn_drop=0., conv_drop=0., drop_path=[0.], downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvTransBlock(dim,
                           num_heads=num_heads,
                           win_size=(win_size, win_size),
                           conv_dim_ratio=conv_dim_ratio,
                           attn_drop=attn_drop,
                           conv_drop=conv_drop,
                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                           convDotMul=nn.Conv2d(dim // num_heads * 2,
                                                win_size * win_size,
                                                kernel_size=win_size,
                                                stride=1,
                                                padding=win_size // 2)
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


class convTransformer(nn.Module):
    def __init__(self, num_classes=10, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = patchEmbedding(embed_dim=embed_dim, norm_layer=nn.BatchNorm2d)

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
                               downsample=nn.Conv2d(int(embed_dim * 2 ** i),
                                                    int(embed_dim * 2 ** i * 2),
                                                    kernel_size=2,
                                                    stride=2)
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
    c = convTransformer(num_classes=10, embed_dim=96, window_size=7, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]).cuda()
    a = torch.rand(1, 3, 256, 256).cuda()
    import time

    t = time.time()
    b = c(a)
    print(time.time() - t)
    summary(c, (3, 224, 224), batch_size=8)
