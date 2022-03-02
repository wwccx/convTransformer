import torch
import torch.nn as nn
from torch.nn import functional as F

class convAttention(nn.Module):
    def __init__(self, win_size=(7, 7), dim=96, num_heads=3, attn_drop=0.):
        super(convAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.window_size = win_size
        self.convAttention = torch.nn.Conv2d(dim // num_heads * 2, win_size[0] * win_size[1], kernel_size=win_size,
                                             stride=1, padding=(win_size[0] // 2, win_size[1] // 2))
        self.softmax = nn.Softmax(dim=1)
        self.scale = (dim // num_heads) ** -0.5

        # position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.v_padding = nn.ConstantPad2d((win_size[1] // 2, win_size[1] // 2, win_size[0] // 2, win_size[0] // 2), 0)

        # TODO: position encoding and rewrite the for loop for v mat generation
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj = torch.nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_drop = torch.nn.Dropout2d(p=attn_drop)
        # self.register_buffer()

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W).permute(1, 0, 2, 3, 4, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Batch, num_heads, dim//num_heads, H, W
        attentionMap = self.convAttention(
            torch.cat(
                (torch.flatten(q * self.scale, start_dim=0, end_dim=1), torch.flatten(k, start_dim=0, end_dim=1)), dim=1
            )
        )
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        attentionMap = self.softmax(attentionMap).reshape(B, self.num_heads, self.window_size[0] * self.window_size[1], H, W)
        # Batch, num_heads, w*w, H, W

        v_padding = self.v_padding(
            v.flatten(0, 1)
        )
        v_padding = v_padding.view(B, self.num_heads, C // self.num_heads,
                                   H + self.window_size[0] - 1, W + self.window_size[1] - 1).unsqueeze(3)
        # Batch, num_heads, dim//num_heads, 1, Hp, Wp

        v = torch.cat(
            [v_padding[:, :, :, :, i // self.window_size[0]:i // self.window_size[0] + H,
             i % self.window_size[0]:i % self.window_size[0] + H]
             for i in range(self.window_size[0] * self.window_size[1])
             ], dim=3
        )

        x = torch.sum(v * attentionMap.unsqueeze(2), dim=3).reshape(B, C, H, W)
        # print(torch.max(x - v.reshape(B, C, H, W)))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    at = convAttention().cuda()
    a = torch.rand(8, 96, 24, 24).cuda()
    summary(at, (96, 48, 48), batch_size=8)
    # b = at(a)
    # print(b.shape)


