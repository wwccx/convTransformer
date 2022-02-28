import torch
import torch.nn as nn


class convAttention(nn.Module):
    def __init__(self, winSize=7, dim=96, num_heads=3, attn_drop=0.):
        super(convAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.windowSize = winSize
        self.convAttention = torch.nn.Conv2d(dim // num_heads * 2, winSize * winSize, kernel_size=(winSize, winSize),
                                             stride=1, padding=(winSize // 2, winSize // 2))
        self.softmax = nn.Softmax(dim=1)
        self.scale = (dim // num_heads) ** -0.5

        self.proj = torch.nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_drop = torch.nn.Dropout2d(p=attn_drop)

        # TODO: position encoding

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W).permute(1, 0, 2, 3, 4, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Batch, num_heads, dim//num_heads, H, W
        attentionMap = self.convAttention(
            torch.cat(
                (torch.flatten(q * self.scale, start_dim=0, end_dim=1), torch.flatten(q, start_dim=0, end_dim=1)), dim=1
            )
        )
        attentionMap = self.softmax(attentionMap).reshape(B, self.num_heads, self.windowSize ** 2, H, W).unsqueeze(2)
        # Batch, num_heads, w*w, H, W

        x = torch.sum(v.unsqueeze(3) * attentionMap, dim=3).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    at = convAttention()
    a = torch.rand(8, 96, 24, 24)
    b = at(a)
    print(b.shape)