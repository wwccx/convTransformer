import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as t
align = False
a0 = torch.ones(9, 1, 5, 5)  # B 2C H W
# trans = torch.zeros(8, 2, 1)
# trans[:, 1, :] = 4/24
# theta = torch.cat((torch.eye(2).unsqueeze(0).expand(8, -1, -1), trans), dim=2)
# grid = F.affine_grid(theta, size=[8, 2, 24, 24], align_corners=align)
# a = F.grid_sample(a0, grid, align_corners=align, mode='nearest')
# # print(torch.max(a-a0))
# print(a[0, :, :, :])



window_size = (3, 3)
W = 5
H = 5
scale = torch.tensor([W, H])
pos = torch.tensor([window_size[1]//2, window_size[0]//2])
theta = torch.eye(2).unsqueeze(0).expand(window_size[0]*window_size[1], -1, -1)

trans = torch.stack(
    torch.meshgrid([torch.arange(window_size[0]), torch.arange(window_size[1])])[::-1]
)   # 2, w, w
print(trans.flatten(1).permute(1, 0) - pos)
trans = (trans.flatten(1).permute(1, 0) - pos) / scale * 2
print(theta.shape, trans.shape)
theta = torch.cat([theta, trans.unsqueeze(2)], dim=2)

grid = F.affine_grid(theta, a0.shape, align_corners=False)
a = F.grid_sample(a0, grid, mode='nearest', align_corners=False)
print(a)