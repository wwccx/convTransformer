import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as t
# h = 2
# w = 3
#
# coords_h = torch.arange(h)
# coords_w = torch.arange(w)
# coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
# print('coords:\n', coords)
# coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
# print('coords_flatten:\n', coords_flatten)
# relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
# relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
# print('relative coords:\n', relative_coords, '\n', relative_coords[:, :, 0])
# # relative_coords[:, :, 0] += h - 1  # shift to start from 0
# # relative_coords[:, :, 1] += w - 1
# relative_coords[:, :, 0] *= 2 * w - 1
# print('relative coords:\n', relative_coords)
# print(relative_coords[:, :, 0], relative_coords[:, :, 1])
# relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
# print('relative position index:\n', relative_position_index.view(-1))
# a = torch.nn.Conv2d(5, 10, 3)
# b = torch.rand(2, 1, 5, 10, 10)
# print(a(b))

# image = torch.arange(25).view(5, 5).unsqueeze(0).unsqueeze(0).repeat((9, 1, 1, 1)).float()
# image.index_select
# image = image.flatten(0, 1)
# print(image.shape)
# coords_h = torch.arange(3)
# coords_w = torch.arange(3)
#
# coords = torch.stack(torch.meshgrid([coords_h, coords_w])).view(2, -1).unsqueeze(2).permute(1, 0, 2)  # 2, Wh, Ww
# print(coords)
# coords[:, 0, :] = -1
# coords[:, 1, :] = 0
# trans_mat = torch.cat(
#             [
#                 torch.eye(2).repeat((9, 1, 1)),
#                 coords
#             ],
#             dim=2
#             )
# grid = F.affine_grid(trans_mat, image.shape, align_corners=False)
# img = F.grid_sample(image, grid, align_corners=False)
# print(img)
# a = t.functional.affine(img)
img = torch.arange(25).view(5, 5).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
v_padding = nn.ConstantPad2d(1, 0)
img_p = v_padding(img)
# img = torch.roll(img_p, )
# print(img_p)
# print(img_p.size())
# print(torch.argmax(img_p))
print(img_p)
a = torch.cat(
            [img_p[:, :, i//3:i//3+3, (i % 3):((i % 3)+3)]
             for i in range(9)
             ], dim=1
        )
print(a)
# img = torch.take(img_p, torch.tensor([[1, 2, 3], [10, 12, 18]]))
# print(img)

