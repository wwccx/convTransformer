import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as t
from matplotlib import pyplot as plt
from DynamicGraspDataset import TrajectoryGenerator
import numpy as np


time_slices = 4
batch_size = 8
trajectory = TrajectoryGenerator(time_slices)
img = torch.randn(8, 1, 96, 96)
img[:, :, 46:50, 46:50] = 3
img = img.repeat(1, time_slices, 1, 1).unsqueeze(2)
velocity = torch.zeros(batch_size, 2)
angle = torch.rand(batch_size) * np.pi
velocity[:, 0] = torch.cos(angle)
velocity[:, 1] = torch.sin(angle)
velocity *= torch.rand(batch_size, 1) * 2
affine_matrix, target_pos = trajectory(velocity)
img = img.flatten(0, 1)
grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
img_new = F.grid_sample(img, grid, align_corners=False, padding_mode='zeros')
img_new = img_new.squeeze(2).reshape(batch_size, time_slices, img.shape[-2], img.shape[-1])
img = img.squeeze(2).reshape(batch_size, time_slices, img.shape[-2], img.shape[-1])

for j in range(batch_size):
    for i in range(time_slices):
        plt.subplot(time_slices, 2, 2*i + 1)
        plt.imshow(img[j, i, :, :].detach().numpy())
        plt.subplot(time_slices, 2, 2*i + 2)
        plt.imshow(img_new[j, i, :, :].detach().numpy())
    plt.show()