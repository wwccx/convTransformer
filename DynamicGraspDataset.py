import torch
import torch.utils.data
import os
import numpy as np
import glob
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', batch_size=32, index_split=0.9, add_noise=False):
        self.dataset_dir = dataset_dir

        split_file = os.path.join(self.dataset_dir, 'splits', 'image_wise', pattern + '_indices.npz')
        self.index_data = np.load(split_file)['arr_0']
        self.num_datapoints = len(self.index_data)

        self.grasps_files = glob.glob(os.path.join(self.dataset_dir, 'tensors', 'grasps*.npz'))
        self.grasps_files.sort()
        # self.grasps_files = self.grasps_files[0:2000]

        self.img_files = glob.glob(os.path.join(self.dataset_dir, 'tensors', 'tf_depth_ims*.npz'))
        self.img_files.sort()
        # self.img_files = self.img_files[0:2000]

        self.metrics_files = glob.glob(os.path.join(self.dataset_dir, 'tensors', 'grasp_metrics_?????.npz'))
        self.metrics_files.sort()
        # self.metrics_files = self.metrics_files[0:2000]

        self.current_file_index = 0

        self.grasp = np.load(self.grasps_files[
                                 self.current_file_index])['arr_0']
        self.img = np.load(self.img_files[
                               self.current_file_index])['arr_0']
        self.metric = np.load(self.metrics_files[
                                  self.current_file_index])['arr_0']
        self.add_noise = add_noise
        self.dataset_partten = pattern
        self.index_split = index_split
        self.batch_size = batch_size
        self.len_files = len(self.grasps_files)
        self.mask_idx_zero_dim = torch.arange(batch_size)
        self.img_std = np.load(os.path.join(dataset_dir, 'im_std.npy'))
        self.img_mean = np.load(os.path.join(dataset_dir, 'im_mean.npy'))
        self.pose_std = np.load(os.path.join(dataset_dir, 'pose_std.npy'))
        self.pose_mean = np.load(os.path.join(dataset_dir, 'pose_mean.npy'))

    def __getitem__(self, item):
        item = item % (self.len_files - 1)
        angular_bins = 16
        if item == 15913:
            item = 15912
        grasp = np.load(self.grasps_files[item])['arr_0']
        img = np.load(self.img_files[item])['arr_0']
        metric = np.load(self.metrics_files[item])['arr_0']
        index_range = int(img.shape[0] * self.index_split)
        if self.dataset_partten == 'train':
            choice_index = np.random.choice(index_range, self.batch_size, replace=False)
        else:
            choice_index = np.random.choice(img.shape[0] - index_range, self.batch_size) + index_range
        grasp = grasp[choice_index]
        img = img[choice_index]
        metric = metric[choice_index]
        angle = grasp[:, 3]
        depth = grasp[:, 2]

        depth = ((depth - self.img_mean) / self.img_std).astype(np.float32)
        # depth = depth.astype(np.float32)
        # img = img.transpose((0, 3, 1, 2))
        img = ((img.transpose((0, 3, 1, 2)) - self.img_mean) / self.img_std).astype(np.float32)
        # img -= np.expand_dims(depth, (1, 2, 3))
        metric = (metric > 0.5).astype(np.long)
        flag = np.ones_like(angle)
        flag[np.where(angle < 0)] *= -1
        angle = np.abs(angle) % np.pi * flag
        # angle[np.where(angle > np.pi/2)] -= np.pi
        # angle[np.where(angle < -np.pi/2)] += np.pi
        angle[np.where(angle < 0)] += np.pi
        angle0 = torch.rand(self.batch_size) * np.pi
        angle = angle0 - torch.from_numpy(angle)
        mask = torch.zeros(self.batch_size, 32)

        affine_matrix = torch.zeros((self.batch_size, 2, 3))
        affine_matrix[:, 0, 0] = torch.cos(angle)
        affine_matrix[:, 0, 1] = torch.sin(angle)
        affine_matrix[:, 1, 0] = -torch.sin(angle)
        affine_matrix[:, 1, 1] = torch.cos(angle)
        grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
        img = F.grid_sample(torch.from_numpy(img), grid, align_corners=False, padding_mode='border')
        if self.add_noise:
            img += torch.randn(img.shape) * 0.03 * torch.rand(1)
        mask_idx_second_dim = torch.floor(angle0/(np.pi / angular_bins)).to(torch.long)
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim] = 1
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim + 1] = 1

        return img, torch.from_numpy(depth), torch.from_numpy(metric), mask

    def __len__(self):
        return int(len(self.grasps_files) * self.index_split * 100 / self.batch_size) if self.dataset_partten == 'train' \
            else int(len(self.grasps_files) * (1 - self.index_split) * 100 / self.batch_size)


class GGSCNNGraspDatasetZip(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', batch_size=64, add_noise=False):
        self.dataset_dir = os.path.join(dataset_dir, pattern)
        self.files = glob.glob(os.path.join(self.dataset_dir, 'Tensor_*.npz'))
        self.files.sort()
        self.len_files = len(self.files)
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.mask_idx_zero_dim = torch.arange(batch_size).to(self.device)
        self.ANGLE_INTERVAL = np.pi / 16
        self.img_mean = 0.4291
        self.img_std = 0.0510
        self.add_noise = add_noise

    def __len__(self):
        return len(self.files) * (64 // self.batch_size)

    def __getitem__(self, item):
        item = item % (self.len_files)
        tensor = np.load(self.files[item])
        choice = torch.randperm(256)[:self.batch_size]
        grasp = torch.from_numpy(tensor['pose']).squeeze().to(torch.float32)[choice, ...]
        img = torch.from_numpy(tensor['img']).squeeze().unsqueeze(1).to(torch.float32)[choice, ...]
        metric = torch.from_numpy(tensor['metric']).squeeze().to(torch.long)[choice, ...]
        angle = torch.rand(self.batch_size).to(self.device) * np.pi
        mask = torch.zeros(self.batch_size, 32).to(self.device)

        affine_matrix = torch.zeros((self.batch_size, 2, 3)).to(self.device)
        affine_matrix[:, 0, 0] = torch.cos(angle)
        affine_matrix[:, 0, 1] = torch.sin(angle)
        affine_matrix[:, 1, 0] = -torch.sin(angle)
        affine_matrix[:, 1, 1] = torch.cos(angle)

        grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, align_corners=False, padding_mode='border')
        if self.add_noise:
            img += torch.randn(img.shape) * 0.03 * torch.rand(1)
        # img_mean = torch.mean(img.flatten(2), dim=2, keepdim=True).unsqueeze(-1)
        # img_std = torch.std(img.flatten(2), dim=2, keepdim=True).unsqueeze(-1)

        mask_idx_second_dim = torch.div(angle, self.ANGLE_INTERVAL, rounding_mode='floor').to(torch.long)
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim] = 1
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim + 1] = 1

        return (img - self.img_mean) / self.img_std, (grasp - self.img_mean) / self.img_std, metric, mask
        # return img.mean(), img.std(), grasp.mean(), grasp.std()


class MixupGraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', batch_size=64, add_noise=False, time_slices=4):
        self.parallel_data = GraspDataset(os.path.join(dataset_dir, 'fc_parallel_jaw'),
                                          pattern, batch_size, add_noise=add_noise)
        self.bullet_data = GGSCNNGraspDatasetZip(os.path.join(dataset_dir, 'grasp'),
                                                 pattern, batch_size, add_noise=add_noise)
        self.len_para = len(self.parallel_data)
        self.len_bullet = len(self.bullet_data)
        self.batch_size = batch_size
        self.time_slices = time_slices
        self.trajectory = TrajectoryGenerator(time_slices)

    def __len__(self):
        return self.len_para + self.len_bullet

    def __getitem__(self, item):
        if item < self.len_para:
            img, depth, metric, mask = self.parallel_data[item]
        else:
            img, depth, metric, mask = self.bullet_data[item - self.len_para]

        img = img.repeat(1, self.time_slices, 1, 1).unsqueeze(2)
        velocity = torch.zeros(self.batch_size, 2)
        angle = torch.rand(self.batch_size) * np.pi * 2
        velocity[:, 0] = torch.cos(angle)
        velocity[:, 1] = torch.sin(angle)
        velocity *= torch.rand(self.batch_size, 1) * 2
        affine_matrix, target_pos = self.trajectory(velocity)
        img = img.flatten(0, 1)

        grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, align_corners=False, padding_mode='border')
        img = img.squeeze(2).reshape(self.batch_size, self.time_slices, img.shape[-2], img.shape[-1])

        return img, depth, metric, mask, target_pos


class DynamicGraspLossFunction(torch.nn.Module):
    def __init__(self, config):
        super(DynamicGraspLossFunction, self).__init__()
        # self.loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([5., 1.]).cuda())
        if not config.DATA.MIXUP_ON:
            self.classify_loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(config.DATA.LOSS_WEIGHT).cuda())
        else:
            self.classify_loss_function = SoftTargetCrossEntropy()
        # self.regress_loss_function = torch.nn.L1Loss()
        self.regress_loss_function = torch.nn.MSELoss()

    def forward(self, classification, position, target, mask, target_pos):
        classification = classification.squeeze()
        position = position.squeeze()
        valid_input = classification[torch.where(mask > 0)].view(-1, 2)
        return self.classify_loss_function(valid_input, target) + self.regress_loss_function(position, target_pos)


class TrajectoryGenerator:
    def __init__(self, time_slices):
        self.time_slices = time_slices

    def __call__(self, velocity, angular_velocity=None, *args, **kwargs):
        """

        :param velocity: (batch_size, 2)  ((x, y), ...), we suppose the velocity is constant of each object
                         make sure x^2 + y^2 <= 1
        :param angular_velocity: (batch_size, 1) ((theta), ...), and so as the angular velocity
        :param args:
        :param kwargs:
        :return:
        """

        batch_size = velocity.shape[0]
        time_step = torch.arange(self.time_slices).float() / self.time_slices - 0.5
        affine_matrix = torch.zeros((batch_size, self.time_slices, 2, 3))
        if angular_velocity is None:
            angular_velocity = torch.zeros((batch_size, 1))
        affine_matrix[:, :, 0, 0] = torch.cos(angular_velocity)
        affine_matrix[:, :, 0, 1] = torch.sin(angular_velocity)
        affine_matrix[:, :, 1, 0] = -torch.sin(angular_velocity)
        affine_matrix[:, :, 1, 1] = torch.cos(angular_velocity)
        affine_matrix[:, :, 0, 2] = velocity[:, 0:1] * time_step
        affine_matrix[:, :, 1, 2] = velocity[:, 1:2] * time_step
        # if angular_velocity is None:
        #     return affine_matrix.flatten(0, 1), velocity * 0.5
        return affine_matrix.flatten(0, 1), velocity * 0.5


if __name__ == '__main__':
    import cv2
    import torch.utils.data as D
    from matplotlib import pyplot as plt
    batch_size = 8
    time_slices = 6
    dataset = MixupGraspDataset('./data/', batch_size=64,
                                add_noise=False, time_slices=time_slices)
    t = D.DataLoader(dataset, batch_size=1, shuffle=False)
    for img, pose, metric, mask, tar_pos in t:
        img = img.flatten(0, 1)
        angle = mask.flatten(0, 1).squeeze()
        angle = torch.argmax(angle, dim=1) // 2 * np.pi / 16
        pose = pose.flatten(0, 1).squeeze()
        metric = metric.flatten(0, 1).squeeze()
        tar_pos = tar_pos.flatten(0, 1).squeeze()
        for j in range(batch_size):
            for i in range(time_slices):
                plt.subplot(time_slices, 1, i + 1)
                plt.imshow(img[j, i, :, :].detach().numpy())
            print(tar_pos[j])
            plt.show()
