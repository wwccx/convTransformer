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

        split_file = os.path.join(self.dataset_dir, 'splits', 'image_wise', pattern+'_indices.npz')
        self.index_data = np.load(split_file)['arr_0']
        self.num_datapoints = len(self.index_data)

        self.grasps_files = glob.glob(os.path.join(self.dataset_dir, 'tensors', 'grasps*.npz'))
        self.grasps_files.sort()
        #self.grasps_files = self.grasps_files[0:2000]

        self.img_files = glob.glob(os.path.join(self.dataset_dir, 'tensors', 'tf_depth_ims*.npz'))
        self.img_files.sort()
        #self.img_files = self.img_files[0:2000]

        self.metrics_files = glob.glob(os.path.join(self.dataset_dir, 'tensors', 'grasp_metrics_?????.npz'))
        self.metrics_files.sort()
        #self.metrics_files = self.metrics_files[0:2000]

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
        #depth = depth.astype(np.float32)
        # img = img.transpose((0, 3, 1, 2))
        img = ((img.transpose((0, 3, 1, 2)) - self.img_mean)/self.img_std).astype(np.float32)
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
        mask_idx_second_dim = torch.div(angle0, (np.pi / angular_bins), rounding_mode='floor').to(torch.long)
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim] = 1
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim + 1] = 1

        # angle *= -1
        # angle += np.pi/2
        # small_ang = np.where(angle < np.pi / 2)
        # big_ang = np.where(angle > np.pi / 2)
        # angle[small_ang] = np.pi / 2 - angle[small_ang]
        # angle[big_ang] = np.pi * 3 / 2 - angle[small_ang]
        # mask = np.zeros((self.batch_size, angular_bins * 2))
        # angle = (angle // (np.pi/angular_bins)).astype(np.int)
        # metric_angle = np.zeros_like(metric)
        # metric_angle[np.where((6 <= angle) * (angle <= 9))] = 1
        # metric = metric * metric_angle.astype(np.int)
        # mask[np.arange(self.batch_size), np.int32(angle // (np.pi / angular_bins)) * 2] = 1
        # mask[np.arange(self.batch_size), np.int32(angle // (np.pi / angular_bins)) * 2 + 1] = 1

        return img, torch.from_numpy(depth), torch.from_numpy(metric), mask

    def __len__(self):
        return int(len(self.grasps_files) * self.index_split * 100 / self.batch_size) if self.dataset_partten == 'train' \
            else int(len(self.grasps_files) * (1 - self.index_split) * 100 / self.batch_size)


class GGSCNNGraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train'):
        self.dataset_dir = os.path.join(dataset_dir, pattern)
        a = '*'
        self.grasps_files = glob.glob(os.path.join(self.dataset_dir, 'poseTensor*.npz'))
        self.grasps_files.sort()

        self.img_files = glob.glob(os.path.join(self.dataset_dir, 'imgTensor*.npz'))
        self.img_files.sort()

        self.metrics_files = glob.glob(os.path.join(self.dataset_dir, 'metricTensor*.npz'))
        self.metrics_files.sort()
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        assert len(self.grasps_files) == len(self.img_files) == len(self.metrics_files),\
            f'number of grasp files not match, grasp: {len(self.grasps_files)}, img: {len(self.img_files)}, metric: {len(self.metrics_files)}'
        # self.grid = F.affine_grid()
        self.mask_idx_zero_dim = torch.arange(8).to(self.device)
        self.ANGLE_INTERVAL = np.pi / 16

    def __len__(self):
        return len(self.grasps_files)

    def __getitem__(self, item):
        grasp = torch.from_numpy(np.load(self.grasps_files[item])['arr_0']).squeeze().to(torch.float32)
        img = torch.from_numpy(np.load(self.img_files[item])['arr_0']).squeeze().unsqueeze(1).to(torch.float32)
        metric = torch.from_numpy(np.load(self.metrics_files[item])['arr_0']).squeeze().to(torch.long)
        angle = torch.rand(8).to(self.device) * np.pi
        mask = torch.zeros(8, 32).to(self.device)
        affine_matrix = torch.zeros((8, 2, 3)).to(self.device)
        affine_matrix[:, 0, 0] = torch.cos(angle)
        affine_matrix[:, 0, 1] = torch.sin(angle)
        affine_matrix[:, 1, 0] = -torch.sin(angle)
        affine_matrix[:, 1, 1] = torch.cos(angle)
        
        grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, align_corners=False, padding_mode='border')
        img_mean = torch.mean(img.flatten(2), dim=2, keepdim=True).unsqueeze(-1)
        img_std = torch.std(img.flatten(2), dim=2, keepdim=True).unsqueeze(-1)
        mask_idx_second_dim = torch.round(angle // self.ANGLE_INTERVAL).to(torch.long)
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim] = 1
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim + 1] = 1

        return (img - img_mean)/img_std, (grasp - img_mean.squeeze()) / img_std.squeeze(), metric, mask


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

        mask_idx_second_dim = torch.div(angle,  self.ANGLE_INTERVAL, rounding_mode='floor').to(torch.long)
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim] = 1
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim + 1] = 1

        return (img - self.img_mean)/self.img_std, (grasp - self.img_mean) / self.img_std, metric, mask
        # return img.mean(), img.std(), grasp.mean(), grasp.std()


class MixupGraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', batch_size=64, add_noise=False):
        self.parallel_data = GraspDataset(os.path.join(dataset_dir, 'fc_parallel_jaw'),
                                          pattern, batch_size, add_noise=add_noise)
        self.bullet_data = GGSCNNGraspDatasetZip(os.path.join(dataset_dir, 'grasp'),
                                                 pattern, batch_size, add_noise=add_noise)
        self.len_para = len(self.parallel_data)
        self.len_bullet = len(self.bullet_data)

    def __len__(self):
        return self.len_para + self.len_bullet
    
    def __getitem__(self, item):
        if item < self.len_para:
            return self.parallel_data[item]
        else:
            return self.bullet_data[item - self.len_para]


class GraspLossFunction(torch.nn.Module):
    def __init__(self, config):
        super(GraspLossFunction, self).__init__()
        # self.loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([5., 1.]).cuda())
        if not config.DATA.MIXUP_ON:
            self.loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor(config.DATA.LOSS_WEIGHT).cuda())
        else:
            self.loss_function = SoftTargetCrossEntropy()

    def forward(self, inputs, target, mask):
        inputs = inputs.squeeze()
        valid_input = inputs[torch.where(mask > 0)].view(-1, 2)
        # target = target * (torch.sum(mask[:, 12:18], dim=1) > 0).to(torch.long)
        return self.loss_function(valid_input, target)


if __name__ == '__main__':
    # dataset = GraspDataset('/home/server/convTransformer/data/fc_parallel_jaw',
    #         batch_size=64, add_noise=False)
    d = GGSCNNGraspDataset('/home/wangchuxuan/PycharmProjects/grasp/output/tensor')
    dataset = GraspDataset('/home/server/convTransformer/data/parallel_jaw',
            batch_size=64, add_noise=False)
    import cv2
    import torch.utils.data as D
    from matplotlib import pyplot as plt
    t = D.DataLoader(dataset, batch_size=1, shuffle=False)
    for img, pose, metric, mask in t:
        img = img.flatten(0, 1)
        angle = mask.flatten(0, 1).squeeze()
        angle = torch.argmax(angle, dim=1) // 2 * np.pi / 16
        pose = pose.flatten(0, 1).squeeze()
        metric = metric.flatten(0, 1).squeeze()
        x, y, grasp_radius = 48, 48, 48
        for idx in range(64):
            image = img[idx, 0, ...].numpy()
            theta = angle[idx].item()
            print(theta/np.pi*180)
            p = pose[idx].item()
            label  = metric[idx].item()
            x1 = x + grasp_radius * np.cos(theta)
            y1 = y + grasp_radius * np.sin(theta)
            x2 = x + grasp_radius * np.cos(theta + np.pi)
            y2 = y + grasp_radius * np.sin(theta + np.pi)
            depth_img = cv2.arrowedLine(image, (int(x1), int(y1)), (int(x2), int(y2)), p, thickness=2)
            depth_img = cv2.arrowedLine(depth_img, (int(x2), int(y2)), (int(x1), int(y1)), p, thickness=2)
            plt.imshow(depth_img)
            plt.title('depth: {:.3f}, metric: {:1d}'.format(p, label))
            plt.show()


    '''
    # dataset = GGSCNNGraspDatasetZip('/home/server/convTransformer/data/grasp')
    # dataset = MixupGraspDataset('/home/server/convTransformer/data', batch_size=64, pattern='train')
    # print(len(dataset))
    import torch.utils.data as D
    import time
    train_data = D.DataLoader(dataset, batch_size=256//64, shuffle=True, num_workers=12)
    # print(len(train_data))
    num = 0
    t0 = time.time()
    total = 0
    batch_idx = 0
    img_mean = 0
    img_std = 0
    pos_mean = 0
    pos_std = 0
    for i_m, i_s, p_m, p_s in train_data:
        # print(i_m.shape)
        img_mean = (img_mean * batch_idx + i_m.mean()) / (batch_idx + 1)
        img_std = (img_std * batch_idx + i_s.mean()) / (batch_idx + 1)
        pos_mean = (pos_mean * batch_idx + p_m.mean()) / (batch_idx + 1)
        pos_std = (pos_std * batch_idx + p_s.mean()) / (batch_idx + 1)
        print(img_mean, img_std, pos_mean, pos_std)
        batch_idx += 1



    # for img, pose, metric, mask, mean, std in train_data:
    #     num += torch.sum(metric)
    #     total += 256
    # print(num, total)
    '''
