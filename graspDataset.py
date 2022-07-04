import torch
import torch.utils.data
import os
import numpy as np
import glob
import torch.nn.functional as F


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', batch_size=32, index_split=0.9):
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

        self.dataset_partten = pattern
        self.index_split = index_split
        self.batch_size = batch_size
        self.len_files = len(self.grasps_files)
        self.img_std = np.load(os.path.join(dataset_dir, 'im_std.npy'))
        self.img_mean = np.load(os.path.join(dataset_dir, 'im_mean.npy'))
        self.pose_std = np.load(os.path.join(dataset_dir, 'pose_std.npy'))
        self.pose_mean = np.load(os.path.join(dataset_dir, 'pose_mean.npy'))

    def __getitem__(self, item):
        item = item % self.len_files
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
        # depth = ((depth - self.pose_mean) / self.pose_std).astype(np.float32)
        depth = depth.astype(np.float32)
        img = img.transpose((0, 3, 1, 2))
        # img = ((img.transpose((0, 3, 1, 2)) - self.img_mean)/self.img_std).astype(np.float32)
        # img -= np.expand_dims(depth, (1, 2, 3))
        metric = (metric > 0.7).astype(np.long)
        flag = np.ones_like(angle)
        flag[np.where(angle < 0)] *= -1
        angle = np.abs(angle) % np.pi * flag
        angle[np.where(angle > np.pi/2)] -= np.pi
        angle[np.where(angle < -np.pi/2)] += np.pi
        angle *= -1
        angle += np.pi/2
        mask = np.zeros((self.batch_size, angular_bins * 2))
        # angle = (angle // (np.pi/angular_bins)).astype(np.int)
        # metric_angle = np.zeros_like(metric)
        # metric_angle[np.where((6 <= angle) * (angle <= 9))] = 1
        # metric = metric * metric_angle.astype(np.int)
        mask[np.arange(self.batch_size), np.int32(angle // (np.pi / angular_bins)) * 2] = 1
        mask[np.arange(self.batch_size), np.int32(angle // (np.pi / angular_bins)) * 2 + 1] = 1

        return torch.from_numpy(img), torch.from_numpy(depth), torch.from_numpy(metric), torch.from_numpy(mask)

    def __len__(self):
        return int(len(self.grasps_files) * self.index_split * 100 / self.batch_size) if self.dataset_partten == 'train' \
            else int(len(self.grasps_files) * (1 - self.index_split) * 100 / self.batch_size)


class GGSCNNGraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train'):
        if pattern == 'train':
            self.dataset_dir = dataset_dir
        elif pattern == 'validation':
            self.dataset_dir = dataset_dir + '_validation'
        else:
            raise ValueError('pattern not supported')
        a = '*'
        self.grasps_files = glob.glob(os.path.join(self.dataset_dir, a, 'tensor', 'poseTensor_*.npz'))
        self.grasps_files.sort()

        self.img_files = glob.glob(os.path.join(self.dataset_dir, a, 'tensor', 'imgTensor_*.npz'))
        self.img_files.sort()

        self.metrics_files = glob.glob(os.path.join(self.dataset_dir, a, 'tensor', 'metricTensor_*.npz'))
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
    def __init__(self, dataset_dir, pattern='train', batch_size=64):
        self.dataset_dir = os.path.join(dataset_dir, pattern)
        self.files = glob.glob(os.path.join(self.dataset_dir, 'Tensor_*.npz'))
        self.files.sort()
        self.len_files = len(self.files)
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.mask_idx_zero_dim = torch.arange(batch_size).to(self.device)
        self.ANGLE_INTERVAL = np.pi / 16

    def __len__(self):
        return len(self.files) * (256 // self.batch_size)

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

        img_mean = torch.mean(img.flatten(2), dim=2, keepdim=True).unsqueeze(-1)
        img_std = torch.std(img.flatten(2), dim=2, keepdim=True).unsqueeze(-1)

        mask_idx_second_dim = torch.div(angle,  self.ANGLE_INTERVAL, rounding_mode='floor').to(torch.long)
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim] = 1
        mask[self.mask_idx_zero_dim, 2 * mask_idx_second_dim + 1] = 1

        return (img - img_mean)/img_std, (grasp - img_mean.squeeze()) / img_std.squeeze(), metric, mask


class MixupGraspDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pattern='train', batch_size=64):
        self.parallel_data = GraspDataset(os.path.join(dataset_dir, 'parallel_jaw'), pattern, batch_size)
        self.bullet_data = GGSCNNGraspDatasetZip(os.path.join(dataset_dir, 'grasp'), pattern, batch_size)
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
    def __init__(self):
        super(GraspLossFunction, self).__init__()
        # self.loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([5., 1.]).cuda())
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, target, mask):
        inputs = inputs.squeeze()
        valid_input = inputs[torch.where(mask > 0)].view(-1, 2)
        # target = target * (torch.sum(mask[:, 12:18], dim=1) > 0).to(torch.long)
        return self.loss_function(valid_input, target)


if __name__ == '__main__':
    # dataset = GGSCNNGraspDataset('/home/server/convTransformer/data')
    dataset = MixupGraspDataset('/home/server/convTransformer/data', batch_size=64, pattern='train')
    print(len(dataset))
    import torch.utils.data as D
    import time
    train_data = D.DataLoader(dataset, batch_size=256//64, shuffle=True, num_workers=12)
    # print(len(train_data))
    num = 0
    t0 = time.time()
    total = 0
    for img, pose, metric, mask in train_data:
        num += torch.sum(metric)
        total += 256
        print(img.shape)
        # if num % 20 == 0:
        #     print('{:2f}'.format(num/train_data.__len__()))
        #     print((time.time() - t0) / num)
        #print(mask.shape)
        #print(metric.shape)
    print(num, total)
    # num = 0
    # l = len(dataset.img_files)
    # for i in range(15912, 17000):
    #     print(dataset.img_files[i])
    #     a = np.load(dataset.img_files[i])['arr_0']
    #     print(i)


