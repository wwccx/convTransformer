import torch
import torch.utils.data
import os
import numpy as np
import glob


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

        return img, depth, metric, mask

    def __len__(self):
        return int(len(self.grasps_files) * self.index_split * 100 / self.batch_size) if self.dataset_partten == 'train' \
            else int(len(self.grasps_files) * (1 - self.index_split) * 100 / self.batch_size)


class GraspLossFunction(torch.nn.Module):
    def __init__(self):
        super(GraspLossFunction, self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 20.]).cuda())

    def forward(self, inputs, target, mask):
        inputs = inputs.squeeze()
        valid_input = inputs[torch.where(mask > 0)].view(-1, 2)
        # target = target * (torch.sum(mask[:, 12:18], dim=1) > 0).to(torch.long)
        return self.loss_function(valid_input, target)


if __name__ == '__main__':
    dataset = GraspDataset('/home/server/library/parallel_jaw',
            batch_size=8, pattern='val')
    import torch.utils.data as D
    import time
    train_data = D.DataLoader(dataset, batch_size=128//8, num_workers=12)
    print(train_data.__len__())
    num = 0
    t0 = time.time()
    for img, metric, mask in train_data:
        num += torch.sum(metric)
        #print(img.shape)
        # if num % 20 == 0:
        #     print('{:2f}'.format(num/train_data.__len__()))
        #     print((time.time() - t0) / num)
        #print(mask.shape)
        #print(metric.shape)
    print(num)
    # num = 0
    # l = len(dataset.img_files)
    # for i in range(15912, 17000):
    #     print(dataset.img_files[i])
    #     a = np.load(dataset.img_files[i])['arr_0']
    #     print(i)


