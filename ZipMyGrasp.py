import numpy as np
import glob
import os


dataset_dir = '/home/server/grasp1/virtual_grasp/fine_tune'
pattern = 'validation'
if pattern == 'train':
    dataset_dir = dataset_dir
elif pattern == 'validation':
    dataset_dir = dataset_dir + '_validation'
a = '*'
grasps_files = glob.glob(os.path.join(dataset_dir, a, 'tensor', 'poseTensor_*.npz'))
grasps_files.sort()

img_files = glob.glob(os.path.join(dataset_dir, a, 'tensor', 'imgTensor_*.npz'))
img_files.sort()

metrics_files = glob.glob(os.path.join(dataset_dir, a, 'tensor', 'metricTensor_*.npz'))
metrics_files.sort()

assert len(grasps_files) == len(img_files) == len(metrics_files), f'number of grasp files not match, grasp: {len(self.grasps_files)}, img: {len(self.img_files)}, metric: {len(self.metrics_files)}'

out_dir = '/home/server/convTransformer/data/grasp'

out_dir = os.path.join(out_dir, 'val')

os.makedirs(out_dir, exist_ok=True)

for item in range(len(grasps_files) // 32):
    grasp = np.concatenate([np.load(grasps_files[item * 32 + x])['arr_0'].squeeze() for x in range(32)], axis=0)
    img = np.concatenate([np.load(img_files[item * 32 + x])['arr_0'].squeeze()[:, None, :, :] for x in range(32)], axis=0)
    metric = np.concatenate([np.load(metrics_files[item * 32 + x])['arr_0'].squeeze() for x in range(32)], axis=0)
    
    np.savez(os.path.join(out_dir, 'Tensor_{:0>4d}.npz'.format(item)), img=img, pose=grasp, metric=metric)
    print(os.path.join(out_dir, 'Tensor_{:0>4d}.npz'.format(item)))

