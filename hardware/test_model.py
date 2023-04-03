import sys
sys.path.append("/home/server/convTransformer/")
import torch
import cv2
import numpy as np
from load_model import build_model
import os
import utils as u
import time
from matplotlib import pyplot as plt
from camera import RS
from utils import in_paint
from torchsummary import summary


class PredictModel():
    def __init__(self, model_path, k=-1):
        print(os.path.join(model_path, 'config.pth'))
        model_config = torch.load(os.path.join(model_path, 'config.pth'))['config']

        print(model_config)
        pths = [i for i in os.listdir(model_path) if i.endswith('.pth')]
        pths.sort(key=lambda x: (os.path.getmtime(os.path.join(model_path, x))))

        # k = -1  # candidate: epoch 41, k = 42 epoch55 66
        self.prediction_model = build_model(model_config).cuda()
        self.prediction_model.load_state_dict(
            torch.load(
                os.path.join(model_path, pths[k])
            )['model']
        )
        print('load model:', pths[k])
        self.prediction_model.eval()
        # self.prediction_model.to('cuda')
        self.camera = RS(640, 480)

        self.tensor = torch.ones(1, 6, 480, 640).cuda()
        # self.tensor = torch.ones(1, 6, 300, 300).cuda()
        for i in range(6):
            self.update_tensor()
     
    @torch.no_grad()
    def prediction(self, depth_bin=8):
        img_tensor = self.tensor
        # depth_reso = (torch.max(img_tensor) - torch.min(img_tensor)) / depth_bin
        depth_reso = 1e-2
        z_pose_init = torch.arange(depth_bin).cuda() * depth_reso + torch.min(img_tensor)
        print(torch.min(img_tensor))

        # img_tensor = (img_tensor - torch.mean(img_tensor[:, ...])) / torch.std(img_tensor[:, ...])
        # z_pose = (z_pose_init - torch.mean(img_tensor[:, ...])) / torch.std(img_tensor[:, ...])
        img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor)) - 0.5
        z_pose = (z_pose_init - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor)) - 0.5
        
        if self.prediction_model.dynamic:
            res, pos = self.prediction_model(img_tensor, z_pose)
        else:
            res = self.prediction_model(img_tensor[:, 3:4, ...], z_pose)
            pos = torch.zeros_like(res)[:, :2, ...]
        s = list(res.shape)
        s[1:2] = [16, 2]
        sf = torch.nn.Softmax(dim=2)
        res = sf(res.view(*s))[:, :, 1, ...]
        # s[2:3] = []

        index = self.get_tensor_idx(res.shape, torch.argmax(res).detach().cpu().item())  # depth, angle, y, x
        # index = [0, 0, 22, 32]
        print(index, "quality:", res[index[0], index[1], index[2], index[3]].item(), 'depth:', z_pose_init[index[0]].item())
        print('velocity:', pos[0, :, index[2], index[3]].detach().cpu().numpy())
        x = index[3] * 8 + 48
        y = index[2] * 8 + 48
        velocity = np.round(-96 * pos[0, :, index[2], index[3]].detach().cpu().numpy()).astype(np.int32)
        return (x, y), index[1], z_pose_init[index[0]].item(), velocity
    
    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]

    def update_tensor(self):
        depth, color = self.camera.get_img()
        depth = in_paint(depth) / 1000.0
        # depth = depth[100:400, 200:500]
        depth = np.clip(depth, 0.48, 0.57)
        # depth = np.ones((480, 640)) * 0.55
        # depth[300:330, 350:450] = 0.49
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda()
        self.tensor[:, 0:5, :, :] = self.tensor[:, 1:6, :, :].clone()
        self.tensor[:, 5, :, :] = depth_tensor

    def plot(self, img, pos_idx, angle_idx, depth, r=36, velocity=None):
        x, y = pos_idx
        angle = np.pi/16 * angle_idx
        x1 = int(x + r * np.cos(angle))
        x2 = int(x + r * np.cos(angle + np.pi))
        y1 = int(y + r * np.sin(angle))
        y2 = int(y + r * np.sin(angle + np.pi))

        img = cv2.arrowedLine(img, (x, y), (x1, y1), depth, thickness=2)
        img = cv2.arrowedLine(img, (x, y), (x2, y2), depth, thickness=2)

        if velocity is not None:
            dx, dy = map(int, velocity)
            img = cv2.arrowedLine(img, (x, y), (x + dx, y + dy), depth, thickness=4)
        return img

    def action(self):
        t = time.time()
        self.update_tensor()
        pos, angle_idx, z, vel = self.prediction()
        # plt.figure(1)
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     x = self.tensor[0, i, :, :].detach().cpu().numpy()
        #     plt.imshow(x)
        print('time:', time.time()-t)
        img = self.tensor[0, 3, :, :].detach().cpu().numpy()
        img = self.plot(img, pos, angle_idx, z, velocity=vel)
        plt.figure(2)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    # model_path = './train/convTrans23_02_21_18_52_dynamic_attcg_removebn'
    # model_path =  './train/convTrans23_02_11_18_11_attcg-grasp'

    # model_path = './train/convTrans23_03_08_11_34_dynamic_adamw'

    # model_path = './train/convTrans23_03_08_02_01_dynamic_attcg'

    model_path = './train/convTrans23_03_16_18_02_dynamic_win73_depth26_nopad'  # epoch 5
    model_path = './train/convTrans23_03_17_13_25_dynamic_win33_depth22_attcg_L2loss_fixedLr_decay005'
    
    model_path = './train/res23_03_31_00_26_dynamic_gqcnn/'  # resnet backbone
    # model_path = './train/convTrans23_03_31_21_04_dynamic_win3_depth22_nopad_decay005'   # AdamW fixed lr
    # model_path = './train/convTrans23_03_21_16_02_dynamic_win33_depth222_attcg_L1loss_decay005_fixedLr'
   
    model_path = './train/convTrans23_04_02_20_28_dynamic_depth26_attcg_pad'  # 26 depth with pad

    # static
    # model_path = './train/convTrans22_07_29_20_53_batchnorm_patch8_win5'
    m = PredictModel(model_path, k=-1)
    summary(m.prediction_model, (6, 96, 96))
    x = torch.rand(1, 6, 96+8*8, 96+8*8).cuda()
    print(m.prediction_model(x)[0][..., 2, 2])
    print(m.prediction_model(x[..., 8*2:8*2+96, 8*2:8*2+96])[0].squeeze())
    x = torch.ones(1, 6, 96, 96).cuda() * 0.7
    x[:, :, 35:-35, 10:-10] = 0.55
    p = torch.tensor([0.5, 0.6, 0.7, 0.8]).cuda()
    x = (x - torch.mean(x)) / torch.std(x)
    p = (p - torch.mean(x)) / torch.std(x)
    res, pos = m.prediction_model(x, p)
    res = res.squeeze().view(-1, 16, 2)
    res = torch.nn.Softmax(dim=2)(res)[:, :, 1]
    print('?>', res)
    while 1:
        m.action()
        # print('???')
        # x = m.tensor[:, :, 224-48:224+48, 304-48:304+48]
        # x_in = (x - torch.mean(x)) / torch.std(x)
        # pos = torch.arange(1).cuda() * 0.02 + torch.min(x)
        # print(m.prediction_model(x_in, pos))

        

    

