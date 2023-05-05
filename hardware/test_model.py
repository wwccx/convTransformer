import sys

sys.path.append("/home/wcx/convTransformer/")
import torch
import cv2
import numpy as np
from load_model import build_model
import os
import utils as u
import time
from matplotlib import pyplot as plt
# from camera import RS
from utils import in_paint
from torchsummary import summary


class PredictModel():
    def __init__(self, model_path, k=-1, data_path=None):
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
        # self.camera = RS(640, 480)

        self.tensor = torch.ones(1, 6, 480, 640).cuda()
        self.color = np.ones((6, 480, 640, 3))
        # self.tensor = torch.ones(1, 6, 300, 300).cuda()
        if data_path is not None:
            self.data = np.load(data_path)
        else:
            self.data = None
        self.data_idx = 0
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
        print(index, "quality:", res[index[0], index[1], index[2], index[3]].item(), 'depth:',
              z_pose_init[index[0]].item())
        print('velocity:', pos[0, :, index[2], index[3]].detach().cpu().numpy())
        x = index[3] * 8 + 48
        y = index[2] * 8 + 48
        velocity = np.round(-96 * pos[0, :, index[2], index[3]].detach().cpu().numpy()).astype(np.int32)
        return (x, y), index[1], z_pose_init[index[0]].item(), velocity, res, pos

    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]

    def update_tensor(self):
        if self.data:
            depth = self.data['d'][self.data_idx]
            color = self.data['c'][self.data_idx]
            self.data_idx += 1

        else:
            depth, color = self.camera.get_img()
            depth = in_paint(depth) / 1000.0
        # depth = depth[100:400, 200:500]
        self.color[0:5] = self.color[1:6]
        self.color[5] = color
        depth = np.clip(depth, 0.48, 0.57)
        # depth = np.ones((480, 640)) * 0.55
        # depth[300:330, 350:450] = 0.49
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda()
        self.tensor[:, 0:5, :, :] = self.tensor[:, 1:6, :, :].clone()
        self.tensor[:, 5, :, :] = depth_tensor

    def grasp_plot(self, axes, pos_idx, angle_idx, arrow_color, r, velocity):
        x, y = pos_idx
        angle = np.pi / 16 * angle_idx
        x1 = int(r * np.cos(angle))
        x2 = int(r * np.cos(angle + np.pi))
        y1 = int(r * np.sin(angle))
        y2 = int(r * np.sin(angle + np.pi))
        axes.arrow(x, y, x1, y1, head_width=5, head_length=5, fc=arrow_color, ec=arrow_color, width=2)
        axes.arrow(x, y, x2, y2, head_width=5, head_length=5, fc=arrow_color, ec=arrow_color, width=2)
        if velocity is not None:
            dx, dy = map(int, velocity)
            axes.arrow(x, y, dx, dy, head_width=5, head_length=5, fc=arrow_color, ec=arrow_color, width=2)

    def velocity_field_vis(self, velocity, axes, arrow_color):
        for i in range(velocity.shape[2]):
            for j in range(velocity.shape[3]):
                if i % 2 or j % 2:
                    continue
                x, y = j * 8 + 48, i * 8 + 48
                dx, dy = (-96 * velocity[0, :, i, j]).astype(np.int32)
                axes.arrow(x, y, dx, dy, head_width=5, head_length=5, fc=arrow_color, ec=arrow_color, width=2,
                           alpha=0.3)

    def quality_vis(self, quality, axes, fig):
        quality = quality[0].sum(axis=0)
        quality = quality / quality.max()
        quality = cv2.resize(quality.astype(np.float32), (640, 480))
        # quality = cv2.applyColorMap(quality, cv2.COLORMAP_JET)
        # quality = cv2.cvtColor(quality, cv2.COLOR_RGB2RGBA)
        fig.colorbar(axes.imshow(quality, cmap='jet'), ax=axes)

    def plot_grasp_distribution(self):
        assert self.data is not None, 'data is None! cannot plot grasp distribution'
        color_img = self.data['c'][self.data_idx]
        color_img /= np.max(color_img)
        fig = plt.figure(figsize=(10, 5), dpi=400)
        axes1 = fig.add_subplot(121)
        axes1.axis('off')
        axes1.imshow(color_img)
        axes2 = fig.add_subplot(122)
        # set axes limits
        axes2.set_xlim([-0.4, 0.4])
        axes2.set_ylim([-0.4, 0.4])
        # draw x and y axis
        axes2.axhline(y=0, color='k')
        axes2.axvline(x=0, color='k')

        while True:
            pos, angle_idx, z, vel, quality, v_field = self.prediction()
            self.grasp_plot(axes1, pos, angle_idx, (0, 0, 1, 1), 36, velocity=vel)
            axes2.plot(vel[0] / 100, vel[1] / 100, 'ro')
            try:
                self.update_tensor()
            except:
                break
        plt.show()

    def action(self):
        t = time.time()
        self.update_tensor()
        pos, angle_idx, z, vel, quality, v_field = self.prediction()
        # plt.figure(1)
        # for i in range(6):
        #     plt.subplot(2, 3, i + 1)
        #     x = self.tensor[0, i, :, :].detach().cpu().numpy()
        #     plt.imshow(x)
        print('time:', time.time() - t)

        # img = self.tensor.detach().cpu().numpy()
        # res = img[0, 0, :, :]
        # for i in range(1, 6):
        #     res = cv2.addWeighted(res, 0.6, img[0, i, :, :], 1, 0)
        # img = self.plot(res, pos, angle_idx, z, velocity=vel)
        # plt.figure(2)
        # plt.imshow(img)
        # plt.show()
        img = self.color
        res = img[0]
        for i in range(1, 6):
            res = cv2.addWeighted(res, 0.6, img[i], 1, 0)
        res /= np.max(res)
        res = cv2.cvtColor(res.astype(np.float32), cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10, 5), dpi=400)
        axes = fig.add_subplot(121)
        axes.imshow(res)
        axes.axis('off')
        quality_axes = fig.add_subplot(122)
        quality_axes.axis('off')
        self.grasp_plot(axes, pos, angle_idx, (0, 0, 1, 1), 36, velocity=vel)
        self.velocity_field_vis(v_field.cpu().numpy(), axes, (0, 1, 0, 0.5))
        self.quality_vis(quality.cpu().numpy(), quality_axes, fig)
        # plt.figure(2)
        # plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    # model_path = './train/convTrans23_02_21_18_52_dynamic_attcg_removebn'
    # model_path =  './train/convTrans23_02_11_18_11_attcg-grasp'

    # model_path = './train/convTrans23_03_08_11_34_dynamic_adamw'

    # model_path = './train/convTrans23_03_08_02_01_dynamic_attcg'

    model_path = './train/convTrans23_03_09_19_35_dynamic_dim192_allBN'
    # model_path = './train/convTrans23_03_10_15_22_dynamic_dim192_allBN_300epoch'
    model_path = './train/convTrans23_03_12_19_47_dynamic_attcg_allBN_win3'
    # model_path = './train/convTrans23_03_13_14_06_dynamic_attcg_allBN_win3_amp0'
    # model_path = './train/convTrans23_03_12_23_33_dynamic_adamw_win3'
    model_path = './train/convTrans23_03_13_17_04_dynamic_attcg_win3_Rlossweight10'
    model_path = './train/convTrans23_03_13_23_23_dynamic_adamw_win3_Rlossweight10'  # 22, 32
    # model_path = './train/convTrans23_03_14_14_44_dynamic_adamw_win3_Rlossweight10_postanh'
    model_path = './train/convTrans23_03_14_16_49_dynamic_adamw_win5_Rlossweight10_posonly'
    # model_path = './train/convTrans22_08_04_23_18_pos_branch_6slices_dynamic_curbest'
    model_path = './train/convTrans23_03_14_18_33_dynamic_adamw_win3_depth26_LN'
    # model_path = './train/convTrans23_03_14_21_13_dynamic_win73_depth26_LN'
    model_path = './train/convTrans23_03_15_17_56_dynamic_adamw_win73_depth26_L2loss'  # epoch 10
    model_path = './train/convTrans23_03_16_18_02_dynamic_win73_depth26_nopad'  # epoch 5
    model_path = './train/convTrans23_03_17_00_24_dynamic_adamw_win33_depth22_L2loss_nopad'
    # model_path = './train/convTrans23_03_17_03_09_dynamic_adamw_win73_depth22_attcg_L2loss'
    model_path = './train/convTrans23_03_17_13_25_dynamic_win33_depth22_attcg_L2loss_fixedLr_decay005'
    # model_path = './train/convTrans23_03_18_11_53_dynamic_win33_depth22_adamw_L2loss_decay005'

    model_path = './train/dynamic_backbone_comparation/convTrans23_04_02_20_28_dynamic_depth26_attcg_pad'  # my
    # model_path = './train/dynamic_backbone_comparation/res23_03_31_00_26_dynamic_gqcnn'  # res
    # model_path = 'train/dynamic_backbone_comparation/gqcnn23_03_30_16_59_dynamic_gqcnn'  # normal cnn

    # model_path = 'train/convTrans23_02_20_22_53_dynamic'  # adamw my
    # amp 01 yes, patch size 8 yes, BN layer yes
    # static
    # model_path = './train/convTrans22_07_29_20_53_batchnorm_patch8_win5'
    m = PredictModel(model_path, k=-1, data_path='./data/img/static_img_rabbit.npz')
    # m = PredictModel(model_path, k=-1, data_path='./data/img/static_img_headphone.npz')
    # m = PredictModel(model_path, k=-1, data_path='./data/img/dynamic_img_banana1.npz')

    # m.plot_grasp_distribution()

    x = torch.rand(1, 6, 96 + 8 * 8, 96 + 8 * 8).cuda()
    print(m.prediction_model(x)[0][..., 2, 2])
    print(m.prediction_model(x[..., 8 * 2:8 * 2 + 96, 8 * 2:8 * 2 + 96])[0].squeeze())
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





