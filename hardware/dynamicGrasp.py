from camera import RS
from gripper import Gripper
from vel_control import VelController
import torch
import cv2
import numpy as np
from ..convTrans import convTransformer
from ..load_model import build_model
import os
import utils as u


class RealEnvironment():
    def __init__(self) -> None:
        self.camera = RS(640, 480)
        self.gripper = Gripper()
        self.robot = VelController()
        self.camera_intrinsics = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])

        self.init_end_pos, self.init_end_ori = self.robot.get_current_pose()
        self.Tend2cam = np.array([
            [0.99770124, -0.06726973, -0.00818663, -0.02744465],
            [0.06610884, 0.99272747, -0.10060715, -0.10060833],
            [0.01489491, 0.09983467, 0.99489255, -0.15038112],
            [0., 0., 0., 1., ]
        ])

    def update_camera(self):
        depth, rgb = self.camera.get_img()
        depth = u.in_paint(depth)
        pos, quat = self.robot.get_current_pose()

        return rgb, depth, pos, quat

    def plot_grasp(self, img, pos_idx, angle_idx, depth, r=36, velocity=None):
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

    def move_joints(self, *pos, **vel):
        self.robot.set_joint_velocity(*pos, **vel)


class DynamicGrasp:
    def __init__(self, model_path):
        print(os.path.join(model_path, 'config.pth'))
        model_config = torch.load(os.path.join(model_path, 'config.pth'))['config']

        print(model_config)
        pths = [i for i in os.listdir(model_path) if i.endswith('.pth')]
        pths.sort(key=lambda x: (os.path.getmtime(os.path.join(model_path, x))))
        print(pths)

        # self._env.reset_all_robots()

        k = -1  # candidate: epoch 41, k = 42 epoch55 66
        self.prediction_model = build_model(model_config)
        self.prediction_model.load_state_dict(
            torch.load(
                os.path.join(model_path, pths[k])
            )['model']
        )
        print('load model:', pths[k])
        self.prediction_model.eval()
        self.prediction_model.to('cuda')
        self._env = RealEnvironment()

    def img_transform(self, img, position_end, ori_end):
        Tend = u.pos_quat2mat(position_end, ori_end)
        Tend_init = u.pos_quat2mat(self._env.init_end_pos, self._env.init_end_ori)

        Tcam = self._env.Tend2cam @ np.linalg.inv(Tend)
        Tcam_init = self._env.Tend2cam @ np.linalg.inv(Tend_init)
        x = np.arange(-0.3, 0.6, 0.6)
        x, y = np.meshgrid(x, x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        points = np.ones((4, x.size))
        points[0, :] = x
        points[1, :] = y
        points[2, :] = 0
        points_cam = (Tcam @ points)[:3, :]
        points_init = (Tcam_init @ points)[:3, :]
        points_cam[2, :] = 1
        points_init[2, :] = 1
        points_cam = (self._env.camera_intrinsics @ points_cam)[:2, :].astype(np.float32)
        points_cam /= points_cam[2, :]
        points_init = (self._env.camera_intrinsics @ points_init)[:2, :].astype(np.float32)
        points_init /= points_init[2, :]
        MP = cv2.getPerspectiveTransform(points_cam.T, points_init.T)

        # plt.figure(2)
        # plt.imshow(img)
        # plt.title('original')
        img = cv2.warpPerspective(img, MP, img.shape, borderMode=cv2.BORDER_REPLICATE)
        # plt.figure(3)
        # plt.imshow(img)
        # plt.title('warped')
        # plt.show()
        # delta_x = (position_end[1] - self._env.init_end_pos[1]) * 640
        # delta_y = (position_end[0] - self._env.init_end_pos[0]) * 640
        # mat = np.array([[1, 0, -delta_x], [0, 1, -delta_y]], dtype=np.float32)
        # img = cv2.warpAffine(img, mat, img.shape, borderMode=cv2.BORDER_REPLICATE)
        img -= (position_end[2] - self._env.init_end_pos[2])
        return img


# class DynamicGrasp:
#     def __init__(self, model_ptr) -> None:
#         self.controller = VelController()
#         self.camera = RS(640, 480)
#         self.gripper = Gripper()
#         self.net = None
#         self.net_config = None
#         self._load_model(model_ptr)
#         self.init_pos, self.init_quat = self.controller.get_current_pose()
#         self.init_tmat = u.pos_quat2mat(self.init_pos, self.init_quat)
#         self.queue_size = 10
#         self.img_tensor = torch.zeros((self.queue_size, 480, 640)).cuda()
#         self.start = 0
#         self.end = self.net_config['time_slices'] + self.start
#         self._init_img_tensor()
#         self.corner_points = np.array([[1, 1, 1], [1, 478, 1], [638, 1, 1], [638, 478, 1]])
#
#     def _load_model(self, model_ptr):
#         ckps = os.listdir(model_ptr)
#         ckps = [os.path.join(model_ptr, c) for c in ckps if c.endswith('.pth')]
#         ckps.sort(key=os.path.getmtime)
#
#         config = torch.load(ckps[0])['config']
#         print(config)
#         self.net = build_model(config).cuda()
#         latest_ckp = ckps[-1]
#
#         save_data = torch.load(latest_ckp)
#         print(f"Epoch:{save_data['epoch']}")
#         self.net.load_state_dict(save_data['model'])
#         self.net.eval()
#         self.net_config = config
#
#     def _init_img_tensor(self):
#         d, c = self.camera.get_img()
#         d = u.in_paint(d)
#         d = torch.from_numpy(d).cuda().unsqueeze(0).repeat((self.end-self.start, 1, 1))
#         self.img_tensor[self.start:self.end] = d
#
#     def warp_perspective(self, img):
#
#         pos, quat = self.controller.get_current_pose()
#         mat = u.pos_quat2mat(pos, quat)
#         points = np.linalg.inv(self.camera.intr) @ self.corner_points.T
#         points = points * img[self.corner_points[:, 1], self.corner_points[:, 0]]
#
#         point_4d = np.ones(4, 4)
#         point_4d[0:3, :] = points
#
#         point_init_frame = np.linalg.inv(self.init_tmat).dot(mat).dot(point_4d)
#         point_init_frame = point_init_frame / point_init_frame[2, :]
#         point_init_frame = self.camera.intr @ point_init_frame[0:3, :]
#         perspective_mat, _ = cv2.findHomograph(self.corner_points[:, :2], point_init_frame[:, :2])
#         perspective_img = cv2.warpPerspective(img, perspective_mat, (img.shape[1], img.shape[0]))
#
#         return u.inpaint(perspective_img)
#
#     def get_img_slice(self):
#         if self.end > self.start:
#             return self.img_tensor[self.start:self.end]
#         else:
#             return torch.stack(
#                     (self.img_tensor[self.start:], self.img_tensor[:self.end]), dim=0
#                     )
#
#     def update_img_tensor(self):
#         d, c = self.camera.get_img()
#         d = u.in_paint(d)
#         d = self.warp_perspective(d)
#         self.start = (self.start + 1) % self.queue_size
#         self.end = (self.end + 1) % self.queue_size
#         d = torch.from_numpy(d).unsqueeze(0).cuda()
#         self.img_tensor[self.end] = d
#
#
#     def grasp_predict(self, depth_reso=5e-3, depth_bin=8):
#         self.update_img_tensor()
#         input_img = self.get_img_slice().unsqueeze(0)
#         z_pose = torch.arange(depth_bin).cuda() * depth_reso + torch.min(input_img)
#         res, pos = self.net(input_img, z_pose)
#         s = list(res.shape)
#         s[1:2] = [16, 2]
#         sf = torch.nn.Softmax(dim=2)
#         res = sf(res)[:, :, 1, ...]
#         s[2:3] = []
#
#         index = self.get_tensor_idx(res.shape, torch.argmax(res).detach().cpu().item())  # depth, angle, y, x
#
#         position = self.pixel2cam(index[2], index[3], index[0])
#
#         #TODO: add the pos prediction to the position
#         return position, index[1]
#
#     @staticmethod
#     def get_tensor_idx(shape, arg_max):
#         idx = []
#         for i in shape[::-1]:
#             idx.append(arg_max % i)
#             arg_max = arg_max // i
#         return idx[::-1]
#
#     @staticmethod
#     def pixel2cam(y, x, depth):
#         pass
    






