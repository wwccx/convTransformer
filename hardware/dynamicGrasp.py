import sys
sys.path.append("/home/server/convTransformer/")
from camera import RS
from gripper import Gripper
from vel_control import VelController
import torch
import cv2
import numpy as np
# from convTrans import convTransformer
from load_model import build_model
import os
import utils as u
from multiprocessing import Process, Queue
from threading import Thread
import time
from matplotlib import pyplot as plt

class RealEnvironment():
    def __init__(self) -> None:
        # self.camera = RS(1280, 720)
        self.camera = RS(640, 480)
        self.gripper = Gripper()
        self.pose_queue = Queue()
        self.robot = VelController(self.pose_queue)
        self.robot.setDaemon(True)
        self.robot.start()
        # self.robot.join()
        self.camera_intrinsics = np.array([[614.887, 0.0, 328.328], [0.0, 614.995, 236.137], [0.0, 0.0, 1.0]])

        self.init_end_pos, self.init_end_ori = self.robot.aim_pose_init
        self.Tcam2end = np.array([
            [0.99770124, -0.06726973, -0.00818663, -0.02744465],
            [0.06610884, 0.99272747, -0.10060715, -0.10060833],
            [0.01489491, 0.09983467, 0.99489255, -0.15038112],
            [0., 0., 0., 1., ]
        ])
        self.Tend2cam = np.linalg.inv(self.Tcam2end)
        self.Rend2cam = self.Tend2cam[0:3, 0:3]

    def update_camera(self):
        depth, rgb = self.camera.get_img()
        pos, ori = self.robot.get_current_tool_pose()  # x y z, rotMat \in \mathbb{R}^{3 \times 3}
        depth = u.in_paint(depth)
        depth /= 1000
        plt.figure(3)
        plt.imshow(depth)
        depth = np.clip(depth, 0.48, float('inf'))

        return rgb, depth, pos, ori

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

    def move_joints(self, pos, ori, block=False, jaw=None):
        # print(pos)
        pos[2] = max(0.033, pos[2])
        self.pose_queue.put((pos, ori))
        if jaw:
            self.gripper.gripper_position(int(100*jaw))
        if block:
            while not self.robot.reach_target:
                try:
                    print('wait until robot reaches target')
                    pass
                except:
                    break

    def pixel2base(self, pixel_index, angle, depth, dx=0, dy=0):
        
        xp, yp = pixel_index
        xp = max(0, min(xp, 639 - dx * 2))
        yp = max(0, min(yp, 479 - dy * 2))
        
        z = depth - 0.05
        # print('z:', z)
        # Pobj2cam = np.dot(np.linalg.inv(self.cameraMat), [[xp+dx], [yp+dy], [1]]) * z  # meter
        xpc, ypc, zpc = np.linalg.inv(self.camera_intrinsics) @ np.array([xp, yp, 1]) * z  # pc: pos in cam
        Tcam2end = self.Tcam2end
        # Tobj2cam = np.vstack((
        #     np.hstack(
        #         (np.array(cv2.Rodrigues(np.array([0, 0, -angle], dtype=np.float64))[0]).dot(self.Rend2cam),
        #          Pobj2cam)
        #     ), np.array([[0, 0, 0, 1]])
        # ))
        # Tobj2cam = np.eye(4)
        # Tobj2cam[0:3, 0:3] = np.array(cv2.Rodrigues(np.array([0, 0, -angle], dtype=np.float64))[0]).dot(self.Rend2cam)
        # Tobj2cam[0:3, 3] = Pobj2cam

        Tend2base = u.pos_ori2mat(self.init_end_pos, self.init_end_ori)
        Tcam2base = Tend2base.dot(Tcam2end)
        Pobj2base = Tcam2base @ np.array([[xpc], [ypc], [zpc], [1]])
        Pobj2base = Pobj2base.squeeze()[:-1]
        Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle], dtype=np.float64))[0]).dot(self.robot.aim_pose_init[1])

        return Pobj2base, Robj2base


class DynamicGrasp:
    def __init__(self, model_path):

        print(os.path.join(model_path, 'config.pth'))
        model_config = torch.load(os.path.join(model_path, 'config.pth'))['config']

        print(model_config)
        pths = [i for i in os.listdir(model_path) if i.endswith('.pth')]
        pths.sort(key=lambda x: (os.path.getmtime(os.path.join(model_path, x))))
        # print(pths)

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

    def img_transform(self, img, position_end, ori_end, color=None):
        Tend = u.pos_ori2mat(position_end, ori_end)
        Tend_init = u.pos_ori2mat(self._env.init_end_pos, self._env.init_end_ori)

        Tcam = self._env.Tend2cam @ np.linalg.inv(Tend)  #Tbase2cam
        Tcam_init = self._env.Tend2cam @ np.linalg.inv(Tend_init)
        x = np.arange(-0.2, 0.4, 0.4)
        x, y = np.meshgrid(x, x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        points = np.ones((4, x.size))
        points[0, :] = x
        points[1, :] = y + 0.6
        points[2, :] = 0
        points_cam = (Tcam @ points)[:3, :]
        points_init = (Tcam_init @ points)[:3, :]
        points_cam /= points_cam[2, :]
        points_init /= points_init[2, :]
        points_cam = (self._env.camera_intrinsics @ points_cam)[:2, :].astype(np.float32)
        points_init = (self._env.camera_intrinsics @ points_init)[:2, :].astype(np.float32)
        MP = cv2.getPerspectiveTransform(points_cam.T, points_init.T)

        # plt.figure(2)
        # plt.imshow(img)
        # plt.title('original')
        img = cv2.warpPerspective(img, MP, img.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        # plt.figure(3)
        # plt.imshow(img)
        # plt.title('warped')
        # plt.show()
        # delta_x = (position_end[1] - self._env.init_end_pos[1]) * 640
        # delta_y = (position_end[0] - self._env.init_end_pos[0]) * 640
        # mat = np.array([[1, 0, -delta_x], [0, 1, -delta_y]], dtype=np.float32)
        # img = cv2.warpAffine(img, mat, img.shape, borderMode=cv2.BORDER_REPLICATE)
        img -= (position_end[2] - self._env.init_end_pos[2])
        # img = np.clip(img, 0.1, self._env.robot.aim_pose_init[0][2] + 0.15)
        if color is None:
            return img
        else:
            color = cv2.warpPerspective(color, MP, img.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
            return img, color

    @torch.no_grad()
    def prediction(self, img_tensor, depth_bin=8):
        # depth_reso = (torch.max(img_tensor) - torch.min(img_tensor)) / depth_bin
        depth_reso = 1e-2
        z_pose_init = -torch.arange(depth_bin).cuda() * depth_reso + torch.max(img_tensor)
        print(torch.min(img_tensor))

        # img_tensor = (img_tensor - torch.mean(img_tensor)) / torch.std(img_tensor)
        # z_pose = (z_pose_init - torch.mean(img_tensor)) / torch.std(img_tensor)
        img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor)) - 0.5
        z_pose = (z_pose_init - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor)) - 0.5
        if self.prediction_model.dynamic:
            res, pos = self.prediction_model(img_tensor, z_pose)
        else:
            res = self.prediction_model(img_tensor, z_pose)
            pos = torch.zeros_like(res)[:, :2, ...]
        s = list(res.shape)
        s[1:2] = [16, 2]
        sf = torch.nn.Softmax(dim=2)
        res = sf(res.view(*s))[:, :, 1, ...]
        # s[2:3] = []

        index = self.get_tensor_idx(res.shape, torch.argmax(res).detach().cpu().item())  # depth, angle, y, x
        print(index, "quality:", res[index[0], index[1], index[2], index[3]].item())
        print('velocity:', pos[0, :, index[2], index[3]].detach().cpu().numpy())
        x = index[3] * 8 + 48
        y = index[2] * 8 + 48
        angle = index[1] * np.pi / 16
        if angle > np.pi / 2:
            angle -= np.pi
        velocity = np.round(-0 * pos[0, :, index[2], index[3]].detach().cpu().numpy()).astype(np.int32)
        pos, ori = self._env.pixel2base((x + velocity[0], y + velocity[0]), angle,
                                                         z_pose[index[0]].detach().cpu().item(),)
        return pos, ori, (x, y, index[1], z_pose_init[index[0]].item(), velocity)
    
    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]
    
    def test_img_transform(self):
        from matplotlib import pyplot as plt
        pos_bias = np.array([0, 0, 0.02])
        ori_bias = cv2.Rodrigues(np.array([0, 0, np.pi/20]))[0]
        color_image, depth_image, position_end, ori_end = self._env.update_camera()
        time.sleep(3)
        for i in range(5):
            print('image', i)
            color_image, depth_image, position_end, ori_end = self._env.update_camera()
            #cv2.imwrite(f'../data/p_test/color_real{i}.png', color_image)
            #cv2.imwrite(f'../data/p_test/depth_real{i}.png', depth_image)
            plt.imsave(f'./data/p_test/color_real{i}.png', color_image)
            plt.imsave(f'./data/p_test/depth_real{i}.png', (depth_image - np.min(depth_image))/(np.max(depth_image) - np.min(depth_image)))
            depth_image, color_image = self.img_transform(depth_image, position_end, ori_end, color_image)
            #cv2.imwrite(f'../data/p_test/color_pers{i}.png', color_image)
            #cv2.imwrite(f'../data/p_test/depth_pers{i}.png', depth_image)
            plt.imsave(f'./data/p_test/color_pers{i}.png', color_image)
            plt.imsave(f'./data/p_test/depth_pers{i}.png', (depth_image - np.min(depth_image))/(np.max(depth_image) - np.min(depth_image)))
            self._env.move_joints(position_end-pos_bias, ori_bias @ ori_end)
            time.sleep(3)
        self._env.pose_queue.put(None)
        return
        # while 1:
        #     color_image, depth_image, position_end, ori_end = self._env.update_camera()
        #     plt.figure(1)
        #     plt.imshow(depth_image)
        #     depth_image, color_image = self.img_transform(depth_image, position_end, ori_end, color_image)
        #     plt.figure(2)
        #     plt.imshow(depth_image)
        #     # cv2.imshow('depth', depth_image * 100)
        #     # cv2.waitKey(2)
        #     plt.show()

    def main_action(self, v=-0.02):
        img_tensor = torch.ones(1, 6, 480, 640).cuda()
        for i in range(6):
            color_image, depth_image, position_end, ori_end = self._env.update_camera()
            depth_image = self.img_transform(depth_image, position_end, ori_end)
            # depth_image = np.clip(depth_image, 0.48, 0.57)
            time_step = time.time()
            # depth_image = np.ones((480, 480)) * 0.7
            # depth_image[200:300, 370:400] = 0.4
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).cuda()
            img_tensor[:, i, :, :] = depth_tensor
            time.sleep(0.2)

        while position_end[2] > 0.10:
            while time.time() - time_step < 0.2:
                pass
            try:
                _, depth_image, position_end, ori_end = self._env.update_camera()
                time_step = time.time()
                # depth_image = np.ones((480, 480)) * 0.7
                # depth_image[200:300, 370:400] = 0.4
                depth_image = self.img_transform(depth_image, position_end, ori_end)
                # depth_image = np.clip(depth_image, 0.48, 0.57)
                depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).cuda()
                img_tensor[:, 0:5, :, :] = img_tensor[:, 1:6, :, :].clone()
                img_tensor[:, 5, :, :] = depth_tensor
                if self.prediction_model.dynamic:
                    pos, ori, index = self.prediction(img_tensor)
                else:
                    pos, ori, index = self.prediction(img_tensor[:, 0:1, :, :])
                print('quat:\n', ori, '\n', 'pos:\n', pos)
                print(index)
                angle = index[1] * np.pi / 16
                if angle > np.pi / 2:
                    angle -= np.pi
                pos_bias_better_campos = np.array([0.12*np.sin(angle), -0.12*np.cos(angle), 0.05])
                img = img_tensor.cpu().numpy()
                for i in range(6):
                    plt.figure(1)
                    plt.subplot(2, 3, i + 1)
                    velocity = index[4] if self.prediction_model.dynamic else None
                    if i == 3:
                        x = self._env.plot_grasp(img[0, i, :, :], index[0:2], index[2], index[3], velocity=velocity)
                        # cv2.imshow('grasp', x)
                        # cv2.waitKey(1)
                    else:
                        x = img[0, i, :, :]
                    plt.imshow(x)
                self._env.robot.suspend()
                plt.show()
                self._env.robot.continue_listen()
                # self._env.robot.start()
                # self._env.robot.end_joint_rotation()
                self._env.move_joints(pos + pos_bias_better_campos, ori, jaw=0.8)
            except KeyboardInterrupt:
                break
            # finally:
            #     break

        self._env.move_joints(pos + [0, 0, 0.05], ori, jaw=0.8)
        time.sleep(2)
        self._env.move_joints(pos, ori, jaw=0.8)
        time.sleep(2)
        self._env.move_joints(pos, ori, jaw=0)
        time.sleep(1)
        self._env.move_joints(pos + [0, 0, 0.25], ori)
        time.sleep(3)

        # self._env.move_joints(*self._env.robot.aim_pose_init)


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


if __name__ == '__main__':
    # x = np.random.rand(400, 400)
    # from matplotlib import pyplot as plt
    # plt.imsave('./data/p_test/hmm.png', x)
    # model_path = './train/convTrans23_02_21_18_52_dynamic_attcg_removebn'
    # model_path = './train/convTrans23_03_12_19_47_dynamic_attcg_allBN_win3'
    model_path = './train/convTrans23_03_17_13_25_dynamic_win33_depth22_attcg_L2loss_fixedLr_decay005'
    # model_path =  './train/convTrans23_02_11_18_11_attcg-grasp'
    dynamic_grasp = DynamicGrasp(model_path)
    # time.sleep(5)
    # dynamic_grasp.test_img_transform()
    dynamic_grasp.main_action()

    # dynamic_grasp._env.move_joints(pos, ori)
    time.sleep(5)
    dynamic_grasp._env.robot.end_joint_rotation()
