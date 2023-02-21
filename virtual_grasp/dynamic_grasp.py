from virtual_grasp.environment import VirtualEnvironment
import time
import torch
from load_model import build_model
import cv2
import numpy as np
import os
from config import _C
import pybullet as p

from glob import glob
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')


class DynamicEnv(VirtualEnvironment):
    def __init__(self, num_robots):
        super(DynamicEnv, self).__init__(num_robots=num_robots)

        self.ori_init = self.p.getQuaternionFromEuler([0, 3.141592654 / 2, 0])
        ori = self.p.getQuaternionFromEuler(np.random.rand(3) * np.pi)
        self.init_joints_angle = [0.23813124282501125, -1.66718850408224, -0.9366544912538485, -2.125146459392779,
                                  1.5701356825001347, 0.23663993167973682, 0.0, 0.0]
        self.position_end_init = self.p.getLinkState(self.robotId[0], self.endEffectorIndex)[4]
        for d, uid in enumerate(self.robotId):
            self.move_joints(self.init_joints_angle, uid)

        time.sleep(1)
        self.objects = glob(os.path.join(self.meshPath, '*.obj'))
        self.obj_idx = []
        self.z_pos = []
        self.scale = []
        urdf, s, z = self.generate_urdf(self.objects[0])
        self.obj_idx.append(self.p.loadURDF(urdf, basePosition=[0.0, 0 * self.d, 0.3],
                                            baseOrientation=self.ori_init,
                                            globalScaling=s))
        self.z_pos.append(z)
        self.scale.append(s)
        # self.p.changeDynamics(self.obj_idx[0], -1, mass=1, lateralFriction=7,
        #                       restitution=0.98, rollingFriction=5, spinningFriction=5,
        #                       contactStiffness=1e9, contactDamping=5)

        time.sleep(3)



    def update_camera(self, h=640, w=640, n=0.0001, f=1.5):
        projectionMatrix = (2, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2 / (f - n), 0, 0, 0, -(f + n) / (f - n), 1)
        image_renderer = p.ER_BULLET_HARDWARE_OPENGL

        init_camera_vector = np.array((0, 0, -1))  # x-axis
        init_up_vector = np.array((1, 0, 0))  # z-axis

        obs = p.getLinkState(self.robotId[0], self.endEffectorIndex)
        posEnd = obs[4]
        oriEnd = obs[5]

        rot_matrix = p.getMatrixFromQuaternion(oriEnd)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix_gripper = p.computeViewMatrix(posEnd + 0.2 * init_camera_vector, posEnd + 0.3 * init_camera_vector,
                                                  up_vector)
        # print(np.array(view_matrix_gripper).reshape(4, 4))
        width, height, rgb, depth, segment = p.getCameraImage(w, h, view_matrix_gripper, projectionMatrix,
                                                              shadow=0,
                                                              flags=p.ER_NO_SEGMENTATION_MASK,
                                                              renderer=image_renderer)
        depth = 2 * depth - 1
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        depth = (f + n + depth * (f - n)) / 2

        rgb = rgb[80:560, 80:560, :]
        depth = depth[80:560, 80:560]

        return rgb, depth, posEnd

    def pixel2base(self, pixelIndex, angle, depthImage=None, depth=None, dx=0, dy=0):
        assert depthImage is not None or depth is not None, 'depthImage and depth cannot be None at the same time'
        xp, yp = pixelIndex
        xp = max(0, min(xp, 639 - dx * 2))
        yp = max(0, min(yp, 639 - dy * 2))

        if depthImage is not None:
            z = depthImage[yp, xp]  # meter
        else:
            z = depth - 0.15
        # print('z:', z)
        # Pobj2cam = np.dot(np.linalg.inv(self.cameraMat), [[xp+dx], [yp+dy], [1]]) * z  # meter
        Pobj2cam = np.array([[(xp + dx - 320) / 640], [(yp + dy - 320) / 640], [z]])

        Tcam2end = np.array([
            [0, 0, 1, 0.2],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        Rend2cam = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        Tobj2cam = np.vstack((
            np.hstack(
                (np.array(cv2.Rodrigues(np.array([0, 0, -angle], dtype=np.float64))[0]).dot(Rend2cam),
                 Pobj2cam)
            ), np.array([[0, 0, 0, 1]])
        ))

        obs = p.getLinkState(self.robotId[0], self.endEffectorIndex)
        Pend2base = obs[4]
        Rend2base = np.array(p.getMatrixFromQuaternion(obs[5])).reshape(3, 3)
        Tend2base = np.vstack((
            np.hstack(
                (Rend2base, np.expand_dims(np.array(Pend2base), 0).T)
            ), np.array([[0, 0, 0, 1]])
        ))

        Tobj2base = Tend2base.dot(Tcam2end.dot(Tobj2cam))

        quatObj2Base = self.rotMat2Quat(Tobj2base[0:3, 0:3])
        posObj2Base = Tobj2base[0:3, 3]
        _, quatObj2Base = p.invertTransform([0, 0, 0], quatObj2Base)
        return quatObj2Base, posObj2Base

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
            dx, dy = velocity
            dx = int(96*dx)
            dy = int(96*dy)
            img = cv2.arrowedLine(img, (x, y), (x + dx, y + dy), depth, thickness=2)
        return img



class DynamicGrasp():
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
        self._env = DynamicEnv(num_robots=1)

    def main_action(self):
        img_tensor = torch.ones(1, 6, 480, 480).cuda()
        self.set_obj_movement()
        for i in range(6):
            color_image, depth_image, position_end = self._env.update_camera()
            # depth_image = np.ones((480, 480)) * 0.7
            # depth_image[200:300, 370:400] = 0.4
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).cuda()
            img_tensor[:, i, :, :] = depth_tensor
            time.sleep(0.1)
        while position_end[2] > 0.2:
            try:
                _, depth_image, position_end = self._env.update_camera()
                # depth_image = np.ones((480, 480)) * 0.7
                # depth_image[200:300, 370:400] = 0.4
                depth_image = self.img_transform(depth_image, position_end)
                depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).cuda()
                img_tensor[:, 0:5, :, :] = img_tensor[:, 1:6, :, :].clone()
                img_tensor[:, 5, :, :] = depth_tensor
                if self.prediction_model.dynamic:
                    quat, pos, index = self.prediction(img_tensor)
                else:
                    quat, pos, index = self.prediction(img_tensor[:, 0:1, :, :])
                print('quat:', quat, 'pos:', pos)
                img = img_tensor.cpu().numpy()
                for i in range(6):
                    plt.subplot(2, 3, i + 1)
                    velocity = index[4] if self.prediction_model.dynamic else None
                    if i == 3:
                        x = self._env.plot_grasp(img[0, i, :, :], index[0:2], index[2], index[3], velocity=velocity)
                    else:
                        x = img[0, i, :, :]
                    plt.imshow(x)
                plt.show()

                # self.move((quat, pos))
            except KeyboardInterrupt:
                break

        self._env.move_joints(self._env.init_joints_angle, self._env.robotId[0])
        time.sleep(3)

    def img_transform(self, img, position_end):
        delta_x = position_end[1] - self._env.position_end_init[1]
        delta_y = position_end[0] - self._env.position_end_init[0]
        mat = np.array([[1, 0, -delta_x], [0, 1, -delta_y]], dtype=np.float32)
        img = cv2.warpAffine(img, mat, img.shape, borderMode=cv2.BORDER_REPLICATE)
        return img

    def prediction(self, img_tensor, depth_bin=8):
        # depth_reso = (torch.max(img_tensor) - torch.min(img_tensor)) / depth_bin
        depth_reso = 1e-2
        z_pose_init = torch.arange(depth_bin).cuda() * depth_reso + torch.min(img_tensor)

        img_tensor = (img_tensor - torch.mean(img_tensor)) / torch.std(img_tensor)
        z_pose = (z_pose_init - torch.mean(img_tensor)) / torch.std(img_tensor)
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
        print('velocity:', pos[index[0], :, index[2], index[3]].detach().cpu().numpy())
        x = index[3] * 8 + 48
        y = index[2] * 8 + 48

        quatObj2Base, posObj2Base = self._env.pixel2base((x, y), index[1]*np.pi/16, None,
                                                         z_pose[index[0]].detach().cpu().item(), 180, 180)
        return quatObj2Base, posObj2Base, (x, y, index[1], z_pose_init[index[0]].item(),
                                           pos[index[0], :, index[2], index[3]].detach().cpu().numpy())

    def move(self, tar_pose):
        quat, pos = tar_pose
        joint_angle = self._env.get_joints_angle(pos, quat, 0.5, self._env.robotId[0])
        self._env.move_joints(joint_angle, self._env.robotId[0])

    def set_obj_movement(self):
        self._env.p.resetBaseVelocity(self._env.obj_idx[0], [0.1, 0, 0], [0, 0, 0])

    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]


if __name__ == '__main__':
    # model_path = './train/convTrans22_08_04_23_18_pos_branch_6slices_dynamic_curbest'
    # model_path = './train/convTrans22_07_29_20_53_batchnorm_patch8_win5'
    # model_path = './train/convTrans23_02_20_22_53_dynamic'
    model_path = './train/convTrans23_02_21_17_25_dynamic_attcg'
    dynamic_grasp = DynamicGrasp(model_path)
    while 1:
        dynamic_grasp.main_action()

