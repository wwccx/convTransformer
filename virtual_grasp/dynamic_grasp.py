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


def trans_mat_from_quat_pos(quat, pos):
    mat = np.eye(4)
    r = p.getMatrixFromQuaternion(quat)
    mat[0:3, 0:3] = np.array(r).reshape(3, 3)
    mat[0:3, 3] = pos
    return mat


class DynamicEnv(VirtualEnvironment):
    def __init__(self, num_robots):
        super(DynamicEnv, self).__init__(num_robots=num_robots)

        self.ori_init = self.p.getQuaternionFromEuler([0, 3.141592654 / 2, 0])
        ori = self.p.getQuaternionFromEuler(np.random.rand(3) * np.pi)
        self.init_joints_angle = [0.23813124282501125, -1.66718850408224, -0.9366544912538485, -2.125146459392779,
                                  1.5701356825001347, 0.23663993167973682, 0.0, 0.0]
        self.camera_intrinsics = np.array(
            [[640, 0, 320],
             [0, 640, 320],
             [0, 0, 1]]
        )

        self.Tcam2end = np.array([
            [0, 0, 1, 0.2],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        self.Tend2cam = np.linalg.inv(self.Tcam2end)

        for d, uid in enumerate(self.robotId):
            self.move_joints(self.get_joints_angle((0, 0, 0.6), self.p.getQuaternionFromEuler((0, 3.14/2, 0)),
                                                   -0.5, uid), uid)

        time.sleep(1)
        obs = p.getLinkState(self.robotId[0], self.endEffectorIndex)
        self.init_end_pos = obs[4]
        self.init_end_ori = obs[5]
        self.objects = glob(os.path.join(self.meshPath, '*.obj'))
        self.obj_idx = []
        self.z_pos = []
        self.scale = []
        urdf, s, z = self.generate_urdf(self.objects[0])
        self.obj_idx.append(self.p.loadURDF(urdf, basePosition=[0., 0 * self.d, 0.3],
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

        init_camera_vector = np.array((1, 0, 0))  # x-axis
        init_up_vector = np.array((0, 0, 1))  # z-axis

        obs = p.getLinkState(self.robotId[0], self.endEffectorIndex)
        posEnd = obs[4]
        oriEnd = obs[5]

        rot_matrix = p.getMatrixFromQuaternion(oriEnd)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix_gripper = p.computeViewMatrix(posEnd + 0.05 * camera_vector, posEnd + 0.15 * camera_vector,
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

        return rgb, depth, posEnd, oriEnd

    def pixel2base(self, pixelIndex, angle, depthImage=None, depth=None, dx=0, dy=0):
        assert depthImage is not None or depth is not None, 'depthImage and depth cannot be None at the same time'
        xp, yp = pixelIndex
        xp = max(0, min(xp, 639 - dx * 2))
        yp = max(0, min(yp, 639 - dy * 2))

        if depthImage is not None:
            z = depthImage[yp, xp]  # meter
        else:
            z = depth - 0.07  # length of the jaw: about 0.12
        # print('z:', z)
        # Pobj2cam = np.dot(np.linalg.inv(self.cameraMat), [[xp+dx], [yp+dy], [1]]) * z  # meter
        Pobj2cam = np.array([[(xp + dx - 320) / 640], [(yp + dy - 320) / 640], [z]])

        Tcam2end = np.array([
            [0, 0, 1, 0.05],
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
        Tend2base = trans_mat_from_quat_pos(self.init_end_ori, self.init_end_pos)



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
            dx, dy = map(int, velocity)
            img = cv2.arrowedLine(img, (x, y), (x + dx, y + dy), depth, thickness=4)
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

    def main_action(self, v=0.02):
        img_tensor = torch.ones(1, 6, 480, 480).cuda()
        self.set_obj_movement(y=v)
        for i in range(6):
            color_image, depth_image, position_end, ori_end = self._env.update_camera()
            time_step = time.time()
            # depth_image = np.ones((480, 480)) * 0.7
            # depth_image[200:300, 370:400] = 0.4
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).cuda()
            img_tensor[:, i, :, :] = depth_tensor
            time.sleep(0.5)

        while position_end[2] > 0.25:
            while time.time() - time_step < 0.5:
                pass
            try:
                _, depth_image, position_end, ori_end = self._env.update_camera()
                time_step = time.time()
                # depth_image = np.ones((480, 480)) * 0.7
                # depth_image[200:300, 370:400] = 0.4
                depth_image = self.img_transform(depth_image, position_end, ori_end)
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
                        # cv2.imshow('grasp', x)
                        # cv2.waitKey(1)
                    else:
                        x = img[0, i, :, :]
                    plt.imshow(x)
                self.set_obj_movement()
                plt.show()
                self.set_obj_movement(y=v)

                # pos[2] = self._env.init_end_pos[2]
                self.move((quat, pos + [0, 0, 0.05]), jaw=1)
            except KeyboardInterrupt:
                break
        # self.set_obj_movement()
        # self.move((quat, pos + [0, 0, 0.1]), jaw=0.8)
        self.move((quat, pos + [0, 0, 0.05]), jaw=1)
        time.sleep(0.2)
        self.move((quat, pos), jaw=1)
        time.sleep(0.2)
        self.move((quat, pos), jaw=0)
        time.sleep(1)
        self.move((quat, pos + [0, 0, 0.3]), jaw=0)
        time.sleep(3)

        self._env.move_joints(self._env.get_joints_angle((0, 0, 0.6),
                                                         self._env.p.getQuaternionFromEuler((0, 3.14 / 2, 0)),
                                                         -0.5, self._env.robotId[0]), self._env.robotId[0])

    def img_transform(self, img, position_end, ori_end):
        Tend = trans_mat_from_quat_pos(ori_end, position_end)  # here, we got Tend2base
        Tend_init = trans_mat_from_quat_pos(self._env.init_end_ori, self._env.init_end_pos)
        Tcam = self._env.Tend2cam @ np.linalg.inv(Tend)
        Tcam_init = self._env.Tend2cam @ np.linalg.inv(Tend_init)
        x = np.arange(-0.3, 0.6, 0.6)
        x, y = np.meshgrid(x, x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        points = np.ones((4, x.size))
        points[0, :] = x
        points[1, :] = y
        # points[2, :] = 0
        points_cam = (Tcam @ points)[:3, :]
        points_init = (Tcam_init @ points)[:3, :]
        points_cam[2, :] = 1
        points_init[2, :] = 1
        points_cam = (self._env.camera_intrinsics @ points_cam)[:2, :].astype(np.float32)
        points_init = (self._env.camera_intrinsics @ points_init)[:2, :].astype(np.float32)
        MP = cv2.getPerspectiveTransform(points_cam.T, points_init.T)

        # plt.figure(2)
        # plt.imshow(img)
        # plt.title('original')
        img = cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
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
        return img[80:-80, 80:-80]

    def test_img_transform(self):
        userParameters = []
        abs_distance = 1.0
        userParameters.append(self._env.p.addUserDebugParameter("X", -abs_distance, abs_distance, 0))
        userParameters.append(self._env.p.addUserDebugParameter("Y", -abs_distance, abs_distance, 0))
        userParameters.append(self._env.p.addUserDebugParameter("Z", -abs_distance, abs_distance, 0.5))
        userParameters.append(self._env.p.addUserDebugParameter("roll", -np.pi, np.pi, 0.0))
        userParameters.append(self._env.p.addUserDebugParameter("pitch", -np.pi, np.pi, 1.5774))
        userParameters.append(self._env.p.addUserDebugParameter("yaw", -np.pi, np.pi, 0))
        userParameters.append(self._env.p.addUserDebugParameter('gripper', -1, 1, 0))
        while 1:

            obs = p.getLinkState(self._env.robotId[0], self._env.endEffectorIndex)
            posEnd = obs[4]
            # oriEnd = obs[5]
            action = []

            for para in userParameters:
                action.append(self._env.p.readUserDebugParameter(para))
            # posEnd = action[0:3]
            g = action[-1]
            oriEnd = p.getQuaternionFromEuler([action[3], action[4], action[5]])
            jointPose = self._env.get_joints_angle([action[0], action[1], action[2]], oriEnd, g, self._env.robotId[0])
            self._env.move_joints(jointPose, self._env.robotId[0])

            color_image, depth_image, position_end, ori_end = self._env.update_camera()
            depth_image = self.img_transform(depth_image, position_end, ori_end)
            cv2.imshow('depth', depth_image)
            cv2.waitKey(2)

    @torch.no_grad()
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
        velocity = np.round(-100 * pos[index[0], :, index[2], index[3]].detach().cpu().numpy()).astype(np.int32)
        quatObj2Base, posObj2Base = self._env.pixel2base((x + velocity[0], y + velocity[0]), index[1]*np.pi/16, None,
                                                         z_pose[index[0]].detach().cpu().item(), 80, 80)
        return quatObj2Base, posObj2Base, (x, y, index[1], z_pose_init[index[0]].item(),
                                           velocity)

    def move(self, tar_pose, jaw=0.8):
        jaw = 1 - 2 * jaw
        quat, pos = tar_pose
        joint_angle = self._env.get_joints_angle(pos, quat, jaw, self._env.robotId[0])
        self._env.move_joints(joint_angle, self._env.robotId[0])

    def set_obj_movement(self, x=0., y=0., z=0.):
        self._env.p.resetBaseVelocity(self._env.obj_idx[0], [x, y, z], [0, 0, 0])

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
    # model_path = './train/convTrans23_02_21_17_25_dynamic_attcg'
    model_path = './train/convTrans23_02_21_18_52_dynamic_attcg_removebn'
    dynamic_grasp = DynamicGrasp(model_path)
    # d = VirtualEnvironment(1)
    # d.calibrate_camera()
    # dynamic_grasp.test_img_transform()
    while 1:
        dynamic_grasp.main_action()

