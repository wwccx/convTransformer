from virtual_grasp.environment import VirtualEnvironment
import time
import torch
from ..load_model import build_model
import cv2
import numpy as np


class DynamicGrasp():
    def __init__(self, model_path):
        self._env = VirtualEnvironment(num_robots=1)
        self._env.reset_all_robots()
        self.ori_init = self._env.p.getQuaternionFromEuler([0, 3.141592654 / 2, 0])
        self.init_joints_angle = [0.23813124282501125, -1.66718850408224, -0.9366544912538485, -2.125146459392779,
                                  1.5701356825001347, 0.23663993167973682, 0.0, 0.0]
        self.position_end_init =  p.getLinkState(self._env.robotId[0], self._env.endEffectorIndex)[4]
        for d, uid in enumerate(self._env.robotId):
            self._env.move_joints(self.init_joints_angle, uid)
        time.sleep(1)

        self.prediction_model = build_model(model_path)
        self.prediction_model.eval()

    def main_action(self):
        img_tensor = torch.ones(1, 6, 640, 640)
        _, depth_image, position_end = self._env.update_camera()
        depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
        img_tensor[:, 0:6, :, :] = depth_tensor
        while position_end[2] > 0.01:
            try:
                _, depth_image, position_end = self._env.update_camera()
                depth_image = self.img_transform(depth_image, position_end)
                depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
                img_tensor[:, 0:5, :, :] = img_tensor[:, 1:6, :, :]
                img_tensor[:, 5, :, :] = depth_tensor
                tar_pose = self.prediction(img_tensor)
                self.move(tar_pose)
            except KeyboardInterrupt:
                break

    def img_transform(self, img, position_end):
        delta_x = position_end[1] - self.position_end_init[1]
        delta_y = position_end[0] - self.position_end_init[0]
        mat = np.array([[1, 0, delta_x], [0, 1, delta_y]], dtype=np.float32)
        img = cv2.warpAffine(img, mat, img.shape, borderMode=cv2.BORDER_REPLICATE)
        return img

    def prediction(self, img_tensor, depth_reso=5e-3, depth_bin=8):
        z_pose = torch.arange(depth_bin).cuda() * depth_reso + torch.min(input_img)
        res, pos = self.prediction_model(img_tensor, z_pose)
        s = list(res.shape)
        s[1:2] = [16, 2]
        sf = torch.nn.Softmax(dim=2)
        res = sf(res)[:, :, 1, ...]
        s[2:3] = []

        index = self.get_tensor_idx(res.shape, torch.argmax(res).detach().cpu().item())  # depth, angle, y, x
        quatObj2Base, posObj2Base = self._env.pixel2base((index[3], index[2]), index[1]*np.pi/16, z_pose[index[0]])

        return quatObj2Base, posObj2Base

    def move(self, tar_pose):
        quat, pos = tar_pose
        joint_angle = self._env.p.get_joint_angles(pos, quat, 0.5, self._env.robotId[0])
        self._env.move_joints(joint_angle, self._env.robotId[0])

    def set_obj_movement(self):
        pass

    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]


if __name__ == '__main__':
    model_path = 'virtual_grasp\pretrained_model\dynamic_grasp.pth'
    dynamic_grasp = DynamicGrasp(model_path)
    dynamic_grasp.main_action()