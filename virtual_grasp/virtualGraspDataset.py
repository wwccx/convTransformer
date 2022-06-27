import torch
from matplotlib import pyplot as plt
from virtual_grasp.environment import VirtualEnvironment
import numpy as np
import time
from glob import glob
import os


class VirtualGraspDataset():
    def __init__(self, supervise_model, num_robots=8, data_len=300):
        super().__init__()
        self._env = VirtualEnvironment(num_robots)
        self.supervise_model = supervise_model
        self.ori_init = self._env.p.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.init_joints_angle = [0.23813124282501125, -1.66718850408224, -0.9366544912538485, -2.125146459392779,
                                  1.5701356825001347, 0.23663993167973682, 0.0, 0.0]
        for d, uid in enumerate(self._env.robotId):
            self._env.move_joints(self.init_joints_angle, uid)
        time.sleep(1)

        self.objects = glob(os.path.join(self._env.meshPath, '*.obj'))
        self.objects.sort()
        self.obj_count = 0
        self.data_len = min(data_len, len(self.objects))
        self.objects = self.objects[:self.data_len]
        self.sf_fun = torch.nn.Softmax(2)

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        if self.obj_count >= 300:
            self._env.reset_all_robots()
            self.obj_count = 0
        idx_obj = np.random.randint(0, self.data_len, self._env.numRobots)
        obj_idx = []
        z_pos = []
        scale = []
        for idx_robot in range(self._env.numRobots):
            urdf, s, z = self._env.generate_urdf(self.objects[idx_obj[idx_robot]])
            obj_idx.append(self._env.p.loadURDF(urdf, basePosition=[0.0, idx_robot * self._env.d, z * s * 2],
                                               baseOrientation=(0.400000, 0.000000, 0.0, 1.0),
                                               globalScaling=s))
            z_pos.append(z)
            scale.append(s)
            self._env.p.changeDynamics(idx_obj[idx_robot], -1, mass=0, lateralFriction=15,
                                       restitution=0.98, rollingFriction=15, spinningFriction=15,
                                       contactStiffness=1e9, contactDamping=5)
        self.obj_count += self._env.numRobots
        self.supervise_model.eval()
        self.random_place_objects(obj_idx, z_pos, scale)
        time.sleep(1)
        while 1:
            try:
                rgb_image, depth_image = self._env.update_camera()
                pos_grasp_init = []
                ori_grasp_init = []
                grasp_result = []

                for idx_robot in range(self._env.numRobots):
                    pos, ori = self._env.p.getBasePositionAndOrientation(obj_idx[idx_robot])
                    pos_grasp_init.append(pos)
                    ori_grasp_init.append(ori)
                    grasp_result.append(self.plan_grasp(depth_image[idx_robot], rgb_image[idx_robot]))
                    # print(grasp_result[idx_robot][5])
                return self.execute_grasps(grasp_result, obj_idx, pos_grasp_init)

            except Exception as e:
                print(e)
                self.random_place_objects(obj_idx, z_pos, scale)
                for d, uid in enumerate(self._env.robotId):
                    self._env.move_joints(self.init_joints_angle, uid)

    def plan_grasp(self, depth_image, rgb_image):
        # from matplotlib import pyplot as plt
        # plt.imshow(depth_image)
        # plt.show()
        depth_image = torch.from_numpy(depth_image).to(torch.float32).cuda().unsqueeze(0).unsqueeze(0)
        depth_min = torch.min(depth_image)
        depth_max = torch.max(depth_image)
        # print(depth_max, depth_min)
        pose = (3 - torch.arange(15).cuda()) * 0.02 + depth_max
            # pose = pose.unsqueeze(0).cuda()
        dense_quality = self.supervise_model(depth_image, pose)
        shape = dense_quality.shape
        dense_quality = self.sf_fun(dense_quality.view(shape[0], -1, 2, shape[2], shape[3]))[:, :, 1, ...]
        shape = dense_quality.shape
        # print(dense_quality.shape)
        flatten_max_index = torch.argmax(dense_quality).detach().cpu().numpy()
        max_index = []
        for dim in shape[::-1]:
            max_index.append(flatten_max_index % dim)
            flatten_max_index //= dim
        if torch.rand(1) < 1:
            max_index = max_index[::-1]
            grasp = [
                48 + max_index[2] * 8,
                48 + max_index[3] * 8,
                - max_index[1] * np.pi / 16,
                72
            ]
            depth = pose[max_index[0]].cpu()
        else:
            max_index = max_index[::-1]
            grasp = [
                48 + max_index[2] * 8,
                48 + max_index[3] * 8,
                torch.randint(16, (1,)).item() * np.pi / 16,
                72
            ]
            depth = pose[torch.randint(15, (1,))].cpu()

        pos, quat, color_img, width = self._env.grasp_img2real(rgb_image, depth_image, grasp, depth=depth)
        img = depth_image[
              0, 0, max_index[2] * 8:max_index[2] * 8 + 96, max_index[3] * 8:max_index[3] * 8 + 96
              ].cpu()
        # print(max_index[1])
        return pos, quat, color_img, width, img, depth, max_index[1]
    
    def random_place_objects(self, obj_idx, z_pos, scale):

        for i, obj in enumerate(obj_idx):
            randomOri = self._env.p.getQuaternionFromEuler((np.random.rand(3) - 0.5) * np.pi * 2)
            self._env.p.resetBasePositionAndOrientation(
                obj, [0.0, i * self._env.d, z_pos[i] * scale[i] * 1.8], randomOri
            )
        time.sleep(0.5)

    def execute_grasps(self, grasp_result, obj_idx, pos_grasp_init):
        img_tensor = torch.zeros(self._env.numRobots, 1, 96, 96)
        pose_tensor = torch.zeros(self._env.numRobots)
        metric_tensor = torch.zeros(self._env.numRobots)
        mask_tensor = torch.zeros(self._env.numRobots, 32)
        # go straight down
        for step in range(40):
            for d, uid in enumerate(self._env.robotId):
                self._env.move_joints(
                    self._env.get_joints_angle(grasp_result[d][0] + [0, d * self._env.d, 0.5 * (1 - step / 40)],
                                               grasp_result[d][1], grasp_result[d][3], uid), uid)
            time.sleep(0.01)
        time.sleep(0.5)
        # close the gripper
        for d, uid in enumerate(self._env.robotId):
            self._env.move_joints(
                self._env.get_joints_angle(grasp_result[d][0] + [0, d * self._env.d, 0], grasp_result[d][1], 1, uid),
                uid)
        time.sleep(0.5)
        # elevate the object
        for d, uid in enumerate(self._env.robotId):
            self._env.move_joints(
                self._env.get_joints_angle(grasp_result[d][0] + [0, d * self._env.d, 0.4], grasp_result[d][1], 1, uid),
                uid)
        time.sleep(1.0)
        for objRobot in range(self._env.numRobots):
            pos_grasp, ori_grasp = self._env.p.getBasePositionAndOrientation(obj_idx[objRobot])
            if pos_grasp[2] - pos_grasp_init[objRobot][2] > 0.1:
                metric_tensor[objRobot] = 1
            mask_tensor[objRobot, 2*grasp_result[objRobot][6]:2*grasp_result[objRobot][6] + 2] = 1
            img_tensor[objRobot, ...] = grasp_result[objRobot][4]
            pose_tensor[objRobot, ...] = grasp_result[objRobot][5]
        # for ind in obj_idx:
            self._env.p.removeBody(obj_idx[objRobot])
        # time.sleep(1.0)
        for d, uid in enumerate(self._env.robotId):
            self._env.move_joints(self.init_joints_angle, uid)
        # for d, uid in enumerate(self._env.robotId):
        #     linkState = self._env.p.getJointState(uid, 2)[0]
        #     if linkState > 0:
        #         self._env.resetJointPoses(uid)
        #         time.sleep(0.2)
        #         self._env.move_joints(self.init_joints_angle, uid)
        # print(pose_tensor)
        # print(torch.max(img_tensor), torch.min(img_tensor))
        return img_tensor.to(torch.float32), pose_tensor.to(torch.float32),\
               metric_tensor.to(torch.long), mask_tensor.to(torch.long)


if __name__ == '__main__':
    from convTrans import convTransformer

    network = convTransformer(in_chans=1, num_classes=32,
                              embed_dim=96, depths=(2, 6),
                              num_heads=(3, 12),
                              patch_embedding_size=(4, 4),
                              fully_conv_for_grasp=True).cuda()
    network.load_state_dict(
        torch.load('gqTrans.pth')['model']
    )
    d = VirtualGraspDataset(network, num_robots=8)
    print(d._env._urdfRoot)
    # for x in d.objects:
    #     d._env.generate_urdf(x)
    # l = tdata.DataLoader(d, batch_size=1)
    # d.__getitem__(3)
    for i in range(len(d)):
        data = d[i]
        print(i)
