from camera import RS
from gripper import Gripper
from vel_control import VelController
import torch
import cv2
import numpy as np
from ..convTrans import convTransformer
from ..load_model import build_model
import os


class DynamicGrasp:
    def __init__(self, model_ptr) -> None:
        self.controller = VelController()
        self.camera = RS(640, 480)
        self.gripper = Gripper()
        self.net = None
        self.net_config = None
        self._load_model(model_ptr)
        self.init_pose = []
        self.queue_size = 10
        self.img_tensor = torch.zeros((self.queue_size, 480, 640)).cuda()
        self.start = 0
        self.end = self.net_config['time_slices'] + self.start
        self._init_img_tensor()
        
    def _load_model(model_ptr):
        ckps = os.listdir(model_ptr)
        ckps = [os.path.join(model_ptr, c) for c in ckps if c.endswith('.pth')]
        ckps.sort(key=os.path.getmtime)
        
        config = torch.load(ckps[0])['config']
        print(config)
        self.net = build_model(config).cuda()
        latest_ckp = ckps[-1]

        save_data = torch.load(latest_ckp)
        print(f'Epoch:{save_data['epoch']}')
        self.net.load_state_dict(save_data['model'])
        self.net.eval()
        self.net_config = config

    def _init_img_tensor():
        d, c = self.camera.get_img()
        #TODO: integrate the inpaint method
        d = torch.from_numpy(d).cuda().unsqueeze(0).repeat((self.end-start, 1, 1))
        self.img_tensor[self.start:self.end] = d

    def warp_perspective(self, img):
        #TODO: complete the warp_perspective code
        pose, rpy = self.controller.get_current_pose()
        mat = None
        return img

    def get_img_slice():
        if self.end > self.start:
            return self.img_tensor[self.start:self.end]
        else:
            return torch.stack(
                    (self.img_tensor[self.start:], self.[:end]), dim=0
                    )

    def update_img_tensor():
        d, c = self.camera.get_img()
        d = self.warp_perspective(d)
        self.start = (self.start + 1) % self.queue_size
        self.end = (self.end + 1) % self.queue_size
        d = torch.from_numpy(d).unsqueeze(0).cuda()
        self.img_tensor[self.end] = d


    def grasp_predict(self, depth_reso=5e-3, depth_bin=8):
        self.update_img_tensor()
        input_img = self.get_img_slice().unsqueeze(0)
        z_pose = torch.arange(depth_bin).cuda() * depth_reso + torch.min(input_img)
        res, pos = self.net(input_img, z_pose)
        s = list(res.shape)
        s[1:2] = [16, 2]
        sf = torch.nn.Softmax(dim=2)
        res = sf(res)[:, :, 1, ...]
        s[2:3] = []

        index = self.get_tensor_idx(res.shape, torch.argmax(res).detach().cpu().item())  # depth, angle, y, x

        position = self.pixel2cam(index[2], index[3], index[0])

        #TODO: add the pos prediction to the position
        return position, index[1]

    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]

    @staticmethod
    def pixel2cam(y, x, depth):
        pass
    






