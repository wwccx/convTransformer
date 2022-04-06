import cv2
import torch
import numpy as np


class Image(object):
    def __init__(self, depth, camera_pos, rgb=None, camera_intr=None):
        """
         create an image object
        :param depth: ndarray: (H, W)
        :param camera_pos: ndarray: (4, 4) transformation matrix
        :param rgb: ndarray: (3, H, W)
        :param camera_pos: ndarray: (3, 3) camera intrinsic matrix
        """
        self.depth_raw = depth
        self.H, self.W = depth.shape
        if rgb is not None:
            self.rgb_raw = rgb
        else:
            self.rgb_raw = np.zeros((3, self.H, self.W)).astype(np.float32)

        self.camera_pos = camera_pos if camera_pos is not None else np.eye(4)
        self.camera_intr = camera_intr

    def rgbd_image(self):
        if isinstance(self.depth_raw, np.ndarray):
            return np.stack(np.expand_dims(self.depth_raw, 0), self.rgb_raw, axis=0)
        elif isinstance(self.depth_raw, torch.Tensor):
            return torch.cat((self.depth_raw.unsqueeze(0), self.rgb_raw), dim=0)
        else:
            raise TypeError('Check the image data type')

    def __repr__(self):
        return 'Image(H: {}, W: {}, depth: {}, rgb: {})'.format(self.H, self.W, self.depth_raw.shape,
                                                                self.rgb_raw.shape)

    def device(self):
        if not isinstance(self.depth_raw, torch.Tensor):
            return 'cpu'
        return self.depth_raw.device.type

    def to(self, d):
        if not isinstance(d, torch.device):
            raise TypeError('device must be torch.device')
        if isinstance(self.depth_raw, np.ndarray):
            self.depth_raw = torch.from_numpy(self.depth_raw).to(d)
            self.rgb_raw = torch.from_numpy(self.rgb_raw).to(d)
            self.camera_pos = torch.from_numpy(self.camera_pos).to(d)
            if self.camera_intr is not None:
                self.camera_intr = torch.from_numpy(self.camera_intr).to(d)
        elif isinstance(self.depth_raw, torch.Tensor):
            self.depth_raw = self.depth_raw.to(d)
            self.rgb_raw = self.rgb_raw.to(d)
            self.camera_pos = self.camera_pos.to(d)
            if self.camera_intr is not None:
                self.camera_intr = self.camera_intr.to(d)
        else:
            raise TypeError('The image data must be np.ndarray or torch.Tensor')


class TSDF(object):
    def __init__(self, vpm, volume_size):
        """

        :param vpm: int voxels per meter
        :param volume_size: st the whole volume of scene (in meter)
        """
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.volumeTSDF = torch.ones(list((volume_size * vpm).astype(np.int))).to(self.device)
        self.volumeCoordinate = torch.ones(list(np.append(volume_size, 3).astype(np.int))).to(self.device)
        self.volumeRGB = torch.zeros(list(np.append(volume_size, 3).astype(np.int))).to(self.device).to(torch.float32)
        self.volumeCameraDis = torch.zeros(list((volume_size * vpm).astype(np.int))).to(self.device)
        # the distance along the z-axis between camera and voxel

        indexCoor = torch.meshgrid(*[torch.arange(i) for i in (volume_size * vpm)])
        indexCoor = torch.cat([x.unsqueeze(-1) for x in indexCoor], dim=-1).to(self.device)  # L H W 3 (x y z)
        self.baseCoor = indexCoor / vpm  # indexCoor in base shape: (L H W 3)
        # cameraCoor = inv(cameraPos) @ baseCoor
        # cameraPixelIndex = cameraIntr @ cameraCoor/cameraCoor.z
        self.voxelSize = 1 / vpm

    @torch.no_grad()
    def integrate(self, image):
        """
        integrate the image into the volume
        :param image: Image
        :return:
        """
        if image.device() != self.device:
            image.to(self.device)
        # print(self.baseCoor.shape, self.volumeTSDF.shape)
        baseCoor = torch.cat([self.baseCoor, torch.ones_like(self.volumeTSDF).unsqueeze(-1)], dim=-1)
        cameraCoor = torch.matmul(
            torch.inverse(image.camera_pos).to(torch.float32), baseCoor.unsqueeze(-1)
        )  # (L H W 4 1)
        # transform the camera coordinate to the base coordinate (4 4) @ (L H W 4 1) -> (L H W 4 1)

        cameraCoorZ = cameraCoor[:, :, :, 2:3, :]  # (L H W 1 1)
        cameraPixelIndexCoor = torch.floor(torch.matmul(image.camera_intr.to(torch.float32),
                                                        cameraCoor[..., :3, :] / cameraCoorZ)).squeeze()
        # get the corresponding pixel index of each voxel  (3 3) @ (L H W 3 1) = (L H W 3 1) = (L H W 3)

        validVoxelIndexCoor = torch.where(
            torch.logical_and(
                torch.logical_and(0 <= cameraPixelIndexCoor[..., 0], cameraPixelIndexCoor[..., 0] < image.W),
                torch.logical_and(0 <= cameraPixelIndexCoor[..., 1], cameraPixelIndexCoor[..., 1] < image.H)
            )
        )  # tuple of index, (x, y, z)
        validCameraPixelIndex = cameraPixelIndexCoor[validVoxelIndexCoor][:, :2].to(torch.long)
        # (N 3) -> (N 2) (pixel_x, pixel_y, | 1)
        self.volumeCameraDis[validVoxelIndexCoor] = image.depth_raw[validCameraPixelIndex[:, 1],
                                                                    validCameraPixelIndex[:, 0]].to(torch.float32)

        depth_diff = torch.clamp(self.volumeCameraDis - cameraCoorZ.squeeze(),
                                 min=-5 * self.voxelSize, max=5 * self.voxelSize) / (5 * self.voxelSize)
        self.volumeTSDF[validVoxelIndexCoor] = self.volumeTSDF[validVoxelIndexCoor] * 0.2 + depth_diff[
            validVoxelIndexCoor] * 0.8
        self.volumeRGB[validVoxelIndexCoor] = self.volumeTSDF[
                                                  validVoxelIndexCoor] * 0.2 + image.rgb_raw[
                                                  validCameraPixelIndex[:, 1], validCameraPixelIndex[:, 0]] * 0.8


if __name__ == '__main__':
    v = TSDF(vpm=100, volume_size=np.array([1, 1, 1]))
    import time
    t = time.time()
    for i in range(1000):
        a = Image(np.ones((480, 640)), np.eye(4),
                  camera_intr=np.array([500, 0, 320, 0, 500, 240, 0, 0, 1]).reshape(3, 3))
        v.integrate(a)
        print(i)
    torch.cuda.synchronize()
    print('fps:', 1000/(time.time() - t), 'seconds:', (time.time() - t))


