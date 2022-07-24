import cv2
import numpy
import torch
import numpy as np
from hardware.camera import RS
from hardware.Ursocket import UrSocket
from torch.utils.cpp_extension import load
import tf.transformations as tft
from torchvision.transforms.functional import resize
from PIL.Image import BICUBIC
import time


class Image(object):
    def __init__(self, depth, camera_pos, rgb=None, camera_intr=None):
        """
         create an image object
        :param depth: ndarray: (H, W)
        :param camera_pos: ndarray: (4, 4) transformation matrix
        :param rgb: ndarray: (H, W, 3)
        :param camera_pos: ndarray: (3, 3) camera intrinsic matrix
        """
        self.depth_raw = self.in_paint(depth)
        # self.depth_raw = depth
        self.H, self.W = depth.shape
        if rgb is not None:
            self.rgb_raw = rgb.astype(np.float32)
        else:
            self.rgb_raw = np.zeros((self.H, self.W, 3)).astype(np.float32)

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
            return torch.device('cpu')
        return self.depth_raw.device

    def to(self, d):
        if not isinstance(d, torch.device):
            raise TypeError('device must be torch.device')
        if isinstance(self.depth_raw, np.ndarray):
            self.depth_raw = torch.from_numpy(self.depth_raw).to(d).unsqueeze(0).unsqueeze(0)
            self.rgb_raw = torch.from_numpy(self.rgb_raw).to(d).permute(2, 0, 1).unsqueeze(0)
            self.camera_pos = torch.from_numpy(self.camera_pos).to(d)
            if self.camera_intr is not None:
                self.camera_intr = torch.from_numpy(self.camera_intr).to(d)
        elif isinstance(self.depth_raw, torch.Tensor):
            self.depth_raw = self.depth_raw.to(d).unsqueeze(0).unsqueeze(0)
            self.rgb_raw = self.rgb_raw.to(d).unsqueeze(0)
            self.camera_pos = self.camera_pos.to(d)
            if self.camera_intr is not None:
                self.camera_intr = self.camera_intr.to(d)
        else:
            raise TypeError('The image data must be np.ndarray or torch.Tensor')

    def resize(self, shape: list):
        if isinstance(self.depth_raw, np.ndarray):
            self.depth_raw = cv2.resize(self.depth_raw, tuple(shape))
            self.rgb_raw = cv2.resize(self.rgb_raw, tuple(shape))
        elif isinstance(self.depth_raw, torch.Tensor):
            self.depth_raw = resize(self.depth_raw, shape, BICUBIC)
            self.rgb_raw = resize(self.rgb_raw, shape, BICUBIC)
        else:
            raise TypeError('The image data must be np.ndarray or torch.Tensor')
        self.H, self.W = shape

    @staticmethod
    def in_paint(depth_img, val=0.1):
        depth_img = np.array(depth_img).astype(np.float32)
        # depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0.2)
        # depth_img = cv2.Laplacian(depth_img, cv2.CV_32F, ksize=3)
        depth_img = cv2.copyMakeBorder(depth_img, 10, 10, 10, 10, cv2.BORDER_DEFAULT)
        mask = (depth_img <= val).astype(np.uint8)

        depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
        # Back to original size and value range.
        depth_img = depth_img[10:-10, 10:-10]

        return depth_img

    def depth(self):
        if isinstance(self.depth_raw, np.ndarray):
            return self.depth_raw
        elif isinstance(self.depth_raw, torch.Tensor):
            return self.depth_raw.cpu().squeeze().numpy().astype(np.float32)
        else:
            raise TypeError('Check the image data type')

    def depth_inte(self):
        if isinstance(self.depth_raw, np.ndarray):
            return torch.from_numpy(self.depth_raw).to(torch.float32).squeeze()
        elif isinstance(self.depth_raw, torch.Tensor):
            return self.depth_raw.squeeze().to(torch.float32)
        else:
            raise TypeError('Check the image data type')

    def rgb_inte(self):
        if isinstance(self.rgb_raw, np.ndarray):
            return torch.from_numpy(self.rgb_raw).to(torch.float32).squeeze()
        elif isinstance(self.rgb_raw, torch.Tensor):
            return self.rgb_raw.squeeze().permute(1, 2, 0).to(torch.float32)
        else:
            raise TypeError('Check the image data type')

    def color(self):
        if isinstance(self.rgb_raw, np.ndarray):
            return self.rgb_raw
        elif isinstance(self.rgb_raw, torch.Tensor):
            return self.rgb_raw.permute(0, 2, 3, 1).contiguous().cpu().squeeze().numpy()
        else:
            raise TypeError('Check the image data type')


class Grasp(object):
    def __init__(self,
                 image: Image,
                 net=None,
                 *args,
                 **kwargs
                 ):

        self.image = image
        self.camera_pos = image.camera_pos
        self.camera_intr = image.camera_intr
        self._grasp = {
            'pixel_idx': (0, 0),
            'depth': 0,
            'angle': 0,
            'grasp_pose': np.eye(4),
            'trans_vec': np.zeros(3),
            'rotation_mat': np.eye(3)
        }
        if net is not None:
            self.plan_grasp(net, *args, **kwargs)

    def plan_grasp(self,
                   net,
                   depth_resolution=5e-3,
                   depth_bin=8,
                   network_depth_bias=0.,
                   normalize_depth=True,
                   **kwargs):
        """
        :return:
        """
        sf = torch.nn.Softmax(dim=2)
        self.image.to(torch.device('cuda:0'))
        d = self.image.depth_raw
        depth = torch.arange(depth_bin).cuda() * depth_resolution + torch.min(d)

        start_time = time.time()
        if normalize_depth:
            d0 = (d - d.mean()) / d.std()
            depth0 = (depth - d.mean()) / d.std()
            res = net(d0, depth0 - network_depth_bias, **kwargs)
        else:
            res = net(d, depth - network_depth_bias, **kwargs)
        torch.cuda.synchronize()
        plan_time = time.time() - start_time
        s = list(res.shape)
        s[1:2] = [16, 2]
        res = res.reshape(s)
        res = sf(res)[:, :, 1, ...]  # B, 16, H, W
        s[2:3] = []
        index = self.get_tensor_idx(res.shape, torch.argmax(res).detach().cpu().item())  # depth, angle, y, x
        print('index:', index, 'quality:', '{:.4f}'.format(res[index[0], index[1], index[2], index[3]].item()))
        self._grasp['pixel_idx'] = index[-2], index[-1]
        self._grasp['depth'] = depth[index[0]].item()
        self._grasp['angle'] = index[1] * np.pi / 16
        self._grasp['quality'] = res[index[0], index[1], index[2], index[3]].item()
        self._grasp['time'] = plan_time

    def visualization(self, grasp_radius=48, grasp_color=(1, 0, 0), grasp_thickness=2,
                      pixel_wise_stride=8, pixel_bias=48):
        # self.image.to(torch.device('cuda:0'))
        color_img = self.image.color()
        depth_img = self.image.depth()
        # depth = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))

        h_idx, w_idx = self._grasp['pixel_idx']
        angle = self._grasp['angle']
        pose = self._grasp['depth']
        x = pixel_bias + pixel_wise_stride * w_idx
        y = pixel_bias + pixel_wise_stride * h_idx
        x1 = x + grasp_radius * np.cos(angle)
        y1 = y + grasp_radius * np.sin(angle)
        x2 = x + grasp_radius * np.cos(angle + np.pi)
        y2 = y + grasp_radius * np.sin(angle + np.pi)
        # print('index:', b_idx, c_idx, h_idx, w_idx)
        # print('Quality:', res[b_idx, c_idx, h_idx, w_idx])
        color_img = cv2.circle(color_img, (x, y), int(grasp_thickness*1.5), grasp_color, -1)
        color_img = cv2.arrowedLine(color_img, (int(x1), int(y1)), (int(x2), int(y2)), grasp_color, thickness=grasp_thickness)
        color_img = cv2.arrowedLine(color_img, (int(x2), int(y2)), (int(x1), int(y1)), grasp_color, thickness=grasp_thickness)

        depth_img = cv2.circle(depth_img, (x, y), int(grasp_thickness*1.5), pose, -1)
        depth_img = cv2.arrowedLine(depth_img, (int(x1), int(y1)), (int(x2), int(y2)), pose, thickness=grasp_thickness)
        depth_img = cv2.arrowedLine(depth_img, (int(x2), int(y2)), (int(x1), int(y1)), pose, thickness=grasp_thickness)

        return color_img, depth_img

    def result(self):
        return self._grasp

    @staticmethod
    def get_tensor_idx(shape, arg_max):
        idx = []
        for i in shape[::-1]:
            idx.append(arg_max % i)
            arg_max = arg_max // i
        return idx[::-1]


class TSDF(object):
    def __init__(self, vpm, volume_size):
        """

        :param vpm: int, voxels per meter
        :param volume_size: ndarray, the whole volume of scene (in meter)
        """
        self.volume_size = torch.from_numpy(volume_size).cuda()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.volumeTSDF = torch.ones(list((volume_size * vpm).astype(int))).to(self.device).to(torch.float32)
        # self.volumeCoordinate = torch.ones(list(np.append(volume_size, 3).astype(np.int))).to(self.device)
        self.volumeRGB = torch.zeros(
            list(np.append(volume_size * vpm, 3).astype(int))
        ).to(self.device).to(torch.float32)
        self.volumeCameraDis = torch.zeros(list((volume_size * vpm).astype(int))).to(self.device)
        # the distance along the z-axis between camera and voxel
        self.volume_origin = torch.tensor([-0.8, -0.4, -0.01]).cuda()
        indexCoor = torch.meshgrid(*[torch.arange(i) for i in (volume_size * vpm)])
        indexCoor = torch.cat([x.unsqueeze(-1) for x in indexCoor], dim=-1).to(self.device)  # L H W 3 (x y z)
        self.baseCoor = indexCoor / vpm + self.volume_origin  # indexCoor in base shape: (L H W 3)
        # cameraCoor = inv(cameraPos) @ baseCoor
        # cameraPixelIndex = cameraIntr @ cameraCoor/cameraCoor.z
        self.voxelSize = 1 / vpm
        self._rendering_cuda = load(name="rendering",
                                    extra_include_paths=["include"],
                                    sources=["./cpp/rendering.cpp", "./kernel/rendering.cu"],
                                    verbose=True)

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
                torch.logical_and(0 <= cameraPixelIndexCoor[..., 0], cameraPixelIndexCoor[..., 0] < 1 * image.W),
                torch.logical_and(0 <= cameraPixelIndexCoor[..., 1], cameraPixelIndexCoor[..., 1] < 1 * image.H)
            )
        )  # tuple of index, (x, y, z)
        validCameraPixelIndex = cameraPixelIndexCoor[validVoxelIndexCoor][:, :2].to(torch.long)
        # (N 3) -> (N 2) (pixel_x, pixel_y, | 1)
        self.volumeCameraDis = torch.zeros_like(self.volumeTSDF)
        self.volumeCameraDis[validVoxelIndexCoor] = image.depth_inte()[validCameraPixelIndex[:, 1],
                                                                    validCameraPixelIndex[:, 0]].to(torch.float32)

        depth_diff = torch.clamp(self.volumeCameraDis - cameraCoorZ.squeeze(),
                                 min=-5 * self.voxelSize, max=5 * self.voxelSize) / (5 * self.voxelSize)
        self.volumeTSDF[validVoxelIndexCoor] = self.volumeTSDF[validVoxelIndexCoor] * 0.2 + depth_diff[
            validVoxelIndexCoor] * 0.8
        self.volumeRGB[validVoxelIndexCoor] = self.volumeRGB[
                                                  validVoxelIndexCoor
                                              ] * 0.2 + image.rgb_inte()[
                                                        validCameraPixelIndex[:, 1], validCameraPixelIndex[:, 0], :
                                                        ] * 0.8

    @torch.no_grad()
    def rendering(self, camera_extrinsic, camera_intrinsic, shape=None):
        """
        render the volume into an image
        :param camera_extrinsic: ndarray, tensor, shape (4, 4) camera extrinsic matrix
        :param camera_intrinsic: ndarray, tensor, shape (3, 3) camera intrinsic matrix
        :return: tensor, rendered image
        """
        try:
            if isinstance(camera_extrinsic, numpy.ndarray):
                camera_extrinsic = torch.from_numpy(camera_extrinsic).to(self.device)
                camera_intrinsic = torch.from_numpy(camera_intrinsic).to(self.device)
            elif camera_extrinsic.device() != self.device:
                camera_extrinsic = camera_extrinsic.to(self.device)
                camera_intrinsic = camera_intrinsic.to(self.device)

            camera_extrinsic = camera_extrinsic.to(torch.float32)
            camera_intrinsic = camera_intrinsic.to(torch.float32)

        except Exception:
            raise TypeError("camera_extrinsic should be ndarray or tensor")
        if not shape:
            shape = (int(2 * camera_intrinsic[1, 2]), int(2 * camera_intrinsic[0, 2]))  # (H, W)
        H, W = shape
        origin = camera_extrinsic[:3, 3:4].unsqueeze(0)  # B, 3, 1
        pixel_index = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1
        ).unsqueeze(0).expand(origin.shape[0], H, W, 2).to(self.device)  # B, H, W, 2
        pixel_index = torch.cat(
            [pixel_index[..., 1:2], pixel_index[..., 0:1], torch.ones_like(pixel_index[..., :1])], dim=-1
        ).to(torch.float32)  # B, H, W, 3

        inv_intr = torch.inverse(camera_intrinsic).expand((origin.shape[0], 1, 1, 3, 3))  # B, 1, 1, 3, 3
        direction = torch.matmul(inv_intr, pixel_index.unsqueeze(-1))  # B, H, W, 3, 1
        direction = torch.cat(
            [direction, torch.ones_like(direction[..., -1:, :])], dim=-2
        )  # B, H, W, 4, 1
        direction = camera_extrinsic.expand((origin.shape[0], 1, 1, 4, 4)) @ direction
        # B, 3, 3 @ B, H, W, 3, 1 = B, H, W, 3, 1
        direction = direction[..., :3, :] - origin.expand((origin.shape[0], 1, 1, 3, 1))  # B, H, W, 3
        direction = direction * self.voxelSize
        direction = torch.cat([direction, torch.zeros_like(direction) + origin.expand((origin.shape[0], 1, 1, 3, 1))],
                              dim=-2)  # B, H, W, 6, 1
        image = self._render(origin.squeeze().to(torch.float32), direction.squeeze(), shape)
        return image

    def _render(self, origin, direction, shape):
        B = 1
        H, W = shape
        xmin, ymin, zmin = self.volume_origin.cpu().numpy()
        xmax, ymax, zmax = self.volume_origin.cpu().numpy() + self.volume_size.cpu().numpy()
        image = torch.zeros(B, H, W, 4).to(self.device).to(torch.float32)
        vL, vH, vW = self.volumeTSDF.shape
        self._rendering_cuda.rendering(image, direction, self.volumeRGB, self.volumeTSDF, B,
                                       H, W, vH, vW, int(0.999 + 1/self.voxelSize), self.voxelSize,
                                       xmin, xmax, ymin, ymax, zmin, zmax)
        return image


def vec2mat(rot_vec, trans_vec):
    """
    :param rot_vec: list_like [a, b, c]
    :param trans_vec: list_like [x, y, z]
    :return: transform mat 4*4
    """
    theta = np.linalg.norm(rot_vec, 2)
    if theta:
        rot_vec /= theta
    out_operator = np.array([
        [0, -rot_vec[2], rot_vec[1]],
        [rot_vec[2], 0, -rot_vec[0]],
        [-rot_vec[1], rot_vec[0], 0]
    ])
    rot_vec = np.expand_dims(rot_vec, 0)
    rot_mat = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * rot_vec.T.dot(rot_vec) + np.sin(
        theta) * out_operator

    trans_vec = np.expand_dims(trans_vec, 0).T

    trans_mat = np.vstack(
        (np.hstack((rot_mat, trans_vec)), [0, 0, 0, 1])
    )

    return trans_mat


def get_extrinsic(alpha, beta, radius, camera_extrinsic):

    origin = np.array([0, 0, radius])
    targetT = tft.quaternion_from_euler(*[alpha, beta, 0, 'rxyz'])
    targetT = tft.quaternion_matrix(targetT)
    targetT[:3, 3] = origin - targetT[:3, 2] * radius

    return camera_extrinsic @ targetT


def test(net=None):
    vpm = 400
    v = TSDF(vpm=vpm, volume_size=np.array([0.8, 0.8, 0.3]))
    import time
    import open3d as o3d

    from skimage import measure
    rs = RS(640, 480)
    # Ur = UrSocket()
    init_pose = np.array([-0.3504593, -0.0419833473, 0.44227438, 2.1036984080, 2.250377223, 0.097387733])
    # Ur.change_pose(init_pose)
    Tcam2end = np.array([
        [0.99770124, -0.06726973, -0.00818663, -0.02744465],
        [0.06610884, 0.99272747, -0.10060715, -0.10060833],
        [0.01489491, 0.09983467, 0.99489255, -0.15038112],
        [0., 0., 0., 1., ]
    ])

    t = time.time()
    img_num = 5
    for i in range(img_num):
        depth, rgb = rs.get_img()
        # _ = rs.get_coordinate(240, 240)
        pose = np.array([-0.3504593, -0.0419833473, 0.64227438, 2.1036984080, 2.250377223, 0.097387733])
        print(pose)
        Tend2base = vec2mat(pose[3:6], pose[0:3])
        Tcam2base = np.matmul(Tend2base, Tcam2end)
        a = Image(depth / 1000, Tcam2base, rgb / 255,
                  camera_intr=np.array([614.887, 0, 328.328, 0, 614.955, 236.137, 0, 0, 1]).reshape(3, 3))
        v.integrate(a)
        # Ur.change_pose(init_pose + np.array([0, -0.02, 0, 0, 0, 0])*i)
        # Ur.change_pose(init_pose + np.array([-0.02, 0, 0, 0, 0, -0.1]) * 0)
        # time.sleep(1)
        print(i)
    torch.cuda.synchronize()
    print('fps:', img_num / (time.time() - t), 'seconds:', (time.time() - t))
    volume = v.volumeTSDF.cpu().numpy()
    # np.savez('volume.npz', **{'tsdf': volume, 'weight':np.ones_like(volume)})
    rgb_volume = v.volumeRGB.cpu().numpy()
    verts, faces, normals, values = measure.marching_cubes(volume)
    vert_idx = np.round(verts).astype(int)
    rgb_val = rgb_volume[vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2], :]
    verts = verts / vpm + v.volume_origin.cpu().numpy()
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(verts)
    pt.colors = o3d.utility.Vector3dVector(rgb_val)
    # o3d.visualization.draw_geometries([pt])
    pose = np.array([-0.3504593, -0.0419833473, 0.44227438, 2.1036984080, 2.250377223, 0.097387733])
    print(pose)
    Tend2base = vec2mat(pose[3:6], pose[0:3])
    # Tcam2base = Tend2base @ Tcam2end
    Tcam2base = np.array([
        [0, 1, 0, -0.46],
        [1, 0, 0, -0.09],
        [0, 0, -1, 0.58],
        [0., 0., 0., 1., ]
    ])
    Tcam2base = get_extrinsic(0, 0.35, 0.58, Tcam2base)
    h, w = 400, 400
    img = v.rendering(Tcam2base, np.array([614.887, 0, w // 2, 0, 614.955, h // 2, 0, 0, 1]).reshape(3, 3), (h, w))
    img = img
    from matplotlib import pyplot as plt
    img1 = img.squeeze().cpu().numpy()
    print(np.max(img1), np.min(img1))
    img_net = Image(img1[:, :, -1], Tcam2base, img1[:, :, :3])
    # img_net.to(torch.device('cuda'))
    # img_net.resize([320, 320])
    grasp = Grasp(img_net, net, depth_bin=10)

    result = grasp.result()
    depth = result['depth']
    q = result['quality']
    color_img, depth_img = grasp.visualization()
    plt.subplot(1, 2, 1)
    plt.imshow(color_img)
    plt.subplot(1, 2, 2)
    plt.imshow(depth_img)
    plt.title('depth: {:.4f}, quality: {:.4f}'.format(depth, q))
    plt.show()

    plt.figure(1)
    plt.imshow(img1[:, :, -1])
    plt.figure(2)
    plt.imshow(img1[:, :, 0:3])
    plt.show()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0])
    mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=v.volume_origin.cpu().numpy())
    o3d.visualization.draw_geometries([mesh, mesh_frame, mesh_frame1, pt])


if __name__ == '__main__':
    from convTrans import convTransformer
    import os

    net = convTransformer(in_chans=1, num_classes=32, embed_dim=128, depths=(8,), num_heads=(8,),
                          patch_embedding_size=(8, 8), fully_conv_for_grasp=True, window_size=(3, 3)).cuda()
    save_path = './train/mixupFcParaRotation'
    check_points = os.listdir(save_path)
    check_points = [os.path.join(save_path, ckpt) for ckpt in check_points if ckpt.endswith('.pth')]
    check_points.sort(key=os.path.getmtime)
    latest_ckpt = check_points[-1]
    s = torch.load(latest_ckpt)
    net.load_state_dict(s['model'])
    net.eval()
    test(net)



