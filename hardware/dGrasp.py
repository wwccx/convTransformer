import torch
import camera
# import Ursocket
import numpy as np
# from models.common import post_process_output
from skimage.feature import peak_local_max
# from models.VGG_GA3 import VGG
import time
import cv2
import gripper
# from mygqcnn.resNet import ResNet, Bottleneck
# from gqcnn_server.network import GQCNN
from models.ggscnn import GGSCNN

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from moveit_commander.conversions import pose_to_list
import numpy as np
import tf.transformations as tftrans


class Grasp:
    def __init__(self, hardware=False):
        self.hardware = hardware
        self.init_pose = [-0.35045893, -0.05225801, 0.53983334, 2.14450455, 2.20997734, 0.09563743]

        if self.hardware:
            # joint_state_topic = ['joint_states:=/robot/joint_states']
            # moveit_commander.roscpp_initialize(joint_state_topic)
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node('dynamicGrasping',
                            anonymous=False)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            group_name = "manipulator"
            self.group = moveit_commander.MoveGroupCommander(group_name)
            self.target_pose_publisher = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray,
                                                         queue_size=100)
            # print(self.group.get_current_joint_values())

            self.planning_frame = self.group.get_planning_frame()
            self.eef_link = self.group.get_end_effector_link()
            self.group_names = self.robot.get_group_names()
            self.init_pose = [-0.35045893, -0.05225801, 0.53983334, 2.14450455,  2.20997734,  0.09563743]
            self.place_pose = [-0.38977676, -0.45377277, 0.36905362, 2.11670662, 2.23461536, 0.10700127]

        self.camera = camera.RS(640, 480)
        self.device = torch.device("cuda:0")

        self.Tcam2end = np.array([
            [0.99770124, -0.06726973, -0.00818663, -0.02744465],
            [0.06610884, 0.99272747, -0.10060715, -0.10060833],
    [0.01489491, 0.09983467, 0.99489255, -0.15038112],
            [0., 0., 0., 1., ]
        ])
        self.Ttool2end = np.vstack((
            np.hstack(
                (self.quat2RotMat([0.399, -0.155, 0.369, 0.825]),
                 np.array([[-0.001], [0.038], [-0.248]]))
            ), np.array([[0, 0, 0, 1]])
        ))
        self.Tend2tool = np.linalg.inv(self.Ttool2end)
        self.Rend2cam = np.array([[0.99770124,  0.06610884,  0.01489491],
                                  [-0.06726973,  0.99272746,  0.09983467],
                                  [-0.00818663, -0.10060715,  0.99489254]])

        self.ggsnet = GGSCNN()

    @staticmethod
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

    @staticmethod
    def in_paint(depth_img):
        depth_img = np.array(depth_img).astype(np.float32)
        depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_img == 0).astype(np.uint8)

        depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
        # Back to original size and value range.
        depth_img = depth_img[1:-1, 1:-1]

        return depth_img

    @staticmethod
    def quat2RotMat(quat):
        # print('quat:', quat)
        q = np.array(quat)
        n = np.dot(q, q)
        if n < np.finfo(q.dtype).eps:
            return np.identity(3)
        q /= n
        (x, y, z, w) = q
        rotMat = np.array(
            [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
             [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
             [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
             ], dtype=q.dtype
        )
        return rotMat

    @staticmethod
    def rotMat2Quat(R):
        w = np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1) / 2
        x = -(R[1, 2] - R[2, 1]) / w / 4
        y = -(R[2, 0] - R[0, 2]) / w / 4
        z = -(R[0, 1] - R[1, 0]) / w / 4
        q = np.array([x, y, z, w])
        # q = np.array([x, y, z, w])
        n = np.dot(q, q)
        q /= n
        return q

    def to_tensor(self, depth_img, color_img):

        # depth normalize
        depth_img = self.in_paint(depth_img)
        scale = np.abs(depth_img).max()
        depth_img /= scale
        # depth_img = np.clip((depth_img - depth_img.mean()), -1, 1)
        depth_img = depth_img - depth_img.mean()

        # color normalize
        color_img = np.transpose(color_img, (2, 0, 1)).astype(np.float32)
        color_img /= 255
        color_img -= color_img.mean()

        # to tensor
        img_in = np.vstack(
            (np.expand_dims(depth_img, 0), color_img)
        )
        img_in = np.expand_dims(img_in, 0).astype(np.float32)
        img_in = torch.from_numpy(img_in).to(self.device)

        return img_in

    def grasp_img2real(self, color_img, depth_img, grasp, vis=False, color=(255, 0, 0),
                       note='', collision_check=False, depth_offset=0.0):
        row, column, angle, width = grasp

        # width_gripper = np.linalg.norm(point1[0:2] - point2[0:2], 2)
        width_gripper = width / 614.9 * depth_offset
        coordinate_cam = self.camera.get_coordinate(column+80, row)

        grasp_depth_out = coordinate_cam[2]
        if depth_offset != 0:
            # grasp_depth = croped_depth_img.min() + depth_offset * (croped_depth_img.max() - croped_depth_img.min())
            grasp_depth = depth_offset
        # if grasp_depth > 0.4227:  # z + 0.12286666
        #     grasp_depth_out = 0.4227
            if grasp_depth > self.init_pose[2] + 0.12286666:
                grasp_depth_out = self.init_pose[2] + 0.12286666

                if collision_check:
                    # print('grasp depth prediction:', grasp_depth)
                    print('collision with the plane!!!!')
        else:
            grasp_depth_out = coordinate_cam[2] - 0.1
        coordinate = np.append(coordinate_cam[0:2], grasp_depth_out)

        if vis:
            s = np.sin(angle)
            c = np.cos(angle)

            x1 = column + width / 2 * c
            x2 = column - width / 2 * c
            y1 = row - width / 2 * s
            y2 = row + width / 2 * s

            rect = np.array([
                [x1 - width / 4 * s, y1 - width / 4 * c],
                [x2 - width / 4 * s, y2 - width / 4 * c],
                [x2 + width / 4 * s, y2 + width / 4 * c],
                [x1 + width / 4 * s, y1 + width / 4 * c],
            ]).astype(np.int)

            # grasp center
            color_img = cv2.circle(color_img, (column, row), 3, (0, 0, 255), -1)
            color_img = cv2.arrowedLine(color_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            color_img = cv2.arrowedLine(color_img, (int(x2), int(y2)), (int(x1), int(y1)), color, thickness=2)
            color_img = cv2.line(color_img, tuple(rect[0]), tuple(rect[3]), color, thickness=2)
            color_img = cv2.line(color_img, tuple(rect[2]), tuple(rect[1]), color, thickness=2)
            # cv2.drawContours(color_img, [rect], 0, color, 4)
            if note != '':
                cv2.putText(color_img, note, (column, row), fontScale=1.2, fontFace=cv2.FONT_HERSHEY_SIMPLEX
                            , color=(255, 255, 255))
            from matplotlib import pyplot as plt
            # plt.imshow(depth_img)
            # plt.axis('off')
            # plt.show()
            # cv2.imshow('depth', depth_img)

            cv2.imshow('grasp', color_img)
            cv2.imwrite('prediction.png', color_img)

        return coordinate, width_gripper, angle

    def action(self, vis=False):

        depth_img, color_img = self.camera.get_img()

        depth_img = depth_img[:, 80:560]
        color_img = color_img[:, 80:560, :]

        depth_img = self.in_paint(depth_img)
        depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0)/1000

        img_in = self.to_tensor(depth_img, color_img)

        g, p, q = self.ggsnet(img_in, depth_img, imageSize=140)
        # for i in range(len(g)):
        #     g[i] = g[i].cpu().detach().numpy()
        # p = float(p.cpu().detach().numpy())

        coordinate, width_gripper, angle = self.grasp_img2real(
            color_img, depth_img, g, vis=vis, color=(0, 255, 255), collision_check=True,
            depth_offset=p
        )
        key = cv2.waitKey(0)
        position, orientation = self.get_target(coordinate, angle)

        self.move2Target(position, orientation)
        # if key & 0xFF == ord('q'):  # 按下q键则不进行此次抓取重新预测，按下其他键则进行抓取
        #     return 0
        # elif key & 0xFF == ord('s'):
        #     name = input()
        #     cv2.imwrite('/home/wangchuxuan/pic_grasp/'+name+'.png', color_img)
        # # coordinate[2] = self.init_pose[2] + 0.1228
        # if self.hardware:
        #     mat = cv2.Rodrigues(np.array([0, 0, -angle]).astype(np.float32))[0]
        #     Tobj2cam = np.vstack((
        #                 np.hstack(
        #                     (np.array(mat).dot(self.Rend2cam),
        #                      np.expand_dims(coordinate, 0).T)
        #                 ), np.array([[0, 0, 0, 1]])
        #                 ))
        #
        #     # Tobj2cam = self.vec2mat([0, 0, -angle], coordinate)
        #     self.gripper.gripper_position(int(width_gripper/0.095*100))
        #     pose = self.robot_connect.get_pose()
        #     Tend2base = self.vec2mat(pose[3:6], pose[0:3])
        #     Tobj2base = Tend2base.dot(self.Tcam2end).dot(Tobj2cam)
        #
        #     position = Tobj2base[0:3, 3]
        #     gesture = cv2.Rodrigues(Tobj2base[0:3, 0:3])[0].T.squeeze()
        #
        #     position[2] = min(0.3, max(position[2] - 0.02, 0.015))
        #     pose = np.hstack((position, gesture))
        #     pose_up_to_grasp_position = pose + [0, 0, 0.1, 0, 0, 0]
        #
        #     self.robot_connect.change_pose(pose_up_to_grasp_position)
        #     self.robot_connect.change_pose(pose)
        #     self.gripper.gripper_position(0)
        #     # cv2.waitKey(0)
        #     time.sleep(0.7)
        #     self.robot_connect.change_pose(pose_up_to_grasp_position)
        #     self.robot_connect.change_pose(self.place_pose)
        #     self.gripper.gripper_position(100)
        #     self.robot_connect.change_pose(self.init_pose)

    def grasp_sampling(self, feature_maps, num=20, pixel_wise_stride=1):
        q_out, ang_out, w_out = feature_maps
        grasps = []
        q_out = cv2.GaussianBlur(q_out, (5, 5), 0)
        shape = q_out.shape
        local_max = peak_local_max(q_out, min_distance=num, threshold_abs=0.4)
        for grasp_point_array in local_max:
            grasp_point = tuple(grasp_point_array)
            row = int(grasp_point[0]*pixel_wise_stride + pixel_wise_stride//2)
            column = int(grasp_point[1]*pixel_wise_stride + pixel_wise_stride//2)
            grasp_angle = ang_out[grasp_point]
            grasp_width = w_out[grasp_point]
            grasps.append((row, column, grasp_angle, grasp_width))

        # for row in range(shape[0] // num + 1):
        #     for column in range(shape[1] // num + 1):
        #         area = q_out[row * num:min(row * num + num, shape[0]),
        #                column * num:min(column * num + num, shape[1])]
        #         if len(area.shape) == 2 and not(0 in area.shape):
        #             index = np.unravel_index(np.argmax(area, axis=None), area.shape)
        #             grasp_point = (index[0] + row * num, index[1] + column * num)
        #             if q_out[grasp_point] > 0.5:
        #                 row = grasp_point[0]
        #                 column = grasp_point[1]
        #                 grasp_angle = ang_out[grasp_point]
        #                 grasp_width = w_out[grasp_point]
        #                 grasps.append((row, column, grasp_angle, grasp_width))
        return grasps

    def get_target(self, position_cam, angle):

        mat = cv2.Rodrigues(np.array([0, 0, -angle]).astype(np.float32))[0]
        Tobj2cam = np.vstack((
            np.hstack(
                (np.array(mat).dot(self.Rend2cam),
                 np.expand_dims(position_cam, 0).T)
            ), np.array([[0, 0, 0, 1]])
        ))

        pose = self.group.get_current_pose().pose
        position_tool = np.array([[pose.position.x], [pose.position.y], [pose.position.z]])
        orientation_tool = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        Ttool2base = np.vstack((
            np.hstack(
               (self.quat2RotMat(orientation_tool),
                 position_tool)
            ), np.array([[0, 0, 0, 1]])
        ))
        Tobj2base = Ttool2base.dot(self.Tend2tool).dot(self.Tcam2end).dot(Tobj2cam)
        Ttool2base = Tobj2base.dot(self.Ttool2end)

        position = Ttool2base[0:3, 3]
        orientation = self.rotMat2Quat(Ttool2base[0:3, 0:3])
        orientation = tftrans.euler_from_quaternion(orientation)

        return position, orientation

    def move2Target(self, position_target, orientation_target, pGain=1/5):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # 初始化geometry_msgs::Twist类型的消息
            p = self.group.get_current_pose().pose
            position_current = np.array([p.position.x, p.position.y, p.position.z])
            orientation_current = np.array(tftrans.euler_from_quaternion([p.orientation.x, p.orientation.y,
                                                                   p.orientation.z, p.orientation.w]))
            r, p, y = orientation_current
            vel_position = position_target - position_current
            vel_orientation_tool = orientation_target - orientation_current
            vel_orientation_base = np.array([
                [np.cos(p)*np.cos(y), -np.sin(y), 0],
                [np.cos(p)*np.sin(y), np.cos(y), 0],
                [-np.sin(p), 0, 1]
            ]).dot(vel_orientation_tool)

            vel_tool = np.append(vel_position, vel_orientation_base)

            t = time.time()
            j = self.group.get_jacobian_matrix(self.group.get_current_joint_values())
            # print((time.time()-t)*1000)
            vel_joints = np.dot(np.linalg.inv(np.array(j)), (vel_tool)*pGain)

            # 发布消息
            self.target_pose_publisher.publish(Float64MultiArray(data=vel_joints))
            rospy.loginfo(f"Publish velocity command {vel_joints}")
            # 按照循环频率延时
            rate.sleep()
            # t += 1

    def dynamic_action(self):
        position_target = np.array([0.312, 0.074, 0.654])
        orientation_target = np.array([-2.470, 0.532, -2.247])
        while self.group.get_current_pose().pose.position.z > 0.400:
            depth_img, color_img = self.camera.get_img()
            color_img = color_img[:, 80:560, :]
            depth_img = self.in_paint(depth_img[:, 80:560]) / 1000
            img_in = self.to_tensor(depth_img, color_img)
            g, p, q = self.ggsnet(img_in, depth_img, imageSize=140)

            coordinate, width_gripper, angle = self.grasp_img2real(
                color_img, depth_img, g, vis=False, color=(0, 255, 255), collision_check=True,
                depth_offset=p
            )
            key = cv2.waitKey(0)
            position_target, orientation_target = self.get_target(coordinate, angle)

            # self.move2Target(position, orientation)
            p = self.group.get_current_pose().pose
            position_current = np.array([p.position.x, p.position.y, p.position.z])
            orientation_current = np.array(tftrans.euler_from_quaternion([p.orientation.x, p.orientation.y,
                                                                          p.orientation.z, p.orientation.w]))
            r, p, y = orientation_current
            vel_position = position_target - position_current
            vel_orientation_tool = orientation_target - orientation_current
            vel_orientation_base = np.array([
                [np.cos(p) * np.cos(y), -np.sin(y), 0],
                [np.cos(p) * np.sin(y), np.cos(y), 0],
                [-np.sin(p),                    0, 1]
            ]).dot(vel_orientation_tool)

            vel_tool = np.append(vel_position, vel_orientation_base)

            t = time.time()
            j = self.group.get_jacobian_matrix(self.group.get_current_joint_values())
            print((time.time() - t) * 1000)
            vel_joints = np.dot(np.linalg.inv(np.array(j)), (vel_tool)/5)

            # 发布消息
            self.target_pose_publisher.publish(Float64MultiArray(data=vel_joints))

        self.move2Target(position_target, orientation_target, pGain=1/5)


if __name__ == '__main__':
    a = 1
    grasp = Grasp(hardware=True)
    # grasp.dynamic_action()
    pose = grasp.group.get_current_pose().pose
    position_tool = np.array([[pose.position.x], [pose.position.y], [pose.position.z]])
    orientation_tool = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    print(position_tool, orientation_tool)
    grasp.move2Target([0.4, 0, 0.6], tftrans.euler_from_quaternion(orientation_tool))
    # grasp.move2Target(np.array([0.4, 0, 0.75]), np.array([-2.573, 0.468, -1.258]))
    # grasp.action(vis=True)
    # while 1:
    #     # grasp.action(vis=True)
    #     # torch.cuda.empty_cache()
    #     # pass
    #     Grasp.move2Target()
    #     time.sleep(2)
    #     # grasp.judge()












