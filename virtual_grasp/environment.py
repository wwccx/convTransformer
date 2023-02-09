# import sys
#
# sys.path.append('/home/server/grasp1')

import pybullet as p
import time
import pybullet_data
import math
from collections import namedtuple
from attrdict import AttrDict
import os
import numpy as np
import cv2
import torch
from glob import glob
import logging
import trimesh
logging.basicConfig(level=logging.INFO)


class VirtualEnvironment(object):
    def __init__(self, num_robots=8):
        self._timeStep = 1. / 240.
        self._urdfRoot = pybullet_data.getDataPath()

        self._isEnableSelfCollision = True
        self._envStepCounter = 0
        self._renders = True
        self.terminated = 0
        self.p = p

        self._meshPath = './virtual_grasp/meshes/'
        self.meshPath = self._meshPath

        self.cameraMat = np.array([[180.34067991, 0., 320],
                                   [0., 180.32791943, 320],
                                   [0., 0., 1.]])
        self.cameraDistCoeffs = np.array([[0.0037851, -0.00217189, 0.00016015, -0.00012455, 0.00046416]])

        if self._renders:
            cid = p.connect(p.GUI)

        self._reset()
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        self.jointIds = []
        self.jointForce = []
        self.objects = []
        self.numRobots = num_robots
        self.d = 1.8
        self.load_objects()
       # load tables and robots

    def _reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        p.stepSimulation()
        p.setRealTimeSimulation(1)

    def reset_all_robots(self):
        self.p.resetSimulation(0)
        self._reset()
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        self.jointIds = []
        self.jointForce = []
        self.objects = []
        self.load_objects()
        for d, uid in enumerate(self.robotId):
            self.move_joints(self.get_joints_angle([0, d * self.d, 0.7],
                                                   self.p.getQuaternionFromEuler([0, np.pi / 2, 0]),
                                                   0, uid
                                                   ),
                             uid
                             )

    def load_objects(self):
        # self.floor = p.loadURDF("plane.urdf")
        self.tables = []
        for i in range(self.numRobots):
            t = p.loadURDF((os.path.join(self._urdfRoot, "table/table.urdf")),
                           basePosition=[0.0, i * self.d, -0.63 * 1.7],
                           baseOrientation=(0.000000, 0.000000, 0.0, 1.0),
                           globalScaling=1.7, useFixedBase=True)
            self.p.changeDynamics(t, -1, mass=0, lateralFriction=6,
                                  restitution=0.98, rollingFriction=0.07, spinningFriction=0.15,
                                  contactStiffness=1e6, contactDamping=0.5)
            self.tables.append(t)

        robotStartPos = [0.0, 0.0, 1]
        robotStartOrn = p.getQuaternionFromEuler([0, 0.0, 0])

        uid = p.loadURDF(os.path.join(self._meshPath, "../urdf/real_arm.urdf"), robotStartPos, robotStartOrn,
                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.robotId = [uid]
        joints, controlRobotiqC2, controlJoints, mimicParentName = self.setup_sisbot(p, uid)
        self.endEffectorIndex = 7  # ee_link
        self.numJoints = p.getNumJoints(uid)
        for i, name in enumerate(controlJoints):
            joint = joints[name]
            self.jointIds.append(joint.id)
            self.jointForce.append(joint.maxForce)

        self.resetJointPoses(uid)
        time.sleep(1)
        for i in range(1, self.numRobots):
            self.robotId.append(
                p.loadURDF(os.path.join(self._meshPath, "../urdf/real_arm.urdf"), [0, i * self.d, 1], robotStartOrn,
                           flags=p.URDF_USE_INERTIA_FROM_FILE)
            )
            self.setup_sisbot(self.p, self.robotId[i])
            self.resetJointPoses(self.robotId[i])
        time.sleep(1)
        for i in range(self.numRobots):
            p.resetBasePositionAndOrientation(self.robotId[i], [-0.5, i * self.d, -0.07], [0, 0, 0, 1])

    @staticmethod
    def generate_urdf(obj_file, path=''):
        obj_file_name = os.path.basename(obj_file)
        vhacdPath = os.path.relpath(os.path.join(obj_file, "..", 'vhacd'))

        if not os.path.exists(os.path.join(vhacdPath, obj_file_name.replace('.obj', '_vhacd.obj'))):
            p.vhacd(obj_file, os.path.join(vhacdPath, obj_file_name.replace('.obj', '_vhacd.obj')),
                    os.path.join(path, 'log.txt'), alpha=0.04, resolution=5000000)
        obj = trimesh.load(os.path.join(vhacdPath, obj_file_name.replace('.obj', '_vhacd.obj')))
        iMat = obj.moment_inertia
        cMass = obj.center_mass

        scale = 0.08 / np.abs(obj.bounding_box.bounds).mean()
        z_pos = np.max(obj.bounding_box.bounds)
        if not os.path.exists(obj_file.replace('.obj', '.urdf')):
            text = f"""
<?xml version="1.0" ?>
<robot name="{os.path.abspath(obj_file.replace('.obj', '.urdf'))}">
  <link name="legobrick">
    <contact>
      <lateral_friction value="10.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="10.0"/>
      <contact_erp value="0.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="{cMass[0]} {cMass[1]} {cMass[2]}"/>
       <mass value="{4e3 * obj.mass}"/>
       <inertia ixx="{4e3 * iMat[0, 0]}" ixy="{4e3 * iMat[0, 1]}" ixz="{4e3 * iMat[0, 2]}" iyy="{4e3 * iMat[1, 1]}" iyz="{4e3 * iMat[1, 2]}" izz="{4e3 * iMat[2, 2]}"/>
    </inertial>
    <visual>
      <origin rpy="1.570796 0 0" xyz="0 0 0"/>
      <geometry>
	<mesh filename="{
                os.path.abspath(os.path.join(vhacdPath, obj_file_name.replace('.obj', '_vhacd.obj')))  
            }" scale="1 1 1"/>
      </geometry>
       <material name="blue">
	<color rgba="0.4 0.4 1.0 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="1.570796 0 0" xyz="0 0 0"/>
      <geometry>
	<mesh filename="{os.path.abspath(os.path.join(vhacdPath, obj_file_name.replace('.obj', '_vhacd.obj')))}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
	"""
            with open(obj_file.replace('.obj', '.urdf'), 'w') as f:
                f.write(text)
            f.close()
            return obj_file.replace('.obj', '.urdf'), scale, z_pos
        else:
            return obj_file.replace('.obj', '.urdf'), scale, z_pos

    @staticmethod
    def setup_sisbot(p, uid):

        controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                         "elbow_joint", "wrist_1_joint",
                         "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']

        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(uid)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                "controllable"])
        joints = AttrDict()
        for i in range(numJoints):
            info = p.getJointInfo(uid, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in controlJoints else False
            info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":  # set revolute joint to static
                p.setJointMotorControl2(uid, info.id, p.POSITION_CONTROL, targetVelocity=0, force=0)
            joints[info.name] = info
        controlRobotiqC2 = False
        mimicParentName = False
        return joints, controlRobotiqC2, controlJoints, mimicParentName

    def get_joints_angle(self, pos, ori, jaw, robotId):
        x = pos[0]
        y = pos[1]
        z = pos[2]

        jointPose = list(p.calculateInverseKinematics(robotId, self.endEffectorIndex, [x, y, z], ori))
        jointPose[7] = -jaw / 25
        jointPose[6] = jaw / 25

        return jointPose

    def move_joints(self, jointPose, robotId, force_scale=1.0, ):
        jointPose = list(jointPose)
        jointPose = jointPose[:len(self.jointIds)]
        p.setJointMotorControlArray(robotId, self.jointIds, p.POSITION_CONTROL, targetPositions=jointPose,
                                    targetVelocities=[0] * len(self.jointIds),
                                    forces=[i * 1.0 for i in self.jointForce])

    def resetJointPoses(self, robotId):
        for i in range(0, 60000):
            self.move_joints(
                [0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994,
                 0.19581299859677043, 0.012000000476837159, 1], robotId)

    def update_camera(self, h=640, w=640, n=0.0001, f=1.5):
        # projectionMatrix = p.computeProjectionMatrixFOV(fov=120, aspect=1, nearVal=n, farVal=f)
        projectionMatrix = (2, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2 / (f - n), 0, 0, 0, -(f + n) / (f - n), 1)
        image_renderer = p.ER_BULLET_HARDWARE_OPENGL
        rgbImage = []
        depthImage = []
        init_camera_vector = (1, 0, 0)  # x-axis
        init_up_vector = (0, 0, 1)  # z-axis
        for i in range(self.numRobots):
            obs = p.getLinkState(self.robotId[i], self.endEffectorIndex)
            posEnd = obs[4]
            oriEnd = obs[5]
            rot_matrix = p.getMatrixFromQuaternion(oriEnd)
            rot_matrix = np.array(rot_matrix).reshape(3, 3)
            camera_vector = rot_matrix.dot(init_camera_vector)
            up_vector = rot_matrix.dot(init_up_vector)
            view_matrix_gripper = p.computeViewMatrix(posEnd + 0.2 * camera_vector, posEnd + 0.3 * camera_vector,
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
            rgbImage.append(rgb)
            depthImage.append(depth)

        return rgbImage, depthImage

    def calibrate_camera(self):
        ori = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        self.update_camera()
        jointPose = self.get_joints_angle([0, 0.0, 0.5], ori, 0, self.robotId[0])
        self.move_joints(jointPose, self.robotId[0])

        plane = [p.loadURDF(os.path.join(self._urdfRoot, 'table_square', 'table_square.urdf'),
                            basePosition=[0.0, 0.0, -0.654 * 0.16 / 0.6 + 0.01],
                            baseOrientation=(0.000000, 0.000000, 0.0, 1.0),
                            globalScaling=1 * 0.16 / 0.6)]  # length of the original square table: 0.6
        p3d = []
        p2d = []
        shape = (400, 400)
        coordinate_object = []
        corners_size = (9, 9)
        for index in range(9 * 9):
            coordinate_object.append(
                np.array([index % 9 * 0.06, index // 9 * 0.06, 0], dtype=np.float32)
            )

        userParameters = []
        abs_distance = 1.0
        userParameters.append(self.p.addUserDebugParameter("X", -abs_distance, abs_distance, 0))
        userParameters.append(self.p.addUserDebugParameter("Y", -abs_distance, abs_distance, 0))
        userParameters.append(self.p.addUserDebugParameter("Z", -abs_distance, abs_distance, 0.5))
        userParameters.append(self.p.addUserDebugParameter("roll", -math.pi, math.pi, 0.0))
        userParameters.append(self.p.addUserDebugParameter("pitch", -math.pi, math.pi, 1.5774))
        userParameters.append(self.p.addUserDebugParameter("yaw", -math.pi, math.pi, 0))
        userParameters.append(self.p.addUserDebugParameter('gripper', -1, 1, 0))
        while 1:

            obs = p.getLinkState(self.robotId[0], self.endEffectorIndex)
            posEnd = obs[4]
            # oriEnd = obs[5]
            action = []

            for para in userParameters:
                action.append(self.p.readUserDebugParameter(para))
            # posEnd = action[0:3]
            g = action[-1]
            oriEnd = p.getQuaternionFromEuler([action[3], action[4], action[5]])
            jointPose = vGrasp.get_joints_angle([action[0], action[1], action[2]], oriEnd, g, self.robotId[0])
            vGrasp.move_joints(jointPose, self.robotId[0])
            rgbImage, depth = self.update_camera()
            pass
            rgbImage = cv2.cvtColor(
                np.array(rgbImage[0], dtype=np.uint8), cv2.COLOR_RGBA2BGR
            )
            imageShape = rgbImage.shape
            cv2.imshow('rgb', rgbImage)
            rgb = np.zeros_like(rgbImage)
            rgb = cv2.undistort(rgbImage, self.cameraMat, self.cameraDistCoeffs)
            cv2.imshow('rgb_undistort', rgb)
            key = cv2.waitKey(10)
            if key & 0xFF == ord('c'):
                ret, corners = cv2.findChessboardCorners(cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY), corners_size,
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
                if ret:
                    cv2.drawChessboardCorners(rgbImage, corners_size, corners, ret)
                    corners = np.array([corner for [corner] in corners])
                    cv2.imshow('rgb_chess', rgbImage)
                    key = cv2.waitKey(0)
                    if key & 0xFF == ord('e'):
                        p3d.append(np.array(coordinate_object))
                        p2d.append(corners)
                        if len(p2d) > 4:
                            camera_matrix = cv2.initCameraMatrix2D(p3d, p2d, (imageShape[0], imageShape[1]))
                            results = cv2.calibrateCamera(p3d, p2d, (imageShape[0], imageShape[1]),
                                                          camera_matrix, None)
                            print(results)
                else:
                    print('change the angle !!!')

    @staticmethod
    def rotMat2Quat(R):
        w = np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1) / 2
        x = (R[1, 2] - R[2, 1]) / w / 4
        y = (R[2, 0] - R[0, 2]) / w / 4
        z = (R[0, 1] - R[1, 0]) / w / 4
        q = np.array([x, y, z, w])
        n = np.dot(q, q)
        q /= n
        return q

    def pixel2base(self, pixelIndex, angle, depthImage, depth=None, dx=0, dy=0):
        depthImage = depthImage.squeeze().detach().cpu().numpy()
        xp, yp = pixelIndex
        xp = max(0, min(xp, 639 - dx * 2))
        yp = max(0, min(yp, 639 - dy * 2))

        z = depthImage[yp, xp]  # meter
        # print('z:', z)
        # Pobj2cam = np.dot(np.linalg.inv(self.cameraMat), [[xp+dx], [yp+dy], [1]]) * z  # meter
        Pobj2cam = np.array([[(xp + dx - 320) / 640], [(yp + dy - 320) / 640], [z]])
        if depth:
            Pobj2cam[2][0] = depth
        Pobj2cam[2][0] -= 0.15
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

    def grasp_img2real(self, color_img, depth_img, grasp, vis=True, color=(255, 0, 0),
                       note='', collision_check=False, depth_offset=0.0, dx=80, dy=80, depth=0.0):
        row, column, angle, width = grasp
        f_cam = 180.33
        widthGripper = width / 640 / 0.16
        widthGripper = min(1, max(0, widthGripper))
        widthGripper = 1 - widthGripper * 2

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

        quat, pos = self.pixel2base((column, row), angle, depth_img, depth=depth, dx=dx, dy=dy)

        coordinate = pos
        if vis:
            color_img = np.array(color_img, dtype=np.uint8)
            color_img = cv2.circle(color_img, (column, row), 5, color, -1)
            color_img = cv2.arrowedLine(color_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            color_img = cv2.arrowedLine(color_img, (int(x2), int(y2)), (int(x1), int(y1)), color, thickness=2)
            color_img = cv2.line(color_img, tuple(rect[0]), tuple(rect[3]), color, thickness=2)
            color_img = cv2.line(color_img, tuple(rect[2]), tuple(rect[1]), color, thickness=2)
            cv2.imshow('grasp', color_img)
            cv2.waitKey(10)

        return coordinate, quat, color_img, widthGripper


if __name__ == '__main__':
    vGrasp = VirtualEnvironment(num_robots=1)
    # import cv2
    # cv2.waitKey(0)
    vGrasp.calibrate_camera()


