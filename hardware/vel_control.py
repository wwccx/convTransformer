import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Float64MultiArray
import numpy as np
import tf.transformations as tftrans
# from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import utils as u
import cv2


class VelController:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dynamic Grasping', anonymous=False)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = 'manipulator'
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.target_pose_publisher = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=100)
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_planning_frame()
        self.group_name = self.robot.get_group_names()

        self.Tcam2tool = np.array([
            [0.99770124, -0.06726973, -0.00818663, -0.02744465],
            [0.06610884, 0.99272747, -0.10060715, -0.10060833],
            [0.01489491, 0.09983467, 0.99489255, -0.15038112],
            [0., 0., 0., 1]
        ])

        self.Tend2tool = np.vstack((
                np.hstack(
                    (u.quat2RotMat([0.399, -0.155, 0.369, 0.825]), np.array([[-0.001], [0.038], [-0.248]]))
                    ),
                np.array([0, 0, 0, 1])
                ))
        self.Ttool2end = np.linalg.inv(self.Tend2tool)
        self.Rtool2cam = np.linalg.inv(self.Tcam2tool[0:3, 0:3])
        # print(self.Rtool2cam)

    def get_current_pose(self):

        current_pose = self.get_current_pose().pose
        current_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        current_orientation = np.array([pose.orientation.x, pose.orientation.z, pose.orientation.y, pose.orientation.w])
        return current_position, current_orientation

    def get_current_joint_values(self):
        return np.array(self.group.get_current_joint_values())

    def get_end_pose(self, pos, angle):
        """
        pos: obj pose in the camera frame, x, y, z
        angle: the rotation around -x, ggcnn result
        return: end pose in the base frame, (x, y, z) & (rx, ry, rz)
        """
        Tobj2cam = np.eye(4)
        Tobj2cam[0:3, 3] = pos
        mat = cv2.Rodrigues(np.array([0, 0, -angle]).astype(np.float32))[0]
        Tobj2cam[0:3, 0:3] = mat.dot(self.Rtool2cam)

        current_position, current_orientation = self.get_current_pose()

        Tend2base = np.eye(4)
        Tend2base[0:3, 3] = current_pose
        Tend2base[0:3, 0:3] = u.quat2RotMat(current_orientation)

        Tobj2base = Tend2base.dot(self.Ttool2end).dot(self.Tcam2tool).dot(self.Tobj2cam)
        Tend2base = Tobj2base.dot(self.Tend2tool)
        
        return Tend2base[0:3, 3], np.array(tftrans.euler_from_matrix(Tend2base[0:3, 0:3]))

    def set_joint_velocity(pos, angle, vel=0.1):
        target_position, target_orientation = self.get_end_pose(pos, angle)
        current_position, current_orientation = self.get_current_pose()
        vel_position = target_position - current_position
        vel_orientation_tool = target_orientation - current_orientation
        r, p, y = current_position

        vel_orientation_base = np.array([
            [np.cos(p) * np.cos(y), -np.sin(y), 0],
            [np.cos(p) * np.sin(y), np.cos(y), 0],
            [-np.sin(p), 0, 1]
        ]).dot(vel_orientation_tool)

        vel_tool = np.append(vel_position, vel_orientation_base)
        jacobian = self.group.get_jacobian_matrix(self.get_current_joint_values())

        vel_joints = np.dot(np.linalg.inv(jacobian), vel * vel_tool / np.linalg.norm(vel_tool, 2))

        self.target_pose_publisher.publish(Float64MultiArray(data=vel_joints))


if __name__ == '__main__':
    v = VelController()
