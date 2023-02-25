import sys
import rospy
import moveit_commander
from std_msgs.msg import Float64MultiArray
import numpy as np
import tf.transformations as tftrans
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import utils as u
import cv2


class VelController:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dynamic_grasping', anonymous=False)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = 'manipulator'
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.target_pose_publisher = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=100)
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_planning_frame()
        self.group_name = self.robot.get_group_names()
        self.rate = rospy.Rate(10)

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

    def get_current_end_pose(self):
        
        pose = self.group.get_current_pose().pose
        current_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        current_orientation = np.array([pose.orientation.x, pose.orientation.z, pose.orientation.y, pose.orientation.w])
        
        return current_position, current_orientation

    def get_current_tool_pose(self):
        """
        return: tool pose
        """
        current_end_pos, current_end_ori = self.get_current_end_pose()
        Tend2base = u.pos_quat2mat(current_end_pos, current_end_ori)
        Ttool2base = Tend2base @ self.Ttool2end
        current_position = Ttool2base[0:3, 3]
        current_orientation = u.rotMat2Quat(Ttool2base[0:3, 0:3])
        return current_position, current_orientation

    def get_current_joint_values(self):
        return self.group.get_current_joint_values()

    def get_aim_end_pose(self, pos, ori):
        """
        pos: obj pose in the world frame, x, y, z, aiming to let it equal to tool frame
        ori: obj ori in the world frame
        return: end pose in the base frame, (x, y, z) & (rx, ry, rz)
        """
        # Tobj2cam = np.eye(4)
        # Tobj2cam[0:3, 3] = pos
        # mat = cv2.Rodrigues(np.array([0, 0, -angle]).astype(np.float32))[0]
        # Tobj2cam[0:3, 0:3] = mat.dot(self.Rtool2cam)

        # current_position, current_orientation = self.get_current_pose()

        # Tend2base = np.eye(4)
        # Tend2base[0:3, 3] = pos
        # Tend2base[0:3, 0:3] = u.quat2RotMat(current_orientation)
        Tobj2base = np.eye(4)
        Tobj2base[0:3, 3] = pos
        Tobj2base[0:3, 0:3] = u.quat2RotMat(ori)

        # Tobj2base = Tend2base.dot(self.Ttool2end).dot(self.Tcam2tool).dot(Tobj2cam)
        Tend2base = Tobj2base.dot(self.Tend2tool)
        
        return Tend2base[0:3, 3], np.array(tftrans.euler_from_matrix(Tend2base[0:3, 0:3]))

    def get_joint_velocity(self, pos, ori):
        # pipeline: get the velocity of the tool frame, transform it into end frame
        target_position, target_orientation = self.get_aim_end_pose(pos, ori)
        # target_position, target_orientation = pos, u.quat2rpy(ori)
        current_position, current_orientation = self.get_current_end_pose()
        current_orientation = tftrans.euler_from_quaternion(current_orientation)
        vel_position = target_position - current_position
        vel_orientation_tool = target_orientation - current_orientation
        print('vel_ori:', vel_orientation_tool, np.linalg.norm(vel_orientation_tool, 2))
        print('vel_pos:', vel_position, np.linalg.norm(vel_position, 2))
        r, p, y = current_orientation
        # target: translation and rotation of tool || contorl: translation and rotation of end
        #TODO: need transformations
        vel_orientation_base = np.array([
            [np.cos(p) * np.cos(y), -np.sin(y), 0],
            [np.cos(p) * np.sin(y), np.cos(y), 0],
            [-np.sin(p), 0, 1]
        ]).dot(vel_orientation_tool)

        vel_tool = np.append(vel_position, vel_orientation_tool)

        return vel_tool

    def set_joint_velocity(self, vel_direction, vel=0.1):
        jacobian = self.group.get_jacobian_matrix(self.get_current_joint_values())

        vel_joints = np.dot(np.linalg.inv(jacobian), vel * vel_direction / np.linalg.norm(vel_direction, 2))

        self.target_pose_publisher.publish(Float64MultiArray(data=vel_joints))

    def end_joint_rotation(self):
        self.target_pose_publisher.publish(Float64MultiArray(data=np.zeros(6)))

if __name__ == '__main__':
    import time
    v = VelController()
    # vel = np.array([0, 0, 0.02, 0, 0, 0])
    # for i in range(1000):
    #     jacobian = v.group.get_jacobian_matrix(v.get_current_joint_values())
    #     
    #     vel_joints = np.dot(np.linalg.inv(jacobian), vel)
    #     v.target_pose_publisher.publish(Float64MultiArray(data=vel_joints))
    #     v.rate.sleep()
    #     
    # v.end_joint_rotation() 
    # print('tool:', v.get_current_tool_pose())
    # end = v.get_current_end_pose()
    # print('end:', end[0], u.quat2rpy(end[1]))
    # aim = v.get_aim_end_pose(*list(v.get_current_tool_pose()))
    # print('aim:',aim[0], aim[1]) 
    print(v.get_current_tool_pose())
    
    # '''
    pos, ori = v.get_current_tool_pose()
    print(pos, ori)
    count = 0
    pos_new = np.array([0.03188966, 0.55277825, 0.15720356])
    vel = v.get_joint_velocity(pos_new, ori)
    while np.linalg.norm(pos_new - pos, 2) > 1e-3 and count < 100:

        count += 1
        pos, _ = v.get_current_tool_pose()
        vel = v.get_joint_velocity(pos_new, ori)
        v.set_joint_velocity(vel)
        # print(v.get_current_tool_pose(), ori)
        v.rate.sleep()
    v.end_joint_rotation()
    # '''
    
