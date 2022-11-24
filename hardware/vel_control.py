import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Float64MultiArray

import numpy as np
import tf.transformations as tftrans

# from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


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

    def get_current_pose(self):
        return self.group.get_current_pose().Pose

    def get_current_joint_values(self):
        return self.group.get_current_joint_values()

    def move2pose(self, pose):
        pass

