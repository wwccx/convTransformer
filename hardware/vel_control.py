import sys
import rospy
import moveit_commander
from std_msgs.msg import Float64MultiArray
import numpy as np
import tf.transformations as tftrans
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import utils as u
import cv2
from multiprocessing import Process, Queue
from threading import Thread
from queue import Empty


class PID():
    def __init__(self, p, i, d, time_step=1e-2, max_v=1) -> None:
        self.p = p
        self.i = i
        self.d = d
        self.int_error = 0
        self.last_error = 0
        self.dt = time_step
        self.upper = max_v
    
    def __call__(self, e):
        p = self.p * e
        self.int_error += e * self.dt
        i = self.i * self.int_error
        d = self.d * (e - self.last_error) / self.dt
        self.last_error = e
        return min(p + i + d, self.upper)

    def clear(self):
        self.int_error = 0
        self.last_error = 0

class VelController(Thread):
    def __init__(self, queue):
        super().__init__()
        rospy.init_node('dynamic_grasping', anonymous=False)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = 'manipulator'
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.target_pose_publisher = rospy.Publisher('/joint_group_vel_controller/command', Float64MultiArray, queue_size=100)
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_planning_frame()
        self.group_name = self.robot.get_group_names()
        time_step = 1e-2
        self.rate = rospy.Rate(1000 * time_step)
        self.queue = queue
        pos = np.array([0.0, 0.45, 0.45])
        ori = np.array([[ 0.99807688,  0.06030169, -0.01436103],
                        [ 0.05825503, -0.9916341,  -0.11518751],
                        [-0.02118689,  0.11412939, -0.99323995]])
        self.aim_pose_init = (pos, ori)
        self.aim_pose = (pos, ori)
        self.reach_target = False
        self._suspend = False

        self.Tcam2tool = np.array([
            [0.99770124, -0.06726973, -0.00818663, -0.02744465],
            [0.06610884, 0.99272747, -0.10060715, -0.10060833],
            [0.01489491, 0.09983467, 0.99489255, -0.15038112],
            [0., 0., 0., 1]
        ])

        #self.Tend2tool = np.vstack((
        #        np.hstack(
        #            (u.quat2RotMat([0.399, -0.155, 0.369, 0.825]), np.array([[-0.001], [0.038], [-0.248]]))
        #            ),
        #        np.array([0, 0, 0, 1])
        #        ))
        #self.Ttool2end = np.linalg.inv(self.Tend2tool)
        self.Ttool2end = u.vec2mat(np.array([-0.8489, 0.3304, -0.7849]), np.array([0.11918, 0.11893, 0.18631]))
        self.Tend2tool = np.linalg.inv(self.Ttool2end)
        self.Rtool2cam = np.linalg.inv(self.Tcam2tool[0:3, 0:3])
        # print(self.Rtool2cam)
        self.Tmove2ur = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            ])
        self.pos_pid = PID(5, 1, 0.1, time_step=time_step)
        self.ori_pid = PID(5, 1, 0.1, time_step=time_step, max_v=2)

    def get_current_end_pose(self):
        
        pose = self.group.get_current_pose().pose
        current_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        current_orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        # Tend2base = u.pos_quat2mat(current_position, current_orientation)
        # Tend2ur = self.Tmove2ur @ Tend2base
        # current_position = Tend2ur[0:3, 3]
        # current_orientation = u.rotMat2Quat(Tend2ur[0:3, 0:3])
        
        return current_position, current_orientation

    def get_current_tool_pose(self):
        """
        return: tool pose
        """
        current_end_pos, current_end_ori = self.get_current_end_pose()
        Tend2base = u.pos_quat2mat(current_end_pos, current_end_ori)
        Ttool2base = Tend2base @ self.Ttool2end
        # Ttool2base = self.Tmove2ur @ Ttool2base
        current_position = Ttool2base[0:3, 3]
        # current_orientation = u.rotMat2Quat(Ttool2base[0:3, 0:3])
        current_orientation = Ttool2base[0:3, 0:3]
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
        """
        :param pos np.array 3
        :param ori rotmat np.array 3*3
        
        """
        current_tool_pos, current_tool_ori = self.get_current_tool_pose()
        aim_tool_ori = ori
        aim_tool_pos = pos
        vel_tool_pos = aim_tool_pos - current_tool_pos
        vel_tool_ori = aim_tool_ori @ np.linalg.inv(current_tool_ori)
        vel_end_rpy = np.array(tftrans.euler_from_matrix(vel_tool_ori))
        e_pos = np.linalg.norm(vel_tool_pos, 2) ** 0.5
        e_ori = np.linalg.norm(vel_end_rpy, 2) ** 0.5
        v_pos = self.pos_pid(e_pos)
        v_ori = self.ori_pid(e_ori)
        # print('lambda:', v_pos, v_ori)
        # print('rotmat:\n', vel_tool_ori)
        # vel_tool_ori_rot_vec = cv2.Rodrigues(vel_tool_ori.astype(np.float32))[0].squeeze()
        # print('rotvec:\n', vel_tool_ori_rot_vec)
        # antisymmetric_rot_vec = np.array([
        #     [0, -vel_tool_ori_rot_vec[2], vel_tool_ori_rot_vec[1]],
        #     [vel_tool_ori_rot_vec[2], 0, -vel_tool_ori_rot_vec[0]],
        #     [-vel_tool_ori_rot_vec[1], vel_tool_ori_rot_vec[0], 0]
        #     ])  # such a method gets the Angular velocity of each axis, but not the rpy angular velocity
        # # print('antisymmetric_rot_vec:', antisymmetric_rot_vec)
        # current_end_pos, current_end_ori = self.get_current_end_pose()
        # current_end_ori = u.quat2RotMat(current_end_ori)
        # dot_vel_end_ori = antisymmetric_rot_vec # @ current_end_ori
        # # print('shape:', dot_vel_end_ori.shape)
        # vel_end_rpy = np.linalg.norm(dot_vel_end_ori, 2, axis=0) 
        # print(vel_tool_pos, vel_end_rpy)
        vel_end = np.append(vel_tool_pos * v_pos / e_pos, vel_end_rpy * v_ori / e_ori)
        return vel_end

        # # pipeline: get the velocity of the tool frame, transform it into end frame
        # target_position, target_orientation = self.get_aim_end_pose(pos, ori)
        # # target_position, target_orientation = pos, u.quat2rpy(ori)
        # current_position, current_orientation = self.get_current_end_pose()
        # current_orientation = tftrans.euler_from_quaternion(current_orientation)
        # vel_position = target_position - current_position
        # vel_orientation_tool = target_orientation - current_orientation
        # print('vel_ori:', vel_orientation_tool, np.linalg.norm(vel_orientation_tool, 2))
        # print('vel_pos:', vel_position, np.linalg.norm(vel_position, 2))
        # r, p, y = current_orientation
        # # target: translation and rotation of tool || contorl: translation and rotation of end
        # #TODO: need transformations
        # vel_orientation_base = np.array([
        #     [np.cos(p) * np.cos(y), -np.sin(y), 0],
        #     [np.cos(p) * np.sin(y), np.cos(y), 0],
        #     [-np.sin(p), 0, 1]
        # ]).dot(vel_orientation_tool)

        # vel_tool = np.append(vel_position, vel_orientation_tool)

        return vel_tool

    def set_joint_velocity(self, velocity):
        jacobian = self.group.get_jacobian_matrix(self.get_current_joint_values())

        vel_joints = np.dot(np.linalg.inv(jacobian),  velocity)

        self.target_pose_publisher.publish(Float64MultiArray(data=vel_joints))

    def _end_joint_rotation(self):
        self.target_pose_publisher.publish(Float64MultiArray(data=np.zeros(6)))

    def end_listen(self):
        self.queue.put(None)

    def suspend(self):
        self._suspend = True
        self._end_joint_rotation()

    def continue_listen(self):
        self._suspend = False

    def run(self):
        while True:
            if self._suspend:
                self.rate.sleep()
                continue
            try:
                # print('try to get aim pose')
                aim_pose = self.queue.get(False)
                if aim_pose is None:
                    self._end_joint_rotation()
                    break
                self.aim_pose = aim_pose
                self.reach_target = False
            except KeyboardInterrupt:
                break
            except Empty:
                pass
            
            finally:
                # print('get current pose')
                current_tool_pos, current_tool_ori = self.get_current_tool_pose()
                epos = np.linalg.norm(self.aim_pose[0] - current_tool_pos, 2) ** 0.5
                eori = np.linalg.norm(current_tool_ori - self.aim_pose[1], 2) ** 0.5
                if epos < 1e-2 and eori < 2e-2:
                    self._end_joint_rotation()
                    self.reach_target = True
                else:
                    # print(epos, eori)
                    # print(self.aim_pose[0], current_tool_pos)
                    vel = self.get_joint_velocity(*self.aim_pose)
                    # print(vel)
                    self.set_joint_velocity(vel)
                self.rate.sleep()


if __name__ == '__main__':
    
    from multiprocessing import Queue
    qq= Queue()
    v = VelController(qq)
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
    # print('init tool pose:', v.get_current_tool_pose())
    

    pos, ori = v.get_current_tool_pose()
    print('init tool pose:', pos, ori)

    # end_pos, end_ori = v.get_current_end_pose()
    # print('end:', end_pos, cv2.Rodrigues(u.quat2RotMat(end_ori))[0].squeeze())
    # print('end:', end_pos, '\n', u.quat2RotMat(end_ori))

    # count = 0
    # pos_new = np.array([0.00188966, 0.45277825, 0.35720356])
    # ori = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    # print(ori)
    # # ori = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # vel = v.get_joint_velocity(pos_new, ori)
    # 
    # while np.linalg.norm(pos_new - pos, 2) > 1e-2 and count < 1000:
    # # while count < 1000:
    #     count += 1
    #     pos, _ = v.get_current_tool_pose()
    #     vel = v.get_joint_velocity(pos_new, ori)
    #     # print(vel)
    #     v.set_joint_velocity(vel)
    #     # print(v.get_current_tool_pose(), ori)
    #     v.rate.sleep()
    # v.end_joint_rotation()
    # print('end tool pose:', v.get_current_tool_pose())

