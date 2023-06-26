import socket
import struct
import numpy as np
import time

class UrSocket(object):
    def __init__(self, HOST="192.168.31.246", PORT=30003):
        self.HOST = HOST
        self.PORT = PORT
        self.doSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()
        self.data = 0
        self.t = time.time()

        self.dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
                       'I target': '6d',
                       'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
                       'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d',
                       'Tool vector target': '6d',
                       'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d',
                       'Controller Timer': 'd',
                       'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
                       'Tool Accelerometer values': '3d',
                       'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
                       'softwareOnly2': 'd',
                       'V main': 'd',
                       'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd',
                       'Program state': 'd',
                       'Elbow position': '3d', 'Elbow velocity': '3d'}
        self.status = {}
        self.pose = self.get_pose()

    def connect(self):
       """
       connect your PC to the UR
       :return:
       """

       self.doSocket.connect((self.HOST, self.PORT))

    def receive(self):
        """
         receive current data of the position and pose of the TCP
        :return:  the data is saved in self.data, and the data is in the format of bytes
        """
        # t = time.time()
        self.doSocket.recv(1200*1250)
        # self.t = time.time()
        self.data = bytes(self.doSocket.recv(1200))

    def get_pose(self):
        """
        decode the data message, and get the current x, y, z, rx, ry, rz, the last three is the Rotating vector
        :return: x, y, z, rx, ry, rz
        """
        self.receive()
        if len(self.data[444:468]) != 24:
            return self.pose
        # x, y, z = struct.unpack('!ddd', self.data[444:468])
        # rx, ry, rz = struct.unpack('!ddd', self.data[468:492])
        # self.pose = [x, y, z, rx, ry, rz]
        self.pose = struct.unpack('!6d', self.data[444:492])
        return self.pose
        # # names = []
        # ii = range(len(self.dic))
        # data = self.data
        # for key, i in zip(self.dic, ii):
        #     fmtsize = struct.calcsize(self.dic[key])
        #     data1, data = data[0:fmtsize], data[fmtsize:]
        #     fmt = "!" + self.dic[key]
        #     # names.append(struct.unpack(fmt, data1))
        #     self.status[key] = struct.unpack(fmt, data1)
        # return self.status['Tool vector actual']
        # self.receive()
        # data = self.data
        # dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
        #        'I target': '6d',
        #        'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
        #        'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
        #        'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
        #        'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
        #        'Tool Accelerometer values': '3d',
        #        'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
        #        'softwareOnly2': 'd',
        #        'V main': 'd',
        #        'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
        #        'Elbow position': '3d', 'Elbow velocity': '3d'}
        #
        # names = []
        # ii = range(len(dic))
        # for key, i in zip(dic, ii):
        #     fmtsize = struct.calcsize(dic[key])
        #     data1, data = data[0:fmtsize], data[fmtsize:]
        #     fmt = "!" + dic[key]
        #     names.append(struct.unpack(fmt, data1))
        #     dic[key] = dic[key], struct.unpack(fmt, data1)
        #
        # return dic['Tool vector actual'][1]


    def change_pose(self, newPose, a=0.6, v=1.2):
        """
        :param newPose: [x, y, z, rx, ry, rz]
        :return:
        """
        t = time.time()

        self.pose = tuple(newPose)
        self.send(a=a, v=v)
        self.doSocket.close()
        self.doSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect()

        while np.linalg.norm(np.array(self.get_pose()) - np.array(newPose), 2) > 0.0013:
            # print(np.linalg.norm(np.array(self.get_pose()) - np.array(newPose), 2))
            if time.time() - t > 5:
                raise RuntimeError('collision may happen')

        return

    def send(self, a=0.6, v=1.2, pattern='l'):
        """
        :param a: Acceleration
        :param v: Velocity
        :param pattern: the moving pattern, 'p' or 'l'
        :return: send the pose in class to the UR
        """
        if not(pattern == 'p' or pattern == 'l'):
            print('the pattern is not supported')
            return
        pose = self.pose
        # pose = str(pose).replace('(', '').replace(')', '')
        pose = str(pose)[1:-1]
        message = 'move' + pattern + '(p[' + pose +'],a=' + str(a) + ',v=' +str(v) +')'+'\n'
        # message = 'move' + pattern + '(p[' + pose + '])' + '\n'
        # print(message)
        self.doSocket.sendall(message.encode())

    # def get_rotation_mat(self):
    #     """
    #     from the current pose, get the rotation vec and trans it to the rotation mat (3*3)
    #     :return: rotation mat (3*3)
    #     """
    #     rotation_vec = np.expand_dims(np.array(self.pose[3:6]), 0)
    #     theta = np.linalg.norm(rotation_vec, 2)
    #     out_operator = np.array([
    #         [0, -theta[2], theta[1]],
    #         [theta[2], 0, -theta[0]],
    #         [-theta[1], theta[0], 0]
    #     ])
    #     rotation_mat = np.cos(theta)*np.eye(3) + (1 - np.cos(theta))*theta.T.dot(theta) + np.sin(theta)*out_operator
    #     return rotation_mat


if __name__ == '__main__':

    Ur = UrSocket()
    p1 = Ur.pose
    # Ur.change_pose(np.array([-0.3504593, -0.0419833473, 0.44227438, 2.1036984080, 2.250377223, 0.097387733]))
    while 1:
        p2 = list(Ur.get_pose())
        p2[2] -= 0.1
        print(p2)
        Ur.change_pose(p2)
        time.sleep(0.5)
        break
    # # print(p1)
    # # p2 = p1.copy()
    # # p3 = p1.copy()
    # # p4 = p1.copy()
    # # p2[2] -= 0.1
    # # p3[2] -= 0.1
    # # p3[1] -= 0.1
    # # p4[1] -= 0.1
    # #
    # # while 1:
    # #     Ur.change_pose(p2, a=0.1, v=0.2)
    # #     Ur.change_pose(p3, a=0.1, v=0.2)
    # #     Ur.change_pose(p4, a=0.1, v=0.2)
    # #     Ur.change_pose(p1, a=0.1, v=0.2)
    #
    #
    # while 1:
    #     # time.sleep(0.5)
    #     Ur.receive()
    #     data = Ur.data
    #     dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
    #            'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
    #            'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
    #            'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
    #            'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
    #            'Tool Accelerometer values': '3d',
    #            'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
    #            'V main': 'd',
    #            'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
    #            'Elbow position': '3d', 'Elbow velocity': '3d'}
    #
    #     names = []
    #     ii = range(len(dic))
    #     for key, i in zip(dic, ii):
    #         fmtsize = struct.calcsize(dic[key])
    #         data1, data = data[0:fmtsize], data[fmtsize:]
    #         fmt = "!" + dic[key]
    #         names.append(struct.unpack(fmt, data1))
    #         dic[key] = dic[key], struct.unpack(fmt, data1)
    #     # print(names)
    #     print(dic['Tool vector actual'])
        # for i in dic:
        #     print(i, dic[i])
        # print(Ur.pose)

    #     pose = np.array(Ur.get_pose())
    #     print(pose)
    #     # pose += np.array([0, 0.1, 0, 0, 0, 0])
    #     pose[2] = 0.1
    #     Ur.change_pose(pose)
    #     Ur.send()
    #     # Ur.doSocket.close()
    #     #
    #     # pose = [-0.35046452, -0.06334694, 0.35152359, 2.16532478, 2.25671654, 0.00707517]
    #
    #     # pose = [-0.04486961, -0.23975992, 0.05726057, 2.1661325, 2.2732668, -0.01262092]
    #     # Ur.change_pose(pose)
    #
    # main()









