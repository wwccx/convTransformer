import serial

# sudo chmod 777 /dev/ttyACM0


class Gripper():
    def __init__(self):
        self.ser = serial.Serial("/dev/ttyACM0", 115200)
        # self.gripper_initial()

    def gripper_initial(self):  # 初始化
        self.ser.write('FFFEFDFC 01 0802 0100 00000000 FB'.encode())
        # 写入的命令总共有14字节，前4字节是帧头保持不变，第五位字节是夹爪ID默认为1，
        # 然后是功能标志2字节，前1字节表示主功能，后一字节表示主功能下划分的子功能。
        # 然后一字节的读写，01为写，00为读，之后一位为保留字00
        # 四字节的数据，最后1字节为帧尾。
        data = self.ser.read(42)
        # data = self.ser.read(42)
        # 设置主动反馈后返回FF FE FD FC 01 08 02 00 00 01 00 00 00 FB
        return

    def gripper_force(self, force):  # 设置夹持力，数值范围为20~100,，
        f = hex(force)
        e = f[2:4]
        if e.__len__() == 1:
            e = '0' + e

        s = 'FFFEFDFC0105020100' + e + '000000FB'
        self.ser.write(s.encode())
        self.ser.read(42)
        return

    def gripper_position(self, positon):  # 设置夹爪的张开程度，数值范围0~100
        p = hex(positon)
        e = p[2:4]
        if e.__len__() == 1:
            e = '0' + e

        s = 'FFFEFDFC0106020100' + e + '000000FB'
        self.ser.write(s.encode())
        self.ser.read(42)
        return

    def get_position(self):
        s = 'FFFEFDFC0106020000' + '00' + '000000FB'
        self.ser.write(s.encode())
        print(self.ser.read(42))

    def check_grasp(self):
        s = 'FFFEFDFC010F01000000000000FB'
        self.ser.write(s.encode())
        data = self.ser.read(42)
        while data[12:17] != b'01 0F':
            data = self.ser.read(42)

        if data == b'FF FE FD FC 01 0F 01 00 00 03 00 00 00 FB ':
            return True
        else:
            return False


if __name__ == '__main__':
    import time
    g = Gripper()
    # g.gripper_initial()
    # time.sleep(4)
    # print('???')
    # print(g.get_position())
    g.gripper_position(100)
    # print(g.check_grasp())
    # g = Gripper()
    # g.gripper_initial()
    # time.sleep(10)
    # g.gripper_position(0)
    # time.sleep(3)
    # g.check_grasp()
    # g.gripper_force(50)
    # import time
    # time.sleep(0.2)
    # g.gripper_position(0)
    # g.gripper_position(30)
    # g.gripper_position(100)
    # g.gripper_position(30)



