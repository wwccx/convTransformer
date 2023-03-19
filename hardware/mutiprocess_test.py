from multiprocessing import Process, Queue
from threading import Thread
import time
# from vel_control import VelController
import cv2
import numpy as np
from camera import RS
from matplotlib import pyplot as plt
import utils as u


if __name__ == '__main__':
    c = RS(640, 480)
    depth, rgb = c.get_img()
    depth = u.in_paint(depth)
    # depth = np.clip(depth / 1000.0, 0.1, 0.8)
    print('inpainted depth min:', np.min(depth))
    plt.imshow(depth)
    plt.show()


    # q = Queue()
    # vc = VelController(q)
    # p = Thread(target=vc.run)
    # print(vc.get_current_tool_pose())
    # p.start()
    # pos, ori = vc.get_current_tool_pose()
    # pos_bias = np.array([0, 0, 0.1])
    # ori_bias = cv2.Rodrigues(np.array([0, 0, np.pi/6]))[0]
    # for i in range(3):
    #     pos += pos_bias
    #     ori = ori_bias @ ori
    #     q.put((pos, ori))
    #     time.sleep(3)
    # q.put(None)
    # vc.end_joint_rotation()
