from multiprocessing import Process, Queue
from  threading import Thread
import time
from vel_control import VelController
import cv2
import numpy as np


if __name__ == '__main__':
    q = Queue()
    vc = VelController(q)
    p = Thread(target=vc.run)
    print(vc.get_current_tool_pose())
    p.start()
    pos, ori = vc.get_current_tool_pose()
    pos_bias = np.array([0, 0, 0.05])
    ori_bias = cv2.Rodrigues(np.array([0, 0, np.pi/12]))[0]
    for i in range(3):
        pos += pos_bias
        ori = ori_bias @ ori
        q.put((pos, ori))
        time.sleep(3)
    p.terminate()
    vc.end_joint_rotation()
