import numpy as np
import cv2

a = np.array([[10, 10], [10, 200], [200, 10], [200, 200]])
b = np.array([[10, 10], [8, 180], [180, 8], [180, 180]])
print(cv2.findHomography(a, a//2))
