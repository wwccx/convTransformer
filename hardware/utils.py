import numpy as np
import cv2
import tf.transformations as tftrans


def vec2mat(rot_vec, trans_vec):
    """
    :param rot_vec: list_like [a, b, c]
    :param trans_vec: list_like [x, y, z]
    :return: transform mat 4*4
    """
    theta = np.linalg.norm(rot_vec, 2)
    if theta:
        rot_vec /= theta
    out_operator = np.array([
        [0, -rot_vec[2], rot_vec[1]],
        [rot_vec[2], 0, -rot_vec[0]],
        [-rot_vec[1], rot_vec[0], 0]
    ])
    rot_vec = np.expand_dims(rot_vec, 0)
    rot_mat = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * rot_vec.T.dot(rot_vec) + np.sin(
        theta) * out_operator

    trans_vec = np.expand_dims(trans_vec, 0).T

    trans_mat = np.vstack(
        (np.hstack((rot_mat, trans_vec)), [0, 0, 0, 1])
    )

    return trans_mat


def in_paint(depth_img):
    depth_img = np.array(depth_img).astype(np.float32)
    depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (depth_img == 0).astype(np.uint8)

    depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
    depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
    # Back to original size and value range.
    depth_img = depth_img[1:-1, 1:-1]

    return depth_img


def quat2RotMat(quat):
    # print('quat:', quat)
    q = np.array(quat)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(3)
    q /= n
    (x, y, z, w) = q
    rotMat = np.array(
        [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
         [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
         [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
         ], dtype=q.dtype
    )
    
    return rotMat


def rotMat2Quat(R):
    w = np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1) / 2
    x = -(R[1, 2] - R[2, 1]) / w / 4
    y = -(R[2, 0] - R[0, 2]) / w / 4
    z = -(R[0, 1] - R[1, 0]) / w / 4
    q = np.array([x, y, z, w])
    # q = np.array([x, y, z, w])
    n = np.dot(q, q)
    q /= n
    return q


def pos_quat2mat(pos, quat):
    mat = tftrans.quaternion_matrix(quat)
    mat[0:3, 3] = pos

    return mat

def pos_ori2mat(pos, ori):
    mat = np.eye(4)
    mat[0:3, 0:3] = ori
    mat[0:3, 3] = pos
    return mat

def rpy2quat(rpy):
    return np.array(tftrans.quaternion_from_euler(*rpy))

def quat2rpy(quat):
    return np.array(tftrans.euler_from_quaternion(quat))



