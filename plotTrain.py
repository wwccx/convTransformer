import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dicts", type=int, default=1)
parser.add_argument("--kernel-size", type=int, default=300)
parser.add_argument("--log", type=bool, default=False)

out_path = './train'
out_dict = os.listdir(out_path)
# training_dict = max([os.path.join(out_path, d) for d in out_dict], key=os.path.getmtime)
opt = parser.parse_args()
index = opt.dicts
training_dict = [os.path.join(out_path, d) for d in out_dict]
training_dict.sort(key=os.path.getmtime)
loss_conv = np.load(os.path.join(training_dict[-index], 'loss_value.npy'))
acc_conv = np.load(os.path.join(training_dict[-index], 'acc_value.npy'))
lr_conv = np.load(os.path.join(training_dict[-index], 'lr_value.npy'))
try:
    grad_conv = np.load(os.path.join(training_dict[-index], 'grad_value.npy'))
except:
    grad_conv = np.array([0])
def pooling(a, kernel_size=10):
    output_shape = (a.shape[0] - kernel_size) // kernel_size
    a = a[:output_shape*kernel_size]
    a = a.reshape((-1, kernel_size))
    return a.mean(axis=1)
if opt.log:
    loss_conv = np.log(loss_conv)
    grad_conv = np.log(grad_conv)
kernel_size=opt.kernel_size
loss_conv = pooling(loss_conv, kernel_size=kernel_size)
grad_conv = pooling(grad_conv, kernel_size)
plt.subplot(4, 1, 1)
start_index = 0
plt.plot(loss_conv[start_index:], label='loss')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(acc_conv[:], label='accuracy')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(lr_conv[:], label='lr')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(grad_conv[:], label='grad')
plt.legend()
plt.show()

