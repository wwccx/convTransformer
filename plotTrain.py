import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dicts", type=int, default=1)
parser.add_argument("--kernel-size", type=int, default=300)
parser.add_argument("--log", type=bool, default=False)
parser.add_argument("--compare", type=int, default=0)

out_path = './train'
out_dict = os.listdir(out_path)
# training_dict = max([os.path.join(out_path, d) for d in out_dict], key=os.path.getmtime)
opt = parser.parse_args()
index = opt.dicts
training_dict = [os.path.join(out_path, d) for d in out_dict]
training_dict.sort(key=os.path.getmtime)
print(training_dict[-index])
loss_conv = np.load(os.path.join(training_dict[-index], 'loss_value.npy'))
try:
    acc_conv = np.load(os.path.join(training_dict[-index], 'acc_value.npy'))
except:
    acc_conv = np.array([0])
try:
    lr_conv = np.load(os.path.join(training_dict[-index], 'lr_value.npy'))
except:
    lr_conv = np.array([0])
try:
    grad_conv = np.load(os.path.join(training_dict[-index], 'grad_value.npy'))
except:
    grad_conv = np.array([0])
if opt.compare:
    i = opt.compare
    loss_com = np.load(os.path.join(training_dict[-i], 'loss_value.npy'))
    try:
        acc_com = np.load(os.path.join(training_dict[-i], 'acc_value.npy'))
    except:
        acc_com = np.array([0])
    try:
        lr_com = np.load(os.path.join(training_dict[-i], 'lr_value.npy'))
    except:
        lr_com = np.array([0])
    try:
        grad_com = np.load(os.path.join(training_dict[-i], 'grad_value.npy'))
    except:
        grad_com = np.array([0])
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
if opt.compare:

    loss_com = pooling(loss_com, kernel_size=kernel_size)
    grad_com = pooling(grad_com, kernel_size)
plt.subplot(2, 2, 1)
start_index = 0
plt.plot(loss_conv[start_index:], label='loss')
if opt.compare:
    plt.plot(loss_com[start_index:], label='compare loss')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(acc_conv[:], label='accuracy')
if opt.compare:
    plt.plot(acc_com[:], label='compare acc')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(lr_conv[:], label='lr')
if opt.compare:
    plt.plot(lr_com[start_index:], label='compare lr')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(grad_conv[:], label='grad')
if opt.compare:
    plt.plot(grad_com[start_index:], label='compare grad')
plt.legend()
plt.show()

