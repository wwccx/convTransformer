import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dicts", type=int, default=1)
parser.add_argument("--kernel-size", type=int, default=300)
parser.add_argument("--log", type=bool, default=False)
parser.add_argument("--compare", type=str, default='')
parser.add_argument("--legend", type=str, default='')

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
    compare_dict = list(map(int, opt.compare.split(' ')))
legend = opt.legend.split(' ')

if opt.compare:
    assert len(compare_dict) == len(legend) - 1 or len(legend) == 0, 'legend length must be equal to compare length or empty!'

    loss_com = []
    acc_com = []
    lr_com = []
    grad_com = []
    for i in compare_dict:
        loss_com.append(np.load(os.path.join(training_dict[-i], 'loss_value.npy')))
        try:
            acc_com.append(np.load(os.path.join(training_dict[-i], 'acc_value.npy')))
        except:
            acc_com.append(np.array([0]))
        try:
            lr_com.append(np.load(os.path.join(training_dict[-i], 'lr_value.npy')))
        except:
            lr_com.append(np.array([0]))
        try:
            grad_com.append(np.load(os.path.join(training_dict[-i], 'grad_value.npy')))
        except:
            grad_com.append(np.array([0]))
if len(legend) == 0:
    legend = ['self']

def pooling(a, kernel_size=10):
    output_shape = (a.shape[0] - kernel_size) // kernel_size
    a = a[:output_shape*kernel_size]
    a = a.reshape((-1, kernel_size))
    return a.mean(axis=1)

if opt.log:
    loss_conv = np.log(loss_conv)
    grad_conv = np.log(grad_conv)

kernel_size = opt.kernel_size
loss_conv = pooling(loss_conv, kernel_size=kernel_size)
grad_conv = pooling(grad_conv, kernel_size)

if opt.compare:
    for i in range(len(loss_com)):
        loss_com[i] = pooling(loss_com[i], kernel_size)
        grad_com[i] = pooling(grad_com[i], kernel_size)
plt.subplot(2, 2, 1)
plt.title('loss')
start_index = 0
plt.plot(loss_conv[start_index:], label=legend[0])
if opt.compare:
    for i in range(len(loss_com)):
        plt.plot(loss_com[i][start_index:], label=legend[i + 1])
plt.legend()
plt.subplot(2, 2, 2)
plt.title('accuracy')
plt.plot(acc_conv[:], label=legend[0])
if opt.compare:
    for i in range(len(loss_com)):
        plt.plot(acc_com[i][:], label=legend[i + 1])
plt.legend()
plt.subplot(2, 2, 3)
plt.title('lr')
plt.plot(lr_conv[:], label=legend[0])
if opt.compare:
    for i in range(len(loss_com)):
        plt.plot(lr_com[i][start_index:], label=legend[i + 1])
plt.legend()
plt.subplot(2, 2, 4)
plt.title('grad')
plt.plot(grad_conv[:], label=legend[0])
if opt.compare:
    for i in range(len(loss_com)):
        plt.plot(grad_com[i][start_index:], label=legend[i + 1])
plt.legend()
plt.show()

