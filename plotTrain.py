import numpy as np
from matplotlib import pyplot as plt
import os

out_path = './train'
out_dict = os.listdir(out_path)
# training_dict = max([os.path.join(out_path, d) for d in out_dict], key=os.path.getmtime)

training_dict = [os.path.join(out_path, d) for d in out_dict]
training_dict.sort(key=os.path.getmtime)
loss_conv = np.load(os.path.join(training_dict[-1], 'loss_value.npy'))
acc_conv = np.load(os.path.join(training_dict[-1], 'acc_value.npy'))
lr_conv = np.load(os.path.join(training_dict[-1], 'lr_value.npy'))
def pooling(a, kernel_size=10):
    output_shape = (a.shape[0] - kernel_size) // kernel_size
    a = a[:output_shape*kernel_size]
    a = a.reshape((-1, kernel_size))
    return a.mean(axis=1)

kernel_size=300
loss_conv = pooling(loss_conv, kernel_size=kernel_size)
plt.figure(1)
start_index = 0
plt.plot(loss_conv[start_index:], label='convT')
plt.legend()
plt.figure(2)
plt.plot(acc_conv[1:], label='conv')
plt.figure(3)
plt.plot(lr_conv[1:], label='lr')
plt.legend()
plt.show()

