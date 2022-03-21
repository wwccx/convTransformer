import numpy as np
from matplotlib import pyplot as plt
import os

out_path = './train'
out_dict = os.listdir(out_path)
# training_dict = max([os.path.join(out_path, d) for d in out_dict], key=os.path.getmtime)

training_dict = [os.path.join(out_path, d) for d in out_dict]
training_dict.sort(key=os.path.getmtime)
loss_swin = np.load(os.path.join(training_dict[-1], 'loss_value.npy'))
acc_swin = np.load(os.path.join(training_dict[-1], 'acc_value.npy'))

loss_conv = np.load(os.path.join(training_dict[-2], 'loss_value.npy'))
acc_conv = np.load(os.path.join(training_dict[-2], 'acc_value.npy'))

def pooling(a, kernel_size=10):
    output_shape = (a.shape[0] - kernel_size) // kernel_size
    a = a[:output_shape*kernel_size]
    a = a.reshape((-1, kernel_size))
    return a.mean(axis=1)


loss_swin = pooling(loss_swin, kernel_size=600)
loss_conv = pooling(loss_conv, kernel_size=600)
plt.figure(1)
start_index = 900
plt.plot(loss_swin[start_index:], label='swinT')
plt.plot(loss_conv[start_index:len(loss_swin)], label='convT')
plt.legend()
plt.figure(2)
plt.plot(acc_swin[:], label='swin')
plt.plot(acc_conv[:], label='conv')
plt.legend()
plt.show()

