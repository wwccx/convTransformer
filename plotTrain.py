import numpy as np
from matplotlib import pyplot as plt
import os

out_path = './train'
out_dict = os.listdir(out_path)
training_dict = max([os.path.join(out_path, d) for d in out_dict], key=os.path.getmtime)

loss = np.load(os.path.join(training_dict, 'loss_value.npy'))
acc = np.load(os.path.join(training_dict, 'acc_value.npy'))


def pooling(a, kernel_size=10):
    output_shape = (a.shape[0] - kernel_size) // kernel_size
    a = a[:output_shape*kernel_size]
    a = a.reshape((-1, kernel_size))
    return a.mean(axis=1)


loss = pooling(loss, kernel_size=10)
plt.figure(1)
plt.plot(loss, label='convTrans loss')
plt.legend()
plt.figure(2)
plt.plot(acc, label='acc on CIFAR10')
plt.legend()
plt.show()

