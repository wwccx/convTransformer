import numpy as np
from matplotlib import pyplot as plt

loss = np.load('./train/resnet5022_03_13_15:09/loss_value.npy')
acc = np.load('./train/resnet5022_03_13_15:09/acc_value.npy')


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

