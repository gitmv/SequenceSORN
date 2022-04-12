mnist_folder = '../EMNIST'
import numpy as np

from mnist.loader import MNIST  # pip install python-mnist
mndata = MNIST(mnist_folder)
mndata.select_emnist('balanced')
mnist_pictures, mnist_labels = mndata.load_testing()

import matplotlib.pyplot as plt

chars = {}
for i in range(26):
    chars[10+i] = []

m = 0
for i in range(1000):
    if mnist_labels[i] in chars:#26 #== 9+1
        chars[mnist_labels[i]].append(np.array(mnist_pictures[i]).reshape(28, 28))
        m = np.maximum(m, len(chars[mnist_labels[i]]))


for c in chars:
    l = np.zeros((28, 28))
    for i in range(m):
        if i < len(chars[c]):
            t = chars[c][i]
        else:
            t = np.zeros((28, 28))
        l = np.concatenate([l, t], axis=1)
    chars[c] = l

p = np.zeros((28, 28*m+28))
for c in chars:
    p = np.concatenate([p, chars[c]], axis=0)

plt.matshow(p)
plt.show()
