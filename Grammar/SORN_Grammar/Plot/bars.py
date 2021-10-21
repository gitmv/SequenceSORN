import matplotlib.pyplot as plt
import numpy as np

labels = ['post normalization 30k', 'pre/post normalization 30k']#, 'post normalization 60k', 'pre/post normalization 60k'

bars = np.array([[8.092504681140971, 7.957843982655811, 7.866961064738506, 7.8952831230721126],
                 [8.606395955337627, 9.040536546592282, 8.751880088292427, 9.012484300260896],
                 ])


b_mean = np.mean(bars, axis=1)
b_std = np.std(bars, axis=1)

plt.bar(np.arange(len(labels)), b_mean, yerr=b_std, align='center', alpha=0.5, ecolor='black', capsize=10)

plt.xticks(np.arange(len(labels)), labels)

plt.show()
