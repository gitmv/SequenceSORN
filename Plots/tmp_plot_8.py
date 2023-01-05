import matplotlib.pyplot as plt
import numpy as np
import random

# Import the libraries
from matplotlib import cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0, 1, 0.001)

plt.plot(x, np.power(x,2))
plt.plot(x, 2*x)
plt.plot(x, x*0+2)

plt.show()
