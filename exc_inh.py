import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.clip((x - 0.5) * 2.0, 0.0, 1.0)

x = np.arange(0, 1, 0.01)
xp = np.power(x, 2)
rp = relu(x)

plt.plot(x, x)
plt.plot(x, x-xp*0.5)
plt.plot(x, x-rp*0.5)

#plt.plot(x, relu(x))
#plt.plot(x, np.power(x, 2))

plt.show()
