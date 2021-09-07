import matplotlib.pyplot as plt
import numpy as np

def R(p):
    return np.random.rand() < p

h = 0.02
a = 0.007

P = 0.5
I = -1

x = list(range(1000))
Py = []
Iy = []

for i in x:
    Py.append(P)
    Iy.append(I)
    

    P = I + 0.5
    I = I - (R(P) - h) * a


plt.plot(x, Py)
plt.plot(x, Iy)
plt.show()



