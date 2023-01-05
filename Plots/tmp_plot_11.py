import matplotlib.pyplot as plt
import numpy as np
import random

# Import the libraries
from matplotlib import cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D


def get_ce(h):
    return 0.01 / h + 0.22


def get_ci(h):
    return 0.4 / h + 3.6


def nth_sqrt(x, n):
    return np.power(x, (1 / n))


def f_e(x, ce):
    return np.power(np.abs(x - 0.5) * 2, ce) * (x > 0.5)


def f_e2(x, ce):
    return np.clip(np.power(2 * x - 1, ce), 0, 1)


def f_e_derivative(x, ce):
    return 2 * ce * np.power(2 * x - 1, ce - 1)


def f_e_inverted(h, ce):
    return 0.5 * (nth_sqrt(h, ce) + 1)


def f_i(x, ci):
    return np.tanh(x * ci)


def f_i_derivative(x, ci):
    return (4 * ci) / np.power(np.exp(-ci * x) + np.exp(ci * x), 2)


def rotate_function(x, y, degree, cx=0, cy=0):
    xs = x - cx
    ys = y - cy
    xr = xs * np.cos(degree) - ys * np.sin(degree)
    yr = xs * np.sin(degree) + ys * np.cos(degree)
    # xr = xr+cx
    # yr = yr+cy
    return xr, yr


def center_function(x, y, cx=0, cy=0):
    return x - cx, y - cy


def get_FI_scale_factor(aI_fix, aE_fix, aI_zero=0, aE_zero=0.5):  # for fE FI alignment (fE x stretch)
    return (aE_fix - aE_zero) / (aI_fix - aI_zero)

def fi_2(x):
    px = 0.02
    cx = 24.4
    fx = f_i(px, cx)
    fdx = f_i_derivative(px, cx)
    return fx + (x-px) * fdx
    #0.02+x, fx+x*fdx

x = np.arange(0, 0.1, 0.001)
#y = np.arange(0, 1, 0.001)

plt.plot(x, f_i(x, 24.4))
#plt.plot(x, f_i_derivative(x, 24.4))

#plt.plot(0.02+x, fx+x*fdx)
plt.plot(x, fi_2(x))

plt.scatter([0.02],[f_i(0.02, 24.4)])

#print(f_i_derivative(0.02, 24.4))

#plt.plot(x, np.power(x,2))
#plt.plot(x, 2*x)
#plt.plot(x, x*0+2)

plt.show()

