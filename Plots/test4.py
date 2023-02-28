import numpy as np
import matplotlib.pyplot as plt

def f_i(x, ci):
    return np.tanh(x * ci)

def f_i_derivative(x, ci):
    return (4 * ci) / np.power(np.exp(-ci * x) + np.exp(ci * x), 2)

def fi_2(x, ci, px):
    fx = f_i(px, ci)
    fdx = f_i_derivative(px, ci)
    print(fx-px*fdx)
    print(fdx)
    return fx + (x - px) * fdx

def fi_3(x, ci, px):
    fdx = f_i_derivative(px, ci)
    return x * fdx

def fi_4(x):
    return x * 15.5

def fe(x, ce):
    return np.power(np.abs(x - 0.5) * 2, ce) * (x > 0.5)


def f_e_derivative(x, ce):
    return 2 * ce * np.power(2 * x - 1, ce - 1)


def fe2(x, ce):
    return np.power(np.clip(x, 0.0, 1.0), ce)

def fe3(x, ce, f):
    return np.power(np.clip(x*f, 0.0, 1.0), ce)

ce = 0.60416
ci = 15.5
h = 0.018375
sp = 0.25#0.63
s = 14.3

scale = (sp-0.0)/(h-0)

x=np.arange(0,1.0,0.0001)
plt.plot(x, np.clip(fe(x+0.5, ce), 0.0, 1.0))
plt.plot(x, fe2(x, 0.4))
plt.plot(x, fe3(x, ce, 2))
plt.show()



#x=np.arange(0,1.0,0.0001)
#plt.plot(x, f_e_derivative(x, ce)*scale)
#plt.plot((x-h)*scale+sp, f_i_derivative(x, ci))
#plt.plot((x-h)*scale+sp, x*0+s)
#plt.show()