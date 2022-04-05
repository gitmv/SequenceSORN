import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return x-0.02

def f2(x, exp):
    return (x-0.02)*np.power(np.abs(x-0.02), exp)*10


#def f3(x, exp):
#    t = x-0.02
#    return np.power(np.abs(1+t), exp)*((t>0)*2-1)

def f4(x, slope):
    t = (x-0.02)*slope
    return t/np.sqrt(1+np.power(t, 2))*0.1


def f5(x):
    adj = (x-0.02) * 29.4# - 1  #
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 * 4.75#4.54

def f6(x):
    adj = (x) * 29.4# - 1  #-0.02
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 * 4.75#4.54

def f7(x):
    adj = (x-0.02) * 29.4# - 1  #
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 * 4.75 +0.24#4.54

def f8(x, c=0.02, s=170):
    return np.clip(np.clip(x-c, 0, None)*s,0,1.0)


def f9(x):
    adj = (x-0.02) * 29.4# - 1  #
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1

def ft(x,y=20):
    return np.tanh(x*y)#*0.15

def fe(x):
    return np.power(np.clip(x-0.5,0,1),0.614)



from Grammar.SORN_Grammar.Behaviours_in_use.test import *

x = np.arange(0.0, 0.3, 0.0001)#0.2

plt.axhline(0, color='gray')

plt.axvline(0.02, color='gray')

plt.plot(x, f8(x,0.02,170))

plt.plot(x, f8(ft(x,20),0.37994896225,10))#0.13909244787

#plt.plot(x, f1(x))
#plt.plot(x, f2(x, 0.3))
#plt.plot(x, f2(x, 0.4))
#plt.plot(x, f2(x, 0.5))

#plt.plot(x, f4(x, 1))
#plt.plot(x, f4(x, 20))

#plt.plot(x, f5(x), label='direct inhibition')
#plt.plot(x, f6(x))
#plt.plot(x, f7(x), label='interneuron inhibition')

#plt.plot(x, inhibition_func(x, 29.4, 1, 0.050686943101760265))
#plt.plot(x, f8(x,0.02))
#plt.plot(x, f8(x,0.13909244787))

#plt.plot(x, ft(x))
#plt.plot(x, ft(x,20))
#plt.plot(x, ft(x,10))
#plt.plot(x, ft(x,30))
#plt.plot(x, ft(x,30))

#plt.plot(x, ft(x,7))

#plt.plot(x, f4(x, 100))
#plt.plot(x, f4(x, 1000))

#plt.plot(x, f4(x, 5))
#plt.plot(x, f3(x, 2))

#plt.plot(x, f2(x, 1.0))
#plt.plot(x, f2(x, 1.5))
#plt.plot(x, f2(x, 2.0))



plt.legend(loc='best', frameon=False, fontsize=20)

plt.show()

'''
def f(x):
    return np.power(np.abs(x - 0.5) * 2, 0.614) * (x > 0.5)

x = np.arange(0.0, 1.0, 0.0001)

plt.plot(x, f(x))

plt.show()
'''