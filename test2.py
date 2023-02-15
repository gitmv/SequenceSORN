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


#print(f_i(0.019230769230769232, 15.5))

#x=np.arange(0,0.1,0.0001)
#plt.plot(x, fi_2(x, ci, px))
#plt.plot(x, fi_4())
#plt.plot(x, fe((x*5+0.5), 0.60416))
#plt.plot(x, f_i(x, 15.5))
#plt.plot(x, fi_2(x, 15.5, 0.018375))
#plt.plot(x, x*14.3+0.0144499)
#plt.plot(x, fi_3(x, 15.5, 0.018375))
#plt.show()


a=1
b=0
c=0

ex=0.60416

x=np.arange(0,1.0,0.0001)

plt.plot(x, a*np.power(x,ex)+b*x+c)#f
plt.plot(x, a*(1/ex)*np.power(x,ex-1)+b)#f'

#plt.plot(x, np.power(x,2))#f
#plt.plot(x, 0.5*x)#f'
plt.show()

ce = 0.60416
ci = 15.5
h = 0.018375
sp = 0.63
s = 14.3

scale = (sp-0.5)/(h-0)

x=np.arange(0,1.0,0.0001)
plt.plot(x, fe(x, ce))
plt.plot((x-h)*scale+sp, f_i(x, ci))
plt.plot((x-h)*scale+sp, x*s)
plt.show()

x=np.arange(0,1.0,0.0001)
plt.plot(x, f_e_derivative(x, ce)*scale)
plt.plot((x-h)*scale+sp, f_i_derivative(x, ci))
plt.plot((x-h)*scale+sp, x*0+s)
plt.show()


#for xp in x:
#    print(xp, f_e_derivative(xp, 0.60416)*(xp-0.5)/(0.01923-0))

#e=0.01445
#plt.plot(x, x*14.3+e)

#e=0.3 - 14.3 * 0.01923
#plt.plot(x, x*14.3+e)

#e=0.3 - 14.3 * 0.04
#plt.plot(x, x*14.3+e)

#plt.show()