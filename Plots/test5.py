import numpy as np
import matplotlib.pyplot as plt

def get_ce(h):
       return 0.01 / h + 0.22

def get_ci(h):
       return 0.4 / h + 3.6



def nth_sqrt(x,n):
       return np.power(x, (1 / n))

def f_e(x, ce):
       return np.clip(np.power(np.abs(x - 0.5) * 2, ce) * (x>0.5), 0, 1)

def f_e2(x,ce):
       return np.clip( np.power(2*x - 1, ce) ,0,1)

def f_e_derivative(x, ce):
       return 2 * ce * np.power(2*x-1, ce-1)

def f_e_inverted(h, ce):
       return 0.5*(nth_sqrt(h, ce)+1)


def f_i(x, ci):
       return np.tanh(x * ci)

def f_i_derivative(x, ci):
       return (4*ci)/np.power(np.exp(-ci*x)+np.exp(ci*x), 2)

def rotate_function(x, y, degree, cx=0, cy=0):
       xs = x-cx
       ys = y-cy
       xr = xs*np.cos(degree)-ys*np.sin(degree)
       yr = xs*np.sin(degree)+ys*np.cos(degree)
       #xr = xr+cx
       #yr = yr+cy
       return xr, yr

def center_function(x, y, cx=0, cy=0):
       return x-cx, y-cy


def get_FI_scale_factor(aI_fix, aE_fix, aI_zero=0, aE_zero=0.5): #for fE FI alignment (fE x stretch)
       return (aE_fix-aE_zero)/(aI_fix-aI_zero)


def fi_2(x, ci, px):
    #px = 0.02
    fx = f_i(px, ci)
    fdx = f_i_derivative(px, ci)
    return fx + (x-px) * fdx
    #0.02+x, fx+x*fdx

def fi2(x, ci, e):
       return x*ci+e

def fi3(x, ci):
       return x*ci


def fi4(x, slope, target):
    return np.clip((x * slope)/target,0,1)

def fe2(x, ce):
    return np.power(np.clip(x, 0.0, 1.0), ce)

def fe3(x, ce, f):
    return np.power(np.clip(x*f, 0.0, 1.0), ce)


def f_LI(gaba, th, s):
    return np.clip(1 - (gaba - th) * s, 0.0, 1.0)

def f_LI2(gaba, th, s):
    return np.clip((th-gaba) * s, 0.0, 1.0)

def f_LI3(gaba, th, s):#s=8.158
    return np.clip(s-gaba*s/th, 0.0, 1.0)# #s*(1-gaba/th)

#40: LearningInhibition(transmitter='GABA', strength=29.25278222816641, threshold=0.2462380365675854),
#70: Output_Inhibitory(slope=14.677840043903998, duration=2),

#x=np.arange(0,1.0,0.0001)

#plt.plot(x, np.clip(fe(x+0.5, ce), 0.0, 1.0))
#plt.plot(x, fi3(x, 14.677840043903998))
#plt.plot(x, f_LI(x, 0.2462380365675854, 29.25278222816641))

#plt.plot(x, f_LI(fi3(x, 14.677840043903998), 0.2462380365675854, 29.25278222816641))

#plt.plot(x, f_LI2(x, fi3(0.019, 14.677840043903998), 29.25278222816641))
#plt.plot(x, f_LI2(x, fi3(0.016, 14.677840043903998), 29.25278222816641))

#plt.plot(x, f_LI3(x, fi3(0.019, 14.677840043903998), 8.158))
#plt.plot(x, f_LI3(x, fi3(0.016, 14.677840043903998), 8.158))
#plt.plot(x, f_LI3(x, fi3(0.010, 14.677840043903998), 8.158))


x=np.arange(0,0.5,0.0001)

h=0.021#0.019

plt.plot(x, fe3(x, 0.5716505035597882, 2.0))

gaba=fi4(x, 0.2822661546904615, h)
plt.plot(x, gaba)

#inh=fi4(0.019, 0.2822661546904615, 0.019)
#print(inh)
plt.plot(x, f_LI3(gaba, 0.2822661546904615, 8.158))

plt.axvline(h)
plt.axhline(0.2822661546904615)
#plt.axvline(0.2822661546904615)

#plt.axvline(0.019)
#plt.axvline(fi3(0.019, 14.677840043903998))
#plt.axvline(0.2462380365675854)




plt.show()



#def iteration(self, neurons):
#    o = np.abs(getattr(neurons, self.input_tag))
#    neurons.linh = np.clip(1 - (o - neurons.LI_threshold) * self.strength, 0.0, 1.0)