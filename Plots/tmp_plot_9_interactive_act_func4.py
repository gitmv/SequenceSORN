import matplotlib.pyplot as plt
import numpy as np
import random

# Import the libraries
from matplotlib import cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.widgets import Slider, Button

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

def f_LI4(gaba, th, s):#s=8.158
    return np.clip(s-gaba*s/th, -1.0, 1.0)# #s*(1-gaba/th)



class interactive_fi_ploter():

       def __init__(self):
              self.plot_funcs = []
              self.variables = []

              self.left_plots = []

              self.checkboxes = {}
              self.checkbox_p = 0.05

              self.sliders = {}
              self.slider_p = 0.05

              self.fig, self.ax = plt.subplots(1, 1)

              plt.subplot(111)
              plt.axis([-1.5, 1.5, -1.5, 1.5])
              plt.xlabel('activity (voltage)')
              plt.ylabel('spike chance')


       def add_slider(self, name, vmin, vmax, vinit):
              self.sliders[name] = Slider(ax=plt.axes([self.slider_p, 0.95, 0.1, 0.05]), label=name, valmin=vmin, valmax=vmax, valinit=vinit)
              self.sliders[name].on_changed(self.update)
              self.slider_p += 0.2

       def add_variable(self, var_eval):
              self.variables.append(var_eval)

       def add_plot(self, plot_eval):
              self.plot_funcs.append(plot_eval)

              plt.subplot(111)
              pl, = plt.plot([], [])
              self.left_plots.append(pl)

              self.checkboxes[plot_eval] = Slider(ax=plt.axes([self.checkbox_p, 0.90, 0.1, 0.05]), label=plot_eval, valmin=0, valmax=1, valinit=1)
              self.checkboxes[plot_eval].on_changed(self.update)
              self.checkbox_p += 0.2


       def update(self, val):
              x = np.arange(-5, 5, 0.0001)

              for name, slider in self.sliders.items():
                     globals()[name] = slider.val

              for var in self.variables:
                     exec(var)

              for pfunc, lplot in zip(self.plot_funcs, self.left_plots):
                     _y = eval(pfunc)
                     lplot.set_xdata(x)
                     lplot.set_ydata(_y)

                     if self.checkboxes[pfunc].val<1:
                            lplot.set_xdata([])
                            lplot.set_ydata([])

              self.fig.canvas.draw_idle()


       def show(self):
              self.update(0)
              plt.show()

'''
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
'''

plotter = interactive_fi_ploter()

avg_inh = 0.28

plotter.add_slider('h', 0.0, 0.2, 0.02)
#plotter.add_variable('')#h2
plotter.add_plot('fi4(x, avg_inh, h)')
plotter.add_plot('fe3(x, 0.5716505035597882, 2.0)')
#plotter.add_plot('f_LI3(gaba, 0.2822661546904615, 8.158)')
plotter.add_plot('f_LI3(x, avg_inh, 8.158)')
plotter.add_plot('f_LI4(x, avg_inh, 8.158)')
plotter.add_plot('avg_inh')




plotter.show()








