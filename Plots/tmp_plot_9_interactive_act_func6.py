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

def fe4(x, f):
    return np.clip(x*f, 0.0, 1.0)

class interactive_fi_ploter():

       def __init__(self):
              self.plot_funcs = []#'f_e(x,...)'
              self.plot_stable_points = []
              self.plot_zero_points = []
              self.variables = []

              self.left_plots = []
              self.left_scatters = []
              self.right_plots = []

              self.checkboxes = {}
              self.checkbox_p = 0.05

              self.sliders = {}
              self.slider_p = 0.05

              self.fig, self.ax = plt.subplots(1, 2)

              plt.subplot(121)
              plt.axis([0, 1, 0, 1])
              plt.xlabel('activity (voltage)')
              plt.ylabel('spike chance')

              plt.subplot(122)
              plt.axis([-2, 2, -1, 1])
              plt.scatter([0], [0])

       def add_slider(self, name, vmin, vmax, vinit):
              self.sliders[name] = Slider(ax=plt.axes([self.slider_p, 0.95, 0.1, 0.05]), label=name, valmin=vmin, valmax=vmax, valinit=vinit)
              self.sliders[name].on_changed(self.update)
              self.slider_p += 0.2

       def add_variable(self, var_eval):
              self.variables.append(var_eval)

       def add_plot(self, plot_eval, stable_eval, zero_eval='0'):
              self.plot_funcs.append(plot_eval)
              self.plot_stable_points.append(stable_eval)
              self.plot_zero_points.append(zero_eval)

              plt.subplot(121)
              pl, = plt.plot([], [])
              self.left_plots.append(pl)
              sl = plt.scatter([], [])
              self.left_scatters.append(sl)

              plt.subplot(122)
              p2, = plt.plot([], [])
              self.right_plots.append(p2)

              self.checkboxes[plot_eval] = Slider(ax=plt.axes([self.checkbox_p, 0.90, 0.1, 0.05]), label=plot_eval, valmin=0, valmax=1, valinit=1)
              self.checkboxes[plot_eval].on_changed(self.update)
              self.checkbox_p += 0.2

       def update(self, val):
              for name, slider in self.sliders.items():
                     globals()[name] = slider.val

              for var in self.variables:
                     exec(var)

              for pfunc, sfunc, zfunc, lplot, scatter, rplot in zip(self.plot_funcs,
                                                                    self.plot_stable_points,
                                                                    self.plot_zero_points,
                                                                    self.left_plots,
                                                                    self.left_scatters,
                                                                    self.right_plots):
                     x = np.arange(-5, 5, 0.0001)
                     _y = eval(pfunc)
                     lplot.set_xdata(x)
                     lplot.set_ydata(_y)

                     x = eval(sfunc)
                     _y = eval(pfunc)
                     scatter.set_offsets(np.c_[x, _y])

                     _z = eval(zfunc)

                     _px=x
                     _py=_y

                     x = np.arange(-5, 5, 0.0001)
                     _x1 = (x - _px) / (_px-_z)
                     _y = eval(pfunc) - _py

                     rplot.set_xdata(_x1)
                     rplot.set_ydata(_y)


                     if self.checkboxes[pfunc].val<1:
                            lplot.set_xdata([])
                            lplot.set_ydata([])
                            rplot.set_xdata([])
                            rplot.set_ydata([])
                            scatter.set_offsets(np.c_[[], []])

              self.fig.canvas.draw_idle()


       def show(self):
              self.update(0)
              plt.show()



ifi_plotter = interactive_fi_ploter()

#ifi_plotter.add_slider('h', 0.0, 0.2, 0.02)
ifi_plotter.add_slider('a', 0.0, 1.0, 0.2)

IP_s = 0.008735764741458582
avg_inh = 0.3427857658747104

STDP_s = 0.018 #0.006 #0.018 is worse with 200norm! works with 50norm. works with 50norm

ifi_plotter.add_plot('fi4(x, 0.3427857658747104, 0.019230769230769232)', '0.019230769230769232') # slope target


#3s
#fe_exp = gene('fe_exp', 0.7378726012049153)#important!
#fe_mul = gene('fe_mul', 2.353594052973287)#important!
ifi_plotter.add_plot('fe3(x, 0.7378726012049153, 2.353594052973287)', 'a') # exp mul

#4s
#fe_exp = gene('fe_exp', 0.5226260394007497)#important!
#fe_mul = gene('fe_mul', 1.849497255127404)#important!
ifi_plotter.add_plot('fe3(x, 0.5226260394007497, 1.849497255127404)', 'a') # exp mul

#abcde._ / 7
#fe_exp = gene('fe_exp', 0.38183660298411154)#important!
#fe_mul = gene('fe_mul', 1.1403125408175965)#important!
ifi_plotter.add_plot('fe3(x, 0.38183660298411154, 1.1403125408175965)', 'a') # exp mul

ifi_plotter.show()

