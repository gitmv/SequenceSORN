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
       return np.power(np.abs(x - 0.5) * 2, ce) * (x > 0.5)

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


fig, ax = plt.subplots(1,2)

#axfreq = plt.axes([0.25, 0.95, 0.5, 0.05])


plt.subplot(122)
plt.axis([-0.6,0.6,-0.6,0.6])
e_plot_line, = plt.plot([],[])
i_plot_line, = plt.plot([],[])
i_plot_line_new, = plt.plot([],[])
plt.scatter([0], [0])

plt.subplot(121)
plt.axis([0,1,0,1])
e_plot_line2, = plt.plot([],[])
i_plot_line2, = plt.plot([],[])
i_plot_line2_new, = plt.plot([],[])

e_scatter = plt.scatter([], [])
i_scatter = plt.scatter([], [])

plt.xlabel('activity (voltage)')
plt.ylabel('spike chance')

a_slider = Slider(ax=plt.axes([0.06, 0.95, 0.1, 0.05]), label='h', valmin=0.002, valmax=0.2, valinit=0.02)
b_slider = Slider(ax=plt.axes([0.3, 0.95, 0.1, 0.05]), label='avg(a_E)', valmin=0.5, valmax=1.0, valinit=0.6)
#c_slider = Slider(ax=plt.axes([0.5, 0.95, 0.1, 0.05]), label='I_scale', valmin=1, valmax=25, valinit=1)
#d_slider = Slider(ax=plt.axes([0.84, 0.95, 0.1, 0.05]), label='d', valmin=-1, valmax=2, valinit=0)




def update(val):
       a = a_slider.val
       b = b_slider.val
       #c = c_slider.val
       #d = d_slider.val

       h = a

       print(get_ce(h), get_ci(h))


       x = np.arange(0, 1, 0.0001)
       ce=get_ce(h)#0.72
       ci=get_ci(h)#23.6


       #e_h_pos_y = b
       #e_h_pos_x = f_e_inverted(e_h_pos_y, ce)

       e_h_pos_x = b# + 0.5
       e_h_pos_y = f_e(e_h_pos_x, ce)

       e_x = x - e_h_pos_x
       e_y = f_e(x, ce) - e_h_pos_y

       e_plot_line.set_xdata(e_x)
       e_plot_line.set_ydata(e_y)



       i_h_pos_x = h
       i_h_pos_y = f_i(i_h_pos_x, ci)

       i_scale_x = get_FI_scale_factor(aI_fix=i_h_pos_x, aE_fix=e_h_pos_x, aI_zero=0, aE_zero=0.5)


       i_x = (x - i_h_pos_x) * i_scale_x
       i_y = (f_i(x, ci) - i_h_pos_y) #* c

       i_plot_line.set_xdata(i_x)
       i_plot_line.set_ydata(i_y)


       i_x_new = (x - i_h_pos_x) * i_scale_x
       i_y_new = (fi_2(x, ci, i_h_pos_x) - i_h_pos_y) #* c

       #i_plot_line_new.set_xdata(i_x_new)
       #i_plot_line_new.set_ydata(i_y_new)



       e_plot_line2.set_xdata(x)
       e_plot_line2.set_ydata(f_e(x, ce))

       i_plot_line2.set_xdata(x)
       i_plot_line2.set_ydata(f_i(x, ci))

       #i_plot_line2_new.set_xdata(x)
       #i_plot_line2_new.set_ydata(fi_2(x, ci, i_h_pos_x))


       e_scatter.set_offsets(np.c_[e_h_pos_x, e_h_pos_y])
       i_scatter.set_offsets(np.c_[i_h_pos_x, i_h_pos_y])

       fig.canvas.draw_idle()



a_slider.on_changed(update)
b_slider.on_changed(update)
#c_slider.on_changed(update)
#d_slider.on_changed(update)

#ax.set_xlabel('$\Delta$ t')
#ax.set_ylabel('$\Delta$ s')

#ax.legend()
update(0)
plt.show()








