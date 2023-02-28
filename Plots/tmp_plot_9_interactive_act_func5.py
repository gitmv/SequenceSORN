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


#abcde._
#{'IP_s': '0.009467182335040847', 'avg_inh': '0.3396395276341545', 'LI_s': '10.905932034754143', 'STDP_s': '0.01957449620009728', 'fe_exp': '0.3699129832758579', 'fe_mul': '1.0851621679160406'}
#0.02040816326530612
#Experiment_Layer_default_v3_simplification_evo_old_fe_abcdep_3
#ifi_plotter.add_plot('fi4(x, 0.3396395276341545, 0.02040816326530612)', '0.02040816326530612') # slope target
#ifi_plotter.add_plot('fe3(x, 0.3699129832758579, 1.0851621679160406)', 'a') # exp mul
#0.674


#abcde._
#{'IP_s': 0.010215481922918927, 'avg_inh': 0.3641641301851917, 'LI_s': 13.050587955850654, 'STDP_s': 0.018916056500370957, 'fe_exp': 0.37568066889796553, 'fe_mul': 1.1038899038088466}
#0.02040816326530612
#Experiment_Layer_default_v3_simplification_evo_old_fe_abcdep_3
#ifi_plotter.add_plot('fi4(x, 0.3641641301851917, 0.02040816326530612)', '0.02040816326530612') # slope target
#ifi_plotter.add_plot('fe3(x, 0.37568066889796553, 1.1038899038088466)', 'a') # exp mul
#0.828


#abcde._
#{'IP_s': 0.008973931059651436, 'avg_inh': 0.3231724475180935, 'LI_s': 14.924004407474971, 'STDP_s': 0.018129208184866318, 'fe_exp': 0.38183660298411154, 'fe_mul': 1.1403125408175965}
#0.02040816326530612
#Experiment_Layer_default_v3_simplification_evo_old_fe_abcdep_3
#ifi_plotter.add_plot('fi4(x, 0.3231724475180935, 0.02040816326530612)', '0.02040816326530612') # slope target
#ifi_plotter.add_plot('fe3(x, 0.38183660298411154, 1.1403125408175965)', 'a') # exp mul
#0.555


#3s
#{'IP_s': '0.008735764741458582', 'avg_inh': '0.3427857658747104', 'LI_s': '6.450234496564654', 'STDP_s': '0.0030597477411211885', 'fe_exp': '0.7378726012049153', 'fe_mul': '2.353594052973287'}
#0.019230769230769232
#Experiment_Layer_default_v3_simplification_evo_old_fe
#ifi_plotter.add_plot('fi4(x, 0.3427857658747104, 0.019230769230769232)', '0.019230769230769232') # slope target
#ifi_plotter.add_plot('fe3(x, 0.7378726012049153, 2.353594052973287)', 'a') # exp mul
#0.151


#4s
#{'IP_s': '0.009263675397284607', 'avg_inh': '0.3755245533873526', 'LI_s': '9.137740897721274', 'STDP_s': '0.006146842011975385', 'fe_exp': '0.5226260394007497', 'fe_mul': '1.849497255127404'}
#0.014705882352941176
#Experiment_Layer_default_v3_simplification_evo_old_fe_4s
#ifi_plotter.add_plot('fi4(x, 0.3755245533873526, 0.014705882352941176)', '0.014705882352941176') # slope target
#ifi_plotter.add_plot('fe3(x, 0.5226260394007497, 1.849497255127404)', 'a') # exp mul
#0.283


#3s linear
#{'IP_s': '0.006036403614021368', 'avg_inh': '0.417554713547619', 'LI_s': '8.789963835163837', 'STDP_s': '0.0042143728380237295', 'fe_mul': '2.1500199617670104'}
#0.019230769230769232
#Experiment_Layer_default_v3_simplification_evo_new_fe
#ifi_plotter.add_plot('fi4(x, 0.417554713547619, 0.019230769230769232)', '0.019230769230769232') # slope target
#ifi_plotter.add_plot('fe4(x, 2.1500199617670104)', 'a') # mul
#0.195


####################################################################################################################################
#No Layer version
#grammar=['abcde. ']
#target_activity = 1.0 / len(''.join(grammar))
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#ifi_plotter.add_plot('f_i(x, '+str(inh_output_slope)+')', str(target_activity)) # ci
#ifi_plotter.add_plot('f_e(x, '+str(exc_output_exponent)+')', 'a', '0.5') # ce
#>1


#?????
grammar=['abcde. ']
target_activity = 1.0 / len(''.join(grammar)) / 8
print(target_activity)
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
ifi_plotter.add_plot('f_i(x, '+str(inh_output_slope)+')', str(target_activity)) # ci
ifi_plotter.add_plot('f_e(x, '+str(exc_output_exponent)+')', 'a', '0.5') # ce
#0.701-0.5=0.201

#?????
#grammar=['abcde. ']
#target_activity = 1.0 / len(''.join(grammar)) / 7
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#ifi_plotter.add_plot('f_i(x, '+str(inh_output_slope)+')', str(target_activity)) # ci
#ifi_plotter.add_plot('f_e(x, '+str(exc_output_exponent)+')', 'a', '0.5') # ce
#0.701-0.5=0.201

#?????
#grammar=['abcde. ']
#target_activity = 1.0 / len(''.join(grammar)) / 3 #0.0476
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#ifi_plotter.add_plot('f_i(x, '+str(inh_output_slope)+')', str(target_activity)) # ci
#ifi_plotter.add_plot('f_e(x, '+str(exc_output_exponent)+')', 'a', '0.5') # ce
#0.937-0.5=0.437


#grammar=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']
#target_activity = 1.0 / len(''.join(grammar))
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#ifi_plotter.add_plot('f_i(x, '+str(inh_output_slope)+')', str(target_activity)) # ci
#ifi_plotter.add_plot('f_e(x, '+str(exc_output_exponent)+')', 'a', '0.5') # ce
#0.703-0.5=0.203

ifi_plotter.show()

