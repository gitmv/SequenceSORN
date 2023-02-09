import matplotlib.pyplot as plt
import numpy as np
import random

# Import the libraries
from matplotlib import cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.widgets import Slider, Button


def nth_sqrt(x, n):
    return np.power(x, (1 / n))



def f_e(x, ce):
    return np.power(np.abs(x - 0.5) * 2, ce) * (x > 0.5)

def f_i(x, ci):
    return np.tanh(x * ci)



def f_e_derivative(x, ce):
    return 2 * ce * np.power(2 * x - 1, ce - 1)

def f_i_derivative(x, ci):
    return (4 * ci) / np.power(np.exp(-ci * x) + np.exp(ci * x), 2)



def f_e_inverted(y, ce):
    return 0.5 * (nth_sqrt(y, ce) + 1)

def f_i_inverted(y, ci):
    return np.arctanh(y)/ci

#test inverse functions
#print(f_i_inverted(f_i(0.3,23), 23))
#print(f_e_inverted(f_e(0.6, 0.72), 0.72))


#def get_tangential(x, px, py, slope):
#    return py + (x - px) * slope



def f_e_tangential(x, ce, ae_fix):
    py = f_e(ae_fix, ce)
    slope = f_e_derivative(ae_fix, ce)
    return py + (x - ae_fix) * slope

def f_i_tangential(x, ci, ai_fix):
    py = f_i(ai_fix, ci)
    slope = f_i_derivative(ai_fix, ci)
    return py + (x - ai_fix) * slope



def f_e_tangential_inverted(y, ce, ae_fix):#################################
    py = f_e(ae_fix, ce)
    slope = f_e_derivative(ae_fix, ce)
    return ae_fix + (y - py) / slope

def f_i_tangential_inverted(y, ci, ai_fix):
    py = f_i(ai_fix, ci)
    slope = f_i_derivative(ai_fix, ci)
    return ai_fix + (y - py) / slope



def get_ce(h):
    return 0.01 / h + 0.22

def get_ci(h):
    return 0.4 / h + 3.6



def D1_o_diff_zero(O_E, s, W_fac, ce, z):
    O_E_t = O_E

    #if type(O_E) in [float, np.float64]:
    #    O_E_t=O_E
    #    if (O_E_t * W_fac) < 0.5:
    #        O_E_t = (0.5)/W_fac
    #else:
    #    O_E_t = O_E.copy()
    #    test = (O_E_t * W_fac) < 0.5
    #    O_E_t[test] = (0.5)/W_fac

    O_I = -f_e_inverted(O_E_t, ce) + O_E_t*W_fac + s + z

    return O_I


def D2_o_diff_zero(O_E, ci):
    return f_i(O_E, ci)



def D1_a_diff_zero(A_E, ci, ce, s, z, W_fac):
    return f_i_inverted((-A_E+W_fac*f_e(A_E, ce)+s+z)/   1   , ci)

def D2_a_diff_zero(A_E, ce, ci):
    return 1   *   f_e(A_E, ce)



def D1_o_diff_zero_linear(O_E, ce, ci, s, z, W_fac, ae_fix):################################
    return (-f_e_tangential_inverted(O_E, ce, ae_fix)+W_fac*O_E+s+z)/   1

def D2_o_diff_zero_linear(O_E, ci, ai_fix):
    return f_i_tangential(1   *   O_E, ci, ai_fix)



def D1_a_diff_zero_linear(A_E, ce, ci, s, z, W_fac, ae_fix, ai_fix):
    return f_i_tangential_inverted((-A_E + W_fac * f_e_tangential(A_E, ce, ae_fix)+s+z)/   1, ci, ai_fix)

def D2_a_diff_zero_linear(A_E, ce, ci, ae_fix):
    return 1   *f_e_tangential(A_E, ce, ae_fix)



def get_sensitivity(O_E, ci, ce, W_fac):
    return f_i(O_E, ci)+ f_e_inverted(O_E, ce) - O_E*W_fac




def get_stability_matrix(ce, ci, w, s, oe, oi, z): #curved o
    #(a b)
    #(c d)
    a = -1 + 2 * ce * np.power(2*w*oe-1-2*oi+2*s+2*z, ce-1)
    b = 2 * ce * np.power(2*w*oe-2*oi+2*s+2*z, ce-1)
    c = (4 * ci) / np.power(np.power(np.e, -ci*oe)+np.power(np.e, ci*oe), 2)
    d = -1
    return a,b,c,d


def get_stability_matrix2(ce, ci, w, ae, ai):#curved a
    #(a b)
    #(c d)
    a = -1+2*w*ce*np.power(2*ae-1,ce-1)
    b = -(4*   1    *ci)/np.power(np.power(np.e, -ci*ai)+np.power(np.e, ci*ai),2)
    c = (2*   1   *ce*np.power(2*ae-1,ce-1))/3
    d = -1/3
    return a,b,c,d


def get_stability_matrix3(ce, ci, w, ae, ai):#linear o
    #(a b)
    #(c d)
    slope_e = f_e_derivative(ae, ce)
    slope_i = f_i_derivative(ae, ci)
    a = -1+w*slope_e
    b = -   1   *slope_e
    c = slope_i
    d = -1
    return a,b,c,d


def get_stability_matrix4(ce, ci, w, ae, ai):#linear a
    #(a b)
    #(c d)
    slope_e = f_e_derivative(ae, ce)
    slope_i = f_i_derivative(ai, ci)
    a = -1+w*slope_e
    b = -   1   *slope_i
    c = (1/3)*   1   * slope_e
    d = -(1/3)
    return a,b,c,d




def get_eigenvalues_2x2_abc(a, b, c, d):
    ev1r = 0.5 * (a+d) # before square root
    sq = np.power((-a-d),2) - 4*(a*d-b*c) # inside sqare root

    ev1i = 0.5 + np.sqrt(np.abs(sq))

    ev2r = ev1r
    ev2i = -ev1i

    if sq > 0:  # no imaginary part
        ev1r += ev1i
        ev1i = 0

        ev2r += ev2i
        ev2i = 0

    return ev1r, ev1i, ev2r, ev2i






fig, ax = plt.subplots(2,3)

#axfreq = plt.axes([0.25, 0.95, 0.5, 0.05])

plt.subplot(231)
plt.axis([0,1,0,1])
f_e_plot, = plt.plot([], [])
f_i_plot, = plt.plot([], [])
f_e_plot2, = plt.plot([], [])#optional tangential
f_i_plot2, = plt.plot([], [])#optional tangential
f_e_scatter = plt.scatter([], [])
f_i_scatter = plt.scatter([], [])
plt.xlabel('activity (voltage)')
plt.ylabel('spike chance')

plt.subplot(232)
plt.axis([0, 0.2, 0, 1])
d1_o_diff_zero_plot, = plt.plot([], [])
d2_o_diff_zero_plot, = plt.plot([], [])
d1_o_diff_zero_scatter = plt.scatter([], [])
d2_o_diff_zero_scatter = plt.scatter([], [])
plt.xlabel('o_E')
plt.ylabel('o_I')

plt.plot([], [], label='a')
plt.plot([], [], label='b')
leg_o = plt.legend()

plt.subplot(233)
plt.axis([-1, 1, -1, 1])#[-0.0, 1, -0.2, 0.2]
d1_a_diff_zero_plot, = plt.plot([], [])
d2_a_diff_zero_plot, = plt.plot([], [])
d1_a_diff_zero_scatter = plt.scatter([], [])
d2_a_diff_zero_scatter = plt.scatter([], [])
plt.xlabel('a_E')
plt.ylabel('a_I')

plt.plot([], [], label='a')
plt.plot([], [], label='b')
leg_a = plt.legend()


plt.subplot(234)
plt.axis([0,1,0,1])
f_e_linear_plot, = plt.plot([], [])
f_i_linear_plot, = plt.plot([], [])
f_e_linear_scatter = plt.scatter([], [])
f_i_linear_scatter = plt.scatter([], [])
plt.xlabel('activity (voltage)')
plt.ylabel('spike chance')

plt.subplot(235)
plt.axis([0, 0.2, 0, 1])
d1_o_linear_diff_zero_plot, = plt.plot([], [])
d2_o_linear_diff_zero_plot, = plt.plot([], [])
d1_o_linear_diff_zero_scatter = plt.scatter([], [])
d2_o_linear_diff_zero_scatter = plt.scatter([], [])
plt.xlabel('o_E')
plt.ylabel('o_I')

plt.plot([], [], label='a')
plt.plot([], [], label='b')
leg_o_linear = plt.legend()

plt.subplot(236)
plt.axis([-1, 1, -1, 1])#[-0.0, 1, -0.2, 0.2]
d1_a_linear_diff_zero_plot, = plt.plot([], [])
d2_a_linear_diff_zero_plot, = plt.plot([], [])
d1_a_linear_diff_zero_scatter = plt.scatter([], [])
d2_a_linear_diff_zero_scatter = plt.scatter([], [])
plt.xlabel('a_E')
plt.ylabel('a_I')

plt.plot([], [], label='a')
plt.plot([], [], label='b')
leg_a_linear = plt.legend()


a_slider = Slider(ax=plt.axes([0.06, 0.95, 0.1, 0.05]), label='sensitivity', valmin=-2.0, valmax=2.0, valinit=0.346)#0.56
b_slider = Slider(ax=plt.axes([0.3, 0.95, 0.1, 0.05]), label='w_ee', valmin=1.0, valmax=100.0, valinit=29.81)
c_slider = Slider(ax=plt.axes([0.5, 0.95, 0.1, 0.05]), label='h', valmin=0.002, valmax=0.2, valinit=0.02)
d_slider = Slider(ax=plt.axes([0.7, 0.95, 0.1, 0.05]), label='input', valmin=0.0, valmax=3.0, valinit=0.0)



def update(val):
       s = a_slider.val
       W_fac = b_slider.val
       h = c_slider.val
       z = d_slider.val

       b_slider.label.set_text('(w*{:.2f}={:.2f})  w'.format(h, W_fac*h))

       x = np.arange(0.0, 1.0, 0.0001)

       x_a = np.arange(-1, 1, 0.00001)

       ce=get_ce(h)
       ci=get_ci(h)

       if s == -2.0:
           s = get_sensitivity(h, ci, ce, W_fac)
           a_slider.label.set_text('({:.2f})  s'.format(s))
       else:
           a_slider.label.set_text('s'.format(s))

       o_e_fix = h
       o_i_fix = f_i(h, ci)

       a_e_fix = W_fac * h
       a_i_fix = h

       print(a_e_fix)


       #####################################

       f_e_plot.set_xdata(x)
       f_e_plot.set_ydata(f_e(x, ce))
       f_i_plot.set_xdata(x)
       f_i_plot.set_ydata(f_i(x, ci))
       f_e_scatter.set_offsets(np.c_[a_e_fix, f_e(a_e_fix, ce)])#f_e(a_e_fix, ce)
       f_i_scatter.set_offsets(np.c_[a_i_fix, f_i(a_i_fix, ci)])#f_i(a_i_fix, ci)

       #####################################

       d1_o_diff_zero_plot.set_xdata(x)
       d1_o_diff_zero_plot.set_ydata(D1_o_diff_zero(x, s, W_fac, ce, z))
       d2_o_diff_zero_plot.set_xdata(x)
       d2_o_diff_zero_plot.set_ydata(D2_o_diff_zero(x, ci))
       d1_o_diff_zero_scatter.set_offsets(np.c_[h, D1_o_diff_zero(h, s, W_fac, ce, z)])
       d2_o_diff_zero_scatter.set_offsets(np.c_[h, D2_o_diff_zero(h, ci)])

       a, b, c, d = get_stability_matrix(ce, ci, W_fac, s, h, f_i(h, ci), z)
       ev1r, ev1i, ev2r, ev2i = get_eigenvalues_2x2_abc(a, b, c, d)
       leg_o.texts[0].set_text('ev1: {:.2f} * {:.2f}i'.format(ev1r, ev1i))
       leg_o.texts[1].set_text('ev2: {:.2f} * {:.2f}i'.format(ev2r, ev2i))

       #####################################

       d1_a_diff_zero_plot.set_xdata(x_a)
       d1_a_diff_zero_plot.set_ydata(D1_a_diff_zero(x_a, ci, ce, s, z, W_fac))
       d2_a_diff_zero_plot.set_xdata(x_a)
       d2_a_diff_zero_plot.set_ydata(D2_a_diff_zero(x_a, ce, ci))
       d1_a_diff_zero_scatter.set_offsets(np.c_[a_e_fix, D1_a_diff_zero(a_e_fix, ci, ce, s, z, W_fac)])################
       d2_a_diff_zero_scatter.set_offsets(np.c_[a_i_fix, D2_a_diff_zero(a_i_fix, ce, ci)])

       a, b, c, d = get_stability_matrix2(ce, ci, W_fac, W_fac*h, h)
       ev1r, ev1i, ev2r, ev2i = get_eigenvalues_2x2_abc(a, b, c, d)
       leg_a.texts[0].set_text('ev1: {:.2f} * {:.2f}i'.format(ev1r, ev1i))
       leg_a.texts[1].set_text('ev2: {:.2f} * {:.2f}i'.format(ev2r, ev2i))

       #####################################

       f_e_linear_plot.set_xdata(x)
       f_e_linear_plot.set_ydata(f_e_tangential(x, ce, a_e_fix))
       f_i_linear_plot.set_xdata(x)
       f_i_linear_plot.set_ydata(f_i_tangential(x, ci, a_i_fix))
       f_e_linear_scatter.set_offsets(np.c_[a_e_fix, f_e(a_e_fix, ce)])
       f_i_linear_scatter.set_offsets(np.c_[h, f_i(h, ci)])

       #f_e_plot2.set_xdata(f_e_tangential_inverted(x, ce, a_e_fix))
       #f_e_plot2.set_ydata(x)
       #f_i_plot2.set_xdata(x)
       #f_i_plot2.set_ydata(f_e_tangential(x, ce, a_e_fix))

       f_e_plot2.set_xdata(x)
       f_e_plot2.set_ydata(f_e_tangential(x, ce, a_e_fix))
       f_i_plot2.set_xdata(x)
       f_i_plot2.set_ydata(f_i_tangential(x, ci, a_i_fix))

       #####################################

       d1_o_linear_diff_zero_plot.set_xdata(x)
       d1_o_linear_diff_zero_plot.set_ydata(D1_o_diff_zero_linear(x, ce, ci, s, z, W_fac, a_e_fix))###################
       d2_o_linear_diff_zero_plot.set_xdata(x)
       d2_o_linear_diff_zero_plot.set_ydata(D2_o_diff_zero_linear(x, ci, a_i_fix))
       print(h, ce, ci, s, z, W_fac, a_e_fix)
       d1_o_linear_diff_zero_scatter.set_offsets(np.c_[h, D1_o_diff_zero_linear(h, ce, ci, s, z, W_fac, a_e_fix)])###################
       d2_o_linear_diff_zero_scatter.set_offsets(np.c_[h, D2_o_diff_zero_linear(h, ci, a_i_fix)])
       #d1_o_linear_diff_zero_scatter.set_offsets(np.c_[h, D1_o_diff_zero(h, s, W_fac, ce, z)])
       #d2_o_linear_diff_zero_scatter.set_offsets(np.c_[h, D2_o_diff_zero(h, ci)])

       a, b, c, d = get_stability_matrix3(ce, ci, W_fac, a_e_fix, a_i_fix)
       ev1r, ev1i, ev2r, ev2i = get_eigenvalues_2x2_abc(a, b, c, d)
       leg_o_linear.texts[0].set_text('ev1: {:.2f} * {:.2f}i'.format(ev1r, ev1i))
       leg_o_linear.texts[1].set_text('ev2: {:.2f} * {:.2f}i'.format(ev2r, ev2i))

       #####################################

       d1_a_linear_diff_zero_plot.set_xdata(x_a)
       d1_a_linear_diff_zero_plot.set_ydata(D1_a_diff_zero_linear(x_a, ce, ci, s, z, W_fac, a_e_fix, a_i_fix))
       d2_a_linear_diff_zero_plot.set_xdata(x_a)
       d2_a_linear_diff_zero_plot.set_ydata(D2_a_diff_zero_linear(x_a, ce, ci, a_e_fix))
       d1_a_linear_diff_zero_scatter.set_offsets(np.c_[a_e_fix, D1_a_diff_zero_linear(a_e_fix, ce, ci, s, z, W_fac, a_e_fix, a_i_fix)])######################
       d2_a_linear_diff_zero_scatter.set_offsets(np.c_[a_e_fix, D2_a_diff_zero_linear(a_e_fix, ce, ci, a_e_fix)])

       a, b, c, d = get_stability_matrix4(ce, ci, W_fac, a_e_fix, a_i_fix)
       ev1r, ev1i, ev2r, ev2i = get_eigenvalues_2x2_abc(a, b, c, d)
       leg_a_linear.texts[0].set_text('ev1: {:.2f} * {:.2f}i'.format(ev1r, ev1i))
       leg_a_linear.texts[1].set_text('ev2: {:.2f} * {:.2f}i'.format(ev2r, ev2i))

       #####################################

       fig.canvas.draw_idle()

a_slider.on_changed(update)
b_slider.on_changed(update)
c_slider.on_changed(update)
d_slider.on_changed(update)

update(0)
plt.show()



#def get_eigenvalues(ce, ci, w, s, oe, oi, z):
#    ev1r=-1+ce*w*np.power(2*w*oe-1-2*oe+2*s+2*z, ce-1)

#    sq = np.power(2*ce*w*np.power(2+w*oe-1-2*oi+2*s+2*z, ce-1),2)+(32*ce*ci*(2*w*oe-2*oi+2*s+2*z))/np.power(np.power(np.e,-ci*oe)+np.power(np.e,ci*oe),2)
#    ev1i = 0.5 * np.sqrt(np.abs(sq)) #0.5????

#    ev2r = ev1r
#    ev2i = -ev1i

#    if sq>0:#no imaginary part
#        ev1r+=ev1i
#        ev1i=0

#        ev2r+=ev2i
#        ev2i=0

#    return ev1r, ev1i, ev2r, ev2i


#def get_eigenvalues_2x2_pq(a,b,c,d):
#    ev1r = -0.5 * (-a-d) # before square root
#    sq = np.power(0.5*(-a-d),2) - (a*d-b*c) # inside sqare root

#    ev1i = np.sqrt(np.abs(sq))

#    ev2r = ev1r
#    ev2i = -ev1i

#    if sq > 0:  # no imaginary part
#        ev1r += ev1i
#        ev1i = 0

#        ev2r += ev2i
#        ev2i = 0

#    return ev1r, ev1i, ev2r, ev2i


'''
x = np.arange(0.0, 0.04, 0.0001)

plt.plot(x, f_i(x, 17.6))
plt.scatter([0.02], [f_i(0.02, 17.6)])

s = 0.2

plt.plot(x, f_e_inverted(x, 17.6) - x - s)
plt.scatter([0.02], [f_e_inverted(0.02, 17.6) - 0.02 - s])
plt.show()
'''

'''
fig, ax = plt.subplots(1)


plt.axis([0, 0.2, 0, 1])

#axfreq = plt.axes([0.25, 0.95, 0.5, 0.05])

#plt.axis([-0.6,0.6,-0.6,0.6])
e_plot_line, = plt.plot([],[])
e_scatter = plt.scatter([0], [0])

i_plot_line, = plt.plot([],[])
i_scatter = plt.scatter([0], [0])

plt.xlabel('o_E')
plt.ylabel('o_I')
'''

'''
if type(O_I) is np.float64:
    if (O_E * W_fac - s) < 0.5:
        O_I = -(f_e_inverted(O_E, ce) + 0.5)
else:
    test = (O_E * W_fac - s) < 0.5
    O_I[test] = -(f_e_inverted(O_E, ce) + 0.5)[test]
'''