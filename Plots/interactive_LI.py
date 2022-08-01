import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import cm

def sig(x, a, b, c, d):
    return 1/(1+np.power(np.e, -c*(x-a)))*d+b

t = np.arange(-3, 2, 0.001)

fig, ax = plt.subplots()
sig_line, = plt.plot(t, sig(t, 0, 0, 0, 0), label='sigmoid')

#axfreq = plt.axes([0.25, 0.95, 0.5, 0.05])

a_slider = Slider(ax=plt.axes([0.06, 0.95, 0.1, 0.05]), label='a', valmin=-1, valmax=2, valinit=0)
b_slider = Slider(ax=plt.axes([0.3, 0.95, 0.1, 0.05]), label='b', valmin=-1, valmax=2, valinit=0)
c_slider = Slider(ax=plt.axes([0.5, 0.95, 0.1, 0.05]), label='c', valmin=-1, valmax=2, valinit=0)
d_slider = Slider(ax=plt.axes([0.84, 0.95, 0.1, 0.05]), label='d', valmin=-1, valmax=2, valinit=0)

def update(val):
    a = a_slider.val
    b = b_slider.val
    c = c_slider.val
    d = d_slider.val
    sig_line.set_ydata(sig(t, a, b, c, d))
    fig.canvas.draw_idle()

a_slider.on_changed(update)
b_slider.on_changed(update)
c_slider.on_changed(update)
d_slider.on_changed(update)

#ax.set_xlabel('$\Delta$ t')
#ax.set_ylabel('$\Delta$ s')

ax.legend()

plt.show()




