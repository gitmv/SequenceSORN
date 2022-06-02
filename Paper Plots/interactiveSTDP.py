import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import cm

def ca(x, g):
    return np.power(np.e, -np.power((5 * np.log(-x + 2) * 0.2), 2)) * (1 - g)

def ltp(ca2):
    return np.tanh(ca2)

def ltd(ca2):
    return np.tanh(ca2*6)/3

def stdp(ca2):
    ltp = np.tanh(ca2)
    ltd = np.tanh(ca2*6)/3
    return ltp-ltd

t = np.arange(-3, 2, 0.001)

fig, ax = plt.subplots()
ca_line, = plt.plot(t, ca(t, 0), label='ca2+')
ltp_line, = plt.plot(t, ltp(ca(t,0)), label='LTP')
ltd_line, = plt.plot(t, ltd(ca(t,0)), label='LTD')
stdp_line, = plt.plot(t, stdp(ca(t,0)), label='STDP')

axfreq = plt.axes([0.25, 0.95, 0.5, 0.05])
freq_slider = Slider(ax=axfreq, label='GABA input', valmin=-1, valmax=2, valinit=0)

def update(val):
    g=freq_slider.val
    ca_line.set_ydata(ca(t, g))
    ltp_line.set_ydata(ltp(ca(t,g)))
    ltd_line.set_ydata(ltd(ca(t,g)))
    stdp_line.set_ydata(stdp(ca(t,g)))
    fig.canvas.draw_idle()

freq_slider.on_changed(update)

ax.set_xlabel('$\Delta$ t')
ax.set_ylabel('$\Delta$ s')

ax.legend()

plt.show()




t = np.arange(-3, 2, 0.001)

fig, ax = plt.subplots()

for i in [0.0,0.2,0.4,0.6]:
    plt.bar([1], height=stdp(ca(1, i)), width=1-i/2)

for i in [0.0,0.2,0.4,0.6]:
    ca_line, = plt.plot(t, stdp(ca(t, i)), label=str(i))

leg=plt.legend()
plt.suptitle('STDP modulation', size=16)
leg.set_title('inhibition')
ax.set_xlabel('$\Delta$ t')
ax.set_ylabel('$\Delta$ s')

ax.legend()

plt.show()





fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-3, 2, 0.001)
G = np.arange(0,0.9,0.1)
X, G = np.meshgrid(X, G)
Z = stdp(ca(X, G))

# Plot the surface.
surf = ax.plot_surface(X, G, Z, cmap=cm.RdYlGn, linewidth=0, antialiased=False, vmin=-0.2, vmax=0.5)


# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

#ax.set_zlim(-1.01, 1.01)

# Add a color bar which maps values to colors.
#surf.clim(-1, 1)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('$\Delta$ t')
ax.set_ylabel('GABA')
ax.set_zlabel('$\Delta$ s')

plt.show()