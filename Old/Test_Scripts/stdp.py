import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

X = np.arange(-3, 3, 0.001)

def stdp(x,g):
    ca2 = np.power(np.e, -np.power((5*np.log(-x+2)*0.2),2))*(1-g)
    ltp = np.tanh(ca2)
    ltd = np.tanh(ca2*6)/3
    return ltp-ltd

for g in range(5):#np.arange(0, 0.9, 0.1):
    g=g/5
    Y=stdp(X,g)
    Y[X>2]=0
    plt.plot(X,Y, label=g)

plt.axvline(0, color='gray')
plt.axhline(0, color='gray')

plt.xlabel('$\Delta$ t')
plt.ylabel('$\Delta$ s')
plt.suptitle('STDP modulation', size=16)

leg=plt.legend()
leg.set_title('Learning Inhibition/\ninhibitory input')
plt.show()











fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-3, 2, 0.001)
G = np.arange(0,0.9,0.1)
X, G = np.meshgrid(X, G)
Z = stdp(X, G)

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


plt.show()
