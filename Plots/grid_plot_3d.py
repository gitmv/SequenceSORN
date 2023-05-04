from PymoNNto import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pq
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

app = QtWidgets.QApplication([])
w = gl.GLViewWidget()



g = gl.GLGridItem()
w.addItem(g)

x_scale = 0.2
y_scale = 1
z_scale = 1

def add_datapoints(names, xp, yp, zp):
    x = np.array([])
    y = np.array([])
    z = np.array([])
    s = np.array([])

    for name in names:
        xt, yt, zt, st = load_datapoints([xp, yp, zp, 'score'], name)
        x = np.concatenate([x, xt])
        y = np.concatenate([y, yt])
        z = np.concatenate([z, zt])
        s = np.concatenate([s, st])

    mins=np.min(s)
    maxs=np.max(s)

    mask = s>5
    x = x[mask]
    y = y[mask]
    z = z[mask]
    s = s[mask]

    print(len(x),len(y),len(z),len(s))

    minx=np.min(x)
    maxx=np.max(x)

    miny=np.min(y)
    maxy=np.max(y)

    minz=np.min(z)
    maxz=np.max(z)



    pos = np.zeros((len(x),3), dtype=np.float32)
    pos[:,0]=(x/maxx-0.5)*5
    pos[:,1]=(y/maxy-0.5)*5
    pos[:,2]=(z/maxz-0.5)*5

    color = np.zeros((len(x),4), dtype=np.float32)
    v=(s-mins)/(maxs-mins)
    color[:,0] = 1-v
    color[:,1] = v
    color[:,2] = 0.0
    color[:,3] = 1.0

    sp2 = gl.GLScatterPlotItem(pos=pos, color=color)
    w.addItem(sp2)

#v5_mul_stdp_scatter_3s_3sh_extended2
#v5_ip_li_scatter_3s_3sh
#v5_exp_stdp_scatter_3s_3sh
#v5_mul_stdp_scatter_3s_3sh


#add_datapoints(['v5_exp_mul_scatter_2',
#                'v5_mul_inh_scatter_3s_3sh',
#                'v5_mul_exc_scatter_3s_3sh_05inh',
#                'v5_exp_mul_scatter_3s_3sh_linear_genome'], 'fe_mul', 'fe_exp', 'avg_inh')#v5_exp_mul_scatter_2s_3sh

add_datapoints(['v5_exp_mul_h_scatter_3s_4'], 'fe_mul', 'fe_exp', 'h')

## set OpenGL blend function
w.opts['glBlendFunc'] = (0x0302, 0x0303)
w.show()



if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()