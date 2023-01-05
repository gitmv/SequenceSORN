import matplotlib.pyplot as plt
import numpy as np
import random

# Import the libraries
from matplotlib import cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D
from PymoNNto import *

def p(x,y,c=None,a=None, s=None, plot=True):
    if plot:
        plt.plot(x, y, alpha=0.1, c=c)
    plt.scatter(x, y, alpha=a, c=c, s=s)

# only 1000 free Network7736
#full length Network2126

sm_source = StorageManager('Text Learning Network', 'Text Learning Network2126', add_new_when_exists=False)
#sm_target = StorageManager('test_sm')

x = []
y1 = []
y2 = []
y3 = []

for i in range(45000):

    e_activity = sm_source.load_np('exc_n._activity_' + str(i))
    #i_activity = sm_source.load_np('inh_n._activity_'+str(i))

    e_glu = sm_source.load_np('exc_n.input_GLU_' + str(i))
    e_gaba = sm_source.load_np('exc_n.input_GABA_' + str(i))

    temp=e_activity[e_activity>0.5]
    if len(temp)==0:
        temp = [-1]

    x.append(np.mean(temp))

    meam_eglu = np.mean(e_glu)
    meam_egaba = np.mean(e_gaba)

    y1.append(meam_eglu)
    y2.append(meam_egaba)

    if meam_eglu != 0:
        y3.append(meam_egaba/meam_eglu)
    else:
        y3.append(0)

plt.scatter(x, y1)
plt.scatter(x, y2)
plt.scatter(x, y3)

plt.show()