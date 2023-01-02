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
sm_target = StorageManager('test_sm')

for i in range(45000):

    if i<30000:
        phase="training"
    else:
        phase="free"

    txt = sm_source.load_param('txt_'+str(i))

    fig=plt.figure(figsize=(20,8))
    fig.suptitle(str(i+1)+" timesteps ("+phase+")\r\n"+txt)

    plt.subplot(133)
    i_activity = sm_source.load_np('inh_n._activity_'+str(i))
    print(len(i_activity.flatten()))
    plt.xlim((0, 0.1))
    plt.ylim((0, 100))
    plt.xlabel('inhibitory neuron activity (voltage)')
    plt.ylabel('#neurons')
    plt.hist(i_activity.flatten(), bins=50)
    x = np.arange(0, 0.1, 0.001)
    plt.plot(x, np.tanh(x * 24.4)*100)

    plt.subplot(131)
    e_activity = sm_source.load_np('exc_n._activity_'+str(i))
    plt.xlim((-1.0, 1.5))
    plt.ylim((0, 200))
    plt.xlabel('excitatory neuron activity (voltage)')
    plt.ylabel('#neurons')
    plt.hist(e_activity.flatten(), bins=50)
    x = np.arange(0, 1, 0.001)
    plt.plot(x, np.power(np.abs(x - 0.5) * 2, 0.74) * (x > 0.5) * 200)

    plt.subplot(132)
    e_glu = sm_source.load_np('exc_n.input_GLU_'+str(i))
    plt.xlim((0.0, 1.2))
    plt.ylim((0, 200))
    plt.xlabel('excitatory neuron GLU input')
    plt.ylabel('#neurons')
    plt.hist(e_glu.flatten(), bins=50)

    plt.savefig(sm_target.get_next_frame_name('example'))
    plt.close()

sm_target.render_video('example', True)