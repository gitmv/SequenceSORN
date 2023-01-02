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



sm_source = StorageManager('Text Learning Network', 'Text Learning Network6840', add_new_when_exists=False)
sm_target = StorageManager('test_sm')

for i in range(300+150):

    if i<300:
        phase="training"
    else:
        phase="free"

    fig=plt.figure(figsize=(20,13))
    fig.suptitle(str((i+1)*100)+" timesteps ("+phase+")")

    plt.subplot(236)
    i_activity = sm_source.load_np('inh_n._activity_'+str(i))
    plt.xlim((0, 0.1))
    plt.ylim((0, 400))
    plt.xlabel('inhibitory neuron activity (voltage)')
    plt.ylabel('#neurons')
    plt.hist(i_activity.flatten(), bins=500)
    x = np.arange(0, 0.1, 0.001)
    plt.plot(x, np.tanh(x * 24.4)*400)

    plt.subplot(234)
    e_activity = sm_source.load_np('exc_n._activity_'+str(i))
    plt.xlim((-1.0, 1.5))
    plt.ylim((0, 2000))
    plt.xlabel('excitatory neuron activity (voltage)')
    plt.ylabel('#neurons')
    plt.hist(e_activity.flatten(), bins=500)
    x = np.arange(0, 1, 0.001)
    plt.plot(x, np.power(np.abs(x - 0.5) * 2, 0.74) * (x > 0.5) * 2000)

    plt.subplot(235)
    e_glu = sm_source.load_np('exc_n.input_GLU_'+str(i))
    plt.xlim((0.0, 1.2))
    plt.ylim((0, 2000))
    plt.xlabel('excitatory neuron GLU input')
    plt.ylabel('#neurons')
    plt.hist(e_glu.flatten(), bins=500)

    plt.subplot(232)
    EE = sm_source.load_np('EE_'+str(i))
    plt.xlim((0.0, 0.03))
    plt.ylim((0, 4000))
    plt.xlabel('EE synapse weights')
    plt.ylabel('#synapses')
    plt.hist(EE.flatten(), bins=500)

    plt.subplot(231)
    ea = np.mean(sm_source.load_np('exc_n._activity_'+str(i)), axis=1)
    plt.xlim((-1.0, 1.5))
    plt.ylim((-1.0, 1.5))
    plt.xlabel('mean excitatory neuron activity (t)')
    plt.ylabel('mean excitatory neuron activity (t+1)')
    p(ea[:-1], ea[1:])

    plt.subplot(233)
    ia = np.mean(sm_source.load_np('inh_n._activity_'+str(i)), axis=1)
    plt.xlim((0, 0.1))
    plt.ylim((0, 0.1))
    plt.xlabel('mean inhibitory neuron activity (t)')
    plt.ylabel('mean inhibitory neuron activity (t+1)')
    p(ia[:-1], ia[1:])


    plt.savefig(sm_target.get_next_frame_name('example'))
    plt.close()

sm_target.render_video('example', True)