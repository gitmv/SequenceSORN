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


#'C:/Users/DELL/Programmieren/Python_Modular_Neural_Network_Toolbox/SequenceSORN/Data/StorageManager/Text Learning Network/Text Learning Network2099/'
sm = StorageManager('Text Learning Network', 'Text Learning Network5470', add_new_when_exists=False)

#short:
#Text Learning Network3602

#long:
#Text Learning Network5470
#Text Learning Network3832

e_glu = np.mean(sm.load_np('exc_n.input_GLU'), axis=1)#[0:7]
e_gaba = np.mean(sm.load_np('exc_n.input_GABA'), axis=1)#[0:7]

p(e_glu, e_gaba)

plt.show()



eo = sm.load_np('exc_n.output')
io = sm.load_np('inh_n.output')
#plt.plot(eo, io, alpha=0.1)
#plt.scatter(eo, io)
#plt.show()

e_s = sm.load_np('exc_n.sensitivity')
e_glu = sm.load_np('exc_n.input_GLU')
e_gaba = sm.load_np('exc_n.input_GABA')
#plt.hist(e_gaba.flatten()+e_glu.flatten()+e_s.flatten(), bins=1000)
#p(e_gaba[:-1].flatten()+e_glu[:-1].flatten()+e_s[:-1].flatten(), e_gaba[1:].flatten()+e_glu[1:].flatten()+e_s[1:].flatten(), a=0.01, s=0.1, plot=False)
#plt.plot([0],[0])

#i=0
#p(e_gaba[:-1,i]+e_glu[:-1,i]+e_s[:-1,i], e_gaba[1:,i]+e_glu[1:,i]+e_s[1:,i])

#p(e_glu[:-1].flatten()+e_s[:-1].flatten(), e_glu[1:].flatten()+e_s[1:].flatten())
#p(e_s[:-1].flatten(), e_s[1:].flatten())
#p(e_glu[:-1].flatten(), e_glu[1:].flatten())
p(e_glu.flatten(), -e_gaba.flatten(), a=0.05, s=0.1, plot=False)

#p(eo.flatten(), io.flatten(), a=0.05, s=0.1, plot=False)
#p([],[])
plt.plot([0],[0])

#i=0
#p(e_glu[:,i], e_gaba[:,i])
#p(eo[:,i], io[:,i])


eo = np.mean(sm.load_np('exc_n.output'), axis=1)
io = np.mean(sm.load_np('inh_n.output'), axis=1)

e_s = np.mean(sm.load_np('exc_n.sensitivity'), axis=1)#[0:7]
e_glu = np.mean(sm.load_np('exc_n.input_GLU'), axis=1)#[0:7]
e_gaba = np.mean(sm.load_np('exc_n.input_GABA'), axis=1)#[0:7]
#plt.plot(e_glu, -e_gaba+e_glu, alpha=0.1)
#plt.scatter(e_glu, -e_gaba+e_glu)


#l=len(e_gaba)
#c=[[1/l*i,0,1-1/l*i] for i in range(l)]
#m=np.max(e_glu)
#c=[[e_glu[i]/m,0,0] for i in range(l)] #c=c[1:]

#p(e_s[:-1], +e_s[:-1])
#p(e_gaba[:-1]+e_glu[:-1]+e_s[:-1], e_gaba[1:]+e_glu[1:]+e_s[1:])
#p((-e_gaba+e_glu)[:-1], (-e_gaba+e_glu)[1:])#, c=c[:-1]
#p(e_glu[:-1]+e_s[:-1], e_glu[1:]+e_s[1:])#, c=c[:-1]
#p(e_gaba[:-1], e_gaba[1:])#, c=c[:-1]

p(e_glu, -e_gaba)
#p(eo, io)

#e_gaba = sm.load_np('exc_n.input_GABA')
#p((e_glu)[1:].flatten(), (e_glu)[:-1].flatten())
#plt.hist(e_glu.flatten(), bins=1000)

#p((-e_gaba)[1:], (-e_gaba)[:-1])

#p(e_glu, e_gaba)

plt.show()

#

#plt.show()

#i_glu = sm.load_np('inh_n.input_GLU')
#print(i_glu.shape)

#plt.scatter(np.mean(e_glu, axis=1)[1:], np.mean(eo, axis=1)[:-1])
#plt.show()


#EE = sm.load_np('EE')
#plt.hist(EE.flatten(), bins=1000)
#plt.show()


'''
sm = StorageManager('Text Learning Network', 'Text Learning Network3602', add_new_when_exists=False)
e_glu = np.mean(sm.load_np('exc_n.input_GLU'), axis=1)
e_gaba = -np.mean(sm.load_np('exc_n.input_GABA'), axis=1)
eo = np.mean(sm.load_np('exc_n.output'), axis=1)
io = np.mean(sm.load_np('inh_n.output'), axis=1)
p(eo, io)

sm = StorageManager('Text Learning Network', 'Text Learning Network5470', add_new_when_exists=False)
e_glu = np.mean(sm.load_np('exc_n.input_GLU'), axis=1)
e_gaba = -np.mean(sm.load_np('exc_n.input_GABA'), axis=1)
eo = np.mean(sm.load_np('exc_n.output'), axis=1)
io = np.mean(sm.load_np('inh_n.output'), axis=1)
p(eo, io)

sm = StorageManager('Text Learning Network', 'Text Learning Network3832', add_new_when_exists=False)
e_glu = np.mean(sm.load_np('exc_n.input_GLU'), axis=1)
e_gaba = -np.mean(sm.load_np('exc_n.input_GABA'), axis=1)
eo = np.mean(sm.load_np('exc_n.output'), axis=1)
io = np.mean(sm.load_np('inh_n.output'), axis=1)
p(eo, io)

plt.show()
'''