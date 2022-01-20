from PymoNNto import *
import matplotlib.pyplot as plt

smg = StorageManagerGroup('synapse_d')
x=[]
y=[]
for sm in smg.StorageManagerList:
    #print(sm.load_param('sd'), sm.load_param('score'), sm.load_param('text'))
    x.append(sm.load_param('sd'))
    y.append(sm.load_param('score'))

plt.scatter(x,y)
plt.show()
