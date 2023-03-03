from PymoNNto import *
import matplotlib.pyplot as plt
import numpy as np
import random

#Evolution_Project_Clones
def get_datapoints(xparam, yparam, project_name='test', plot_evo_folder='Plot_Project_Clones'):
    folder = get_data_folder() + '/'+plot_evo_folder+'/' + project_name + '/Data'
    smg = StorageManagerGroup(project_name, data_folder=folder)
    smg.sort_by(xparam)
    data = smg.get_multi_param_dict(params=[xparam, yparam] ,remove_None=True)
    return data[xparam], data[yparam]

x,y=get_datapoints('t', 'score', 'Layer_default_v3_target_activity_compensation')
plt.scatter(x,y, c=(1,0,0,1))

x,y=get_datapoints('t', 'score', 'Layer_v4_3s_h_vs_p')
plt.scatter(x,y, c=(1,0,0,1))

x,y=get_datapoints('t', 'score', 'Layer_default_v3_target_activity')
plt.scatter(x,y, c=(1,0,0,1))

'''
x,y=get_datapoints('t', 'score', 'Layer_v4_linear_abc_h_vs_p')
plt.scatter(x,y, c=(0,0,1,1))
'''

def sweetspots(current_h, best_h):
    return np.cos(((current_h-best_h)*np.pi*2)/best_h)/2.0+0.5

x = np.arange(0, 0.1, 0.0001)
#plt.plot(x, (np.cos((x-0.02)*100)/0.5+0.5)*7)
#plt.plot(x,np.cos(np.power(1.1,x)*10000))
plt.plot(x, sweetspots(x, 0.02)*7.0)

'''
plt.axvline(1/7/10, c=(0,0,1,1))
plt.axvline(1/7/9, c=(0,0,1,1))
plt.axvline(1/7/8, c=(0,0,1,1))
plt.axvline(1/7/7, c=(0,0,1,1))
plt.axvline(1/7/6, c=(0,0,1,1))
plt.axvline(1/7/5, c=(0,0,1,1))
plt.axvline(1/7/4, c=(0,0,1,1))
plt.axvline(1/7/3, c=(0,0,1,1))
plt.axvline(1/7/2, c=(0,0,1,1))
plt.axvline(1/7/1, c=(0,0,1,1))
'''

plt.axvline(1/52/4, c=(1,0,0,1))
plt.axvline(1/52/3, c=(1,0,0,1))
plt.axvline(1/52/2, c=(1,0,0,1))
plt.axvline(1/52/1, c=(1,0,0,1))
plt.axvline(1/52/0.5, c=(1,0,0,1))
plt.axvline(1/52/0.25, c=(1,0,0,1))


def nth_sqrt(x, n):
    return np.power(x, (1 / n))

def steps(start, end, points):
    result = []
    if start <= 0 or end<=start:
        return []

    exp = nth_sqrt(end/start, points)

    while start<end:
        result.append(start)
        start *= exp
    result.append(end)
    return result





#print(steps(0.05, 1, 50))

#for s in steps(0.001, 1, 50):
#    plt.axvline(s, c=(1, 0, 0, 1))

#plt.axvline(1/23/4, c=(0,1,0,1))
#plt.axvline(1/23/3, c=(0,1,0,1))
#plt.axvline(1/23/2, c=(0,1,0,1))
#plt.axvline(1/23/1, c=(0,1,0,1))

plt.show()
