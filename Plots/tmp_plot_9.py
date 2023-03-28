from PymoNNto import *
import matplotlib.pyplot as plt
import numpy as np
import random

import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

# create the function we want to fit
def my_sin(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset

def fit_my_sin_2(x, y):

    guess_freq = 10.0
    guess_amplitude = 2.0
    guess_phase = 0.0
    guess_offset = 0

    p0 = [guess_freq, guess_amplitude, guess_phase, guess_offset]

    def get_error(p):
        y2 = my_sin(x, *p)
        return np.mean(np.abs(y-y2))

    def get_range(p):
        return np.arange(p/3, p*3, p/3)

    p_found = p0
    min_error = get_error(p_found)
    for freq in get_range(guess_freq):
        print(freq)
        for amplitude in get_range(guess_amplitude):
            for phase in np.arange(guess_phase-5, guess_phase+5, 0.2):
                for offset in np.arange(guess_offset-5, guess_offset+5, 0.2):
                    p = [freq, amplitude, phase, offset]
                    err = get_error(p)
                    if err<min_error:
                        min_error = err
                        p_found=p.copy()
    print(p_found)
    return p_found


def fit_my_sin(x, y):

    guess_freq = 3
    guess_amplitude = 3*np.std(data)/(2**0.5)
    guess_phase = 0
    guess_offset = np.mean(data)



    # now do the fit
    fit = curve_fit(my_sin, x, y, p0=p0)

    return fit #my_sin(t, *fit[0])

    # we'll use this to plot our first estimate. This might already be good enough for you
    #data_first_guess = my_sin(t, *p0)

    # recreate the fitted curve using the optimized parameters
    #data_fit = my_sin(t, *fit[0])

#plt.plot(data, '.')
#plt.plot(data_fit, label='after fitting')
#plt.plot(data_first_guess, label='first guess')
#plt.legend()
#plt.show()



def draw_poly_fit(x,y):
    xmi=np.min(x)
    xma=np.max(x)
    ymi=np.min(y)
    yma=np.max(y)

    x = np.log(x)
    pf = np.poly1d(np.polyfit(x, y, 50))
    x = np.arange(xmi, xma, 0.00001)
    #np.power(2, x)
    plt.plot(x, np.clip(pf(np.log(x)),ymi,yma))

def sort_xy(x,y):
    x = np.array(x)
    y = np.array(y)
    ind = np.argsort(x)
    return x[ind],y[ind]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth_xy(x,y):
    x,y = sort_xy(x,y)
    for _ in range(10):
        y = smooth(y, 11)
    return x,y

#x,y = sort_xy(x,y)
#fit = fit_my_sin_2(x, y)
#y = my_sin(x, *fit)
#plt.plot(x,y)
#plt.show()







#Evolution_Project_Clones
def get_datapoints(xparam, yparam, project_name='test', plot_evo_folder='Plot_Project_Clones'):
    folder = get_data_folder() + '/'+plot_evo_folder+'/' + project_name + '/Data'
    smg = StorageManagerGroup(project_name, data_folder=folder)
    smg.sort_by(xparam)
    data = smg.get_multi_param_dict(params=[xparam, yparam] ,remove_None=True)
    return list(data[xparam]), list(data[yparam])


x1,y1=get_datapoints('t', 'txt_score', 'Layer_v4_3s_h_vs_p3')
x2,y2=get_datapoints('t', 'txt_score', 'Layer_v4_3s_h_vs_p_3600')
x = x1+x2
y = y1+y2
plt.scatter(x,y, alpha=0.1)#, c=(1,0,0,1)
draw_poly_fit(x,y)


x,y=get_datapoints('t', 'txt_score', 'Layer_v4_3s_halfHevogenome_h_vs_p')
plt.scatter(x,y, alpha=0.1)#, c=(1,0,0,1)
draw_poly_fit(x,y)

x,y=get_datapoints('t', 'txt_score', 'Layer_v4_3s_thirdHevogenome_h_vs_p2')
plt.scatter(x,y, alpha=0.1)#, c=(1,0,0,1)
draw_poly_fit(x,y)

x,y=get_datapoints('t', 'txt_score', 'Layer_v4_3s_forthHevogenome_h_vs_p2')
plt.scatter(x,y, alpha=0.1)#, c=(0,0,1,1)
draw_poly_fit(x,y)

x,y=get_datapoints('t', 'txt_score', 'Layer_v4_3s_fifthHevogenome_h_vs_p2')
plt.scatter(x,y, alpha=0.1)#, c=(0,0,1,1)
draw_poly_fit(x,y)



#def sweetspots(current_h, best_h):
#    return np.cos(((current_h-best_h)*np.pi*2)/best_h)/2.0+0.5

#x = np.arange(0, 0.1, 0.0001)
#plt.plot(x, (np.cos((x-0.02)*100)/0.5+0.5)*7)
#plt.plot(x,np.cos(np.power(1.1,x)*10000))
#plt.plot(x, sweetspots(x, 0.02)*7.0)

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

plt.axvline(1/52/6, c=(1,0,0,0.1))
plt.axvline(1/52/5, c=(1,0,0,0.1))
plt.axvline(1/52/4, c=(1,0,0,0.1))
plt.axvline(1/52/3, c=(1,0,0,0.1))
plt.axvline(1/52/2, c=(1,0,0,0.1))
plt.axvline(1/52/1, c=(1,0,0,0.1))


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

plt.xscale('log')
plt.show()
