import numpy as np


def f1(x):
    return x-0.02

def f2(x, exp):
    return (x-0.02)*np.power(np.abs(x-0.02), exp)*10


#def f3(x, exp):
#    t = x-0.02
#    return np.power(np.abs(1+t), exp)*((t>0)*2-1)

def f4(x, slope):
    t = (x-0.02)*slope
    return t/np.sqrt(1+np.power(t, 2))*0.1


def f5(x):
    adj = (x-0.02) * 29.4# - 1  #
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 * 4.75#4.54

def f6(x):
    adj = (x) * 29.4# - 1  #-0.02
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 * 4.75#4.54

def f7(x):
    adj = (x-0.02) * 29.4# - 1  #
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 * 4.75 +0.24#4.54

def f8(x, c=0.02, s=170):
    return np.clip(np.clip(x-c, 0, None)*s,0,1.0)


def f9(x):
    adj = (x-0.02) * 29.4# - 1  #
    return adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1

def ft(x,y=20):
    return np.tanh(x*y)#*0.15

def fe(x, e=0.614):
    return np.power(np.clip(x-0.5,0.0,1.0)*2,e)


def get_exc_output_exponent(target_activity):
    return 0.01 / target_activity + 0.22

def get_inh_output_slope(target_activity):
    return 0.4 / target_activity + 3.6

def get_LI_threshold(target_activity):
    return np.tanh(get_inh_output_slope(target_activity) * target_activity)

from Old.Grammar.Behaviours_in_use.test import *
import matplotlib

#x = np.arange(0.0, 1.0, 0.0001)#0.2

#plt.axhline(0, color='gray')

#plt.axvline(0.03, color='gray')
#plt.axvline(0.04, color='gray')
#plt.axvline(0.06, color='gray')



x = np.arange(0.0, 0.15, 0.0001)

#[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

#for target_activity, color in zip([0.01, 0.02, 0.03, 0.05, 0.1], [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']):
    #plt.axvline(target_activity, color=color, ymin=p-0.05, ymax=p+0.05)

for target_activity, color in zip([0.01, 0.02, 0.03, 0.05, 0.1], [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']):
    p=ft(target_activity, get_inh_output_slope(target_activity))
    plt.scatter([target_activity],[p], color=color)
    plt.plot(x, ft(x, get_inh_output_slope(target_activity)), label=target_activity)


#plt.plot(x, ft(x,15))
#plt.plot(x, ft(x,9.87))
#plt.plot(x, ft(x,7.93))
plt.xlabel('activity')
plt.ylabel('spike chance')
leg=plt.legend()
leg.set_title('net. target act.')
plt.suptitle('Inhibitory activation functions', size=16)
plt.show()

#fig.tight_layout()


x = np.arange(0.0, 1.0, 0.0001)

for target_activity in  np.array([0.01, 0.02, 0.03, 0.05, 0.1]):
    plt.plot(x, fe(x, get_exc_output_exponent(target_activity)), label=target_activity)

#plt.plot(x, fe(x,0.614))
#plt.plot(x, fe(x,0.5))
#plt.plot(x, fe(x,0.437))
#plt.plot(x, fe(x,0.348))


#[6.025], [0.348], [7.93]
plt.xlabel('activity')
plt.ylabel('spike chance')
leg=plt.legend()
plt.suptitle('Excitatory activation functions', size=16)
leg.set_title('net. target act.')

plt.show()


def fL(x, th, s):
    return np.clip((x-th)*s,0,1)


x = np.arange(0.3, 1.0, 0.0001)

for target_activity in  np.array([0.01, 0.02, 0.03, 0.05, 0.1]):
    plt.plot(x, fL(x, get_LI_threshold(target_activity), 31), label=target_activity)

for target_activity, color in zip([0.01, 0.02, 0.03, 0.05, 0.1], [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd']):
    plt.scatter([get_LI_threshold(target_activity)],[0], color=color)

#plt.plot(x, fe(x,0.614))
#plt.plot(x, fe(x,0.5))
#plt.plot(x, fe(x,0.437))
#plt.plot(x, fe(x,0.348))


#[6.025], [0.348], [7.93]

plt.xlabel('inhibitory input')
plt.ylabel('learning inhibition')
leg=plt.legend()
plt.suptitle('Learning inhibition functions', size=16)
leg.set_title('net. target act.')

plt.show()

plt.bar([-1], height=-0.5)

plt.bar([0], height=0.5)

plt.bar([1], height=1)

plt.suptitle('STDP', size=16)
plt.xlabel('$\Delta$ t')
plt.ylabel('$\Delta$ s')
plt.show()



#for i in range(100):
#    y = 0.01346 + 1 / 31 + i/1000.0
#    plt.axhline(0.01346 + y, color=(1.0, 0.0, 0.0))

for i in np.arange(-400,2500,1):
    c=np.clip(i/1000.0,0,1)
    y=1/31/1000.0*i
    plt.axhline(0.01346+y, color=(1, 1-c, 1-c))

#for i in (100):
#    y = 0.01346 * i/1000.0
#    plt.axhline(0.01346 + y, color=(0.0, 1.0, 0.0))


plt.plot([0.0, 0.0, 0.0, 0.002916666666666667, 0.015416666666666667, 0.025833333333333333, 0.008333333333333333, 0.02, 0.00125, 0.02375, 0.0025, 0.02125, 0.00375, 0.03333333333333333, 0.00125, 0.006666666666666667, 0.04083333333333333, 0.0, 0.01625, 0.00125, 0.0375, 0.0, 0.0004166666666666667, 0.010833333333333334, 0.029166666666666667, 0.007916666666666667, 0.0325, 0.0, 0.018333333333333333, 0.0020833333333333333, 0.022083333333333333, 0.021666666666666667, 0.020833333333333332, 0.014583333333333334, 0.012083333333333333, 0.014166666666666666, 0.01375, 0.01875, 0.013333333333333334, 0.013333333333333334, 0.01625, 0.014166666666666666, 0.0125, 0.01375, 0.015, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.020833333333333332, 0.019166666666666665, 0.023333333333333334, 0.016666666666666666, 0.012916666666666667, 0.014166666666666666, 0.01375, 0.015833333333333335, 0.013333333333333334, 0.013333333333333334, 0.015416666666666667, 0.014166666666666666, 0.0125, 0.01375, 0.015, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.021666666666666667, 0.018333333333333333, 0.01875, 0.01375, 0.01125, 0.014166666666666666, 0.01375, 0.01625, 0.013333333333333334, 0.013333333333333334, 0.01625, 0.014166666666666666, 0.0125, 0.01375, 0.015, 0.013333333333333334, 0.013333333333333334, 0.014166666666666666, 0.023333333333333334, 0.025, 0.025416666666666667, 0.014583333333333334, 0.009166666666666667, 0.007083333333333333, 0.01125, 0.018333333333333333, 0.009583333333333333, 0.01, 0.01375, 0.011666666666666667, 0.014583333333333334, 0.013333333333333334, 0.015833333333333335, 0.016666666666666666, 0.015416666666666667, 0.015, 0.015833333333333335, 0.012916666666666667, 0.014166666666666666, 0.01375, 0.01875, 0.013333333333333334, 0.013333333333333334, 0.015, 0.014166666666666666, 0.0125, 0.01375, 0.015, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.023333333333333334, 0.024166666666666666, 0.0225, 0.014166666666666666, 0.01, 0.00875, 0.008333333333333333, 0.010833333333333334, 0.009583333333333333, 0.014583333333333334, 0.013333333333333334, 0.014166666666666666, 0.0125, 0.01375, 0.015, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.022083333333333333, 0.02125, 0.025, 0.01375, 0.01, 0.010833333333333334, 0.01375, 0.018333333333333333, 0.013333333333333334, 0.013333333333333334, 0.015416666666666667, 0.014166666666666666, 0.0125, 0.01375, 0.015416666666666667, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.022916666666666665, 0.022083333333333333, 0.02125, 0.010833333333333334, 0.010416666666666666, 0.011666666666666667, 0.01375, 0.018333333333333333, 0.013333333333333334, 0.013333333333333334, 0.01625, 0.014166666666666666, 0.0125, 0.014166666666666666, 0.014166666666666666, 0.013333333333333334, 0.013333333333333334, 0.014166666666666666, 0.024166666666666666, 0.023333333333333334, 0.024583333333333332, 0.013333333333333334, 0.009166666666666667, 0.010833333333333334, 0.012083333333333333, 0.013333333333333334, 0.01375, 0.015833333333333335, 0.0125, 0.01, 0.013333333333333334, 0.018333333333333333, 0.014166666666666666, 0.014166666666666666, 0.014583333333333334, 0.01375, 0.0016666666666666668, 0.005, 0.0075, 0.042083333333333334, 0.0, 0.005416666666666667, 0.006666666666666667, 0.005833333333333334, 0.065, 0.0, 0.0, 0.0, 0.042083333333333334, 0.0, 0.0, 0.018333333333333333, 0.0, 0.007916666666666667, 0.04666666666666667, 0.0, 0.0, 0.042083333333333334, 0.0, 0.005416666666666667, 0.006666666666666667, 0.009166666666666667, 0.0175, 0.012083333333333333, 0.015833333333333335, 0.0125, 0.01, 0.01375, 0.017916666666666668, 0.014166666666666666, 0.014166666666666666, 0.014166666666666666, 0.014583333333333334, 0.0016666666666666668, 0.012083333333333333, 0.017916666666666668, 0.0, 0.025416666666666667, 0.005416666666666667, 0.004166666666666667, 0.04958333333333333, 0.00125, 0.0, 0.00625, 0.00375, 0.06541666666666666, 0.0, 0.0, 0.0, 0.014166666666666666, 0.02125, 0.0, 0.0016666666666666668, 0.0425, 0.0, 0.0, 0.025416666666666667, 0.006666666666666667, 0.006666666666666667, 0.006666666666666667, 0.017916666666666668, 0.006666666666666667, 0.009583333333333333, 0.009166666666666667, 0.019583333333333335, 0.013333333333333334, 0.019583333333333335, 0.014166666666666666, 0.014166666666666666, 0.014583333333333334, 0.014166666666666666, 0.0016666666666666668, 0.045, 0.0, 0.0, 0.004166666666666667, 0.09166666666666666, 0.005833333333333334, 0.0, 0.0, 0.0004166666666666667, 0.03208333333333333, 0.0, 0.0, 0.059166666666666666, 0.0, 0.0, 0.0004166666666666667, 0.0225, 0.0, 0.0, 0.07833333333333334, 0.0, 0.0, 0.007916666666666667, 0.009583333333333333, 0.013333333333333334, 0.020416666666666666, 0.009583333333333333, 0.010416666666666666, 0.012916666666666667, 0.012083333333333333, 0.011666666666666667, 0.013333333333333334, 0.015833333333333335, 0.016666666666666666, 0.015, 0.014166666666666666, 0.016666666666666666, 0.013333333333333334, 0.015833333333333335, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.015833333333333335, 0.0125, 0.01, 0.013333333333333334, 0.020416666666666666, 0.014166666666666666, 0.014166666666666666, 0.014583333333333334, 0.01375, 0.0020833333333333333, 0.019166666666666665, 0.0, 0.03625, 0.0, 0.007916666666666667, 0.01375, 0.01375, 0.019166666666666665, 0.009583333333333333, 0.01, 0.01375, 0.011666666666666667, 0.015416666666666667, 0.01375, 0.015833333333333335, 0.016666666666666666, 0.015833333333333335, 0.014583333333333334, 0.016666666666666666, 0.012916666666666667, 0.014166666666666666, 0.01375, 0.017916666666666668, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.014166666666666666, 0.0125, 0.01375, 0.015833333333333335, 0.013333333333333334, 0.012916666666666667, 0.014583333333333334, 0.0225, 0.022916666666666665, 0.022083333333333333, 0.015, 0.012916666666666667, 0.01375, 0.01375, 0.017916666666666668, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.014166666666666666, 0.0125, 0.01375, 0.015416666666666667, 0.013333333333333334, 0.013333333333333334, 0.014166666666666666, 0.01375, 0.012916666666666667, 0.014166666666666666, 0.007083333333333333, 0.01, 0.015, 0.01375, 0.01875, 0.013333333333333334, 0.013333333333333334, 0.015833333333333335, 0.014166666666666666, 0.0125, 0.01375, 0.015833333333333335, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.020416666666666666, 0.020416666666666666, 0.023333333333333334, 0.015, 0.010416666666666666, 0.01125, 0.0125, 0.013333333333333334, 0.013333333333333334, 0.015833333333333335, 0.0125, 0.01, 0.013333333333333334, 0.018333333333333333, 0.014166666666666666, 0.014166666666666666, 0.014166666666666666, 0.015, 0.0016666666666666668, 0.027916666666666666, 0.0, 0.00125, 0.07541666666666667, 0.0033333333333333335, 0.0, 0.0, 0.0004166666666666667, 0.037083333333333336, 0.0, 0.00125, 0.04125, 0.0, 0.0, 0.04875, 0.0, 0.0004166666666666667, 0.0, 0.04833333333333333, 0.0, 0.0, 0.010833333333333334, 0.020416666666666666, 0.007916666666666667, 0.010833333333333334, 0.014166666666666666, 0.019583333333333335, 0.009166666666666667, 0.01, 0.011666666666666667, 0.011666666666666667, 0.014583333333333334, 0.013333333333333334, 0.01625, 0.016666666666666666, 0.014166666666666666, 0.014583333333333334, 0.015416666666666667, 0.012916666666666667, 0.014166666666666666, 0.01375, 0.016666666666666666, 0.013333333333333334, 0.013333333333333334, 0.015416666666666667, 0.014166666666666666, 0.0125, 0.014166666666666666, 0.015416666666666667, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.020833333333333332, 0.019166666666666665, 0.022083333333333333, 0.014166666666666666, 0.01, 0.010416666666666666, 0.007916666666666667, 0.009583333333333333, 0.009583333333333333, 0.009166666666666667, 0.009583333333333333, 0.035416666666666666, 0.0125, 0.01375, 0.014583333333333334, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.021666666666666667, 0.022916666666666665, 0.02375, 0.01375, 0.010833333333333334, 0.01125, 0.010416666666666666, 0.014583333333333334, 0.012916666666666667, 0.013333333333333334, 0.01625, 0.014166666666666666, 0.0125, 0.01375, 0.015, 0.013333333333333334, 0.012916666666666667, 0.014583333333333334, 0.024583333333333332, 0.025, 0.024583333333333332, 0.015, 0.01125, 0.01125, 0.013333333333333334, 0.019166666666666665, 0.009166666666666667, 0.01, 0.012083333333333333, 0.011666666666666667, 0.014166666666666666, 0.013333333333333334, 0.015833333333333335, 0.016666666666666666, 0.015, 0.014583333333333334, 0.015833333333333335, 0.013333333333333334, 0.013333333333333334, 0.013333333333333334, 0.019166666666666665, 0.009583333333333333, 0.010416666666666666, 0.0125, 0.012083333333333333, 0.014166666666666666, 0.013333333333333334, 0.01625, 0.016666666666666666, 0.015416666666666667, 0.014583333333333334, 0.015833333333333335, 0.012083333333333333, 0.014166666666666666, 0.01375, 0.018333333333333333, 0.013333333333333334, 0.013333333333333334, 0.015416666666666667, 0.014166666666666666, 0.0125, 0.01375, 0.015416666666666667, 0.013333333333333334, 0.013333333333333334, 0.014583333333333334, 0.018333333333333333, 0.015416666666666667, 0.015, 0.01125, 0.01125, 0.014166666666666666, 0.013333333333333334, 0.019166666666666665, 0.01, 0.01, 0.012083333333333333, 0.0125, 0.014166666666666666, 0.013333333333333334, 0.015833333333333335, 0.016666666666666666, 0.015833333333333335, 0.014583333333333334, 0.015833333333333335, 0.013333333333333334, 0.015833333333333335, 0.013333333333333334, 0.013333333333333334, 0.012916666666666667, 0.015833333333333335, 0.0125, 0.01, 0.013333333333333334, 0.017916666666666668, 0.014166666666666666, 0.014166666666666666, 0.014166666666666666, 0.015])

plt.axhline(0.01346, color='gray')
plt.axhline(0.01346+1/31, color='gray')

plt.xlabel('time steps')
plt.ylabel('network average output')
plt.suptitle('Learning inhibition target', size=16)

plt.show()

#5.419513086214809, 'EXP': 0.4375721000971742, 'S': 9.873254753125517

#plt.axvline(0.02, color='gray')
#plt.axvline(0.03988, color='gray')
#plt.axvline(0.0379, color='gray')


#plt.plot(x, f8(x,0.02,170))

#plt.plot(x, f8(ft(x,20),0.37698282278421735,31))

#plt.plot(x, f8(ft(x,7),0.13909244787,25))#0.13909244787

#plt.plot(x, ft(x,7))
#plt.plot(x, ft(x,7)-0.13909244787)

#plt.plot(x, x-0.02)
#plt.plot(x, ft(x,20)-0.37698282278421735)#




#plt.plot(x, f1(x))
#plt.plot(x, f2(x, 0.3))
#plt.plot(x, f2(x, 0.4))
#plt.plot(x, f2(x, 0.5))

#plt.plot(x, f4(x, 1))
#plt.plot(x, f4(x, 20))

#plt.plot(x, f5(x), label='direct inhibition')
#plt.plot(x, f6(x))
#plt.plot(x, f7(x), label='interneuron inhibition')

#plt.plot(x, inhibition_func(x, 29.4, 1, 0.050686943101760265))
#plt.plot(x, f8(x,0.02))
#plt.plot(x, f8(x,0.13909244787))

#plt.plot(x, ft(x))
#plt.plot(x, ft(x,20))
#plt.plot(x, ft(x,10))
#plt.plot(x, ft(x,30))
#plt.plot(x, ft(x,30))

#plt.plot(x, ft(x,7))

#plt.plot(x, f4(x, 100))
#plt.plot(x, f4(x, 1000))

#plt.plot(x, f4(x, 5))
#plt.plot(x, f3(x, 2))

#plt.plot(x, f2(x, 1.0))
#plt.plot(x, f2(x, 1.5))
#plt.plot(x, f2(x, 2.0))



#plt.legend(loc='best', frameon=False, fontsize=20)

#plt.show()

'''
def f(x):
    return np.power(np.abs(x - 0.5) * 2, 0.614) * (x > 0.5)

x = np.arange(0.0, 1.0, 0.0001)

plt.plot(x, f(x))

plt.show()
'''