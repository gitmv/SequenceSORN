from PymoNNto import *
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

#v5_exp_mul_scatter_2

#'v5_exp_mul_scatter_2s_3sh'


'v5_exp_mul_scatter_2s_3sh_linear_genome'
'v5_exp_mul_scatter_2s_2sh_linear_genome'

'v5_exp_mul_scatter_2s_3sh'
'v5_exp_mul_scatter_2s_2sh'



'v5_mul_inh_scatter_3s_3sh'
'v5_exp_stdp_scatter_3s_3sh'
'v5_mul_stdp_scatter_3s_3sh'
'v5_ip_li_scatter_3s_3sh'


'v5_exp_mul_scatter_2'
'v5_mul_exc_scatter_3s_3sh_05inh'
'v5_exp_mul_scatter_3s_3sh_linear_genome'
'v5_exp_mul_scatter_4s_4sh'
'v5_exp_mul_scatter_4s_3sh'




x, y, z = load_datapoints(['STDP_s', 'fe_exp', 'score'], 'v5_exp_stdp_scatter_3s_3sh')#'fe_mul', 'fe_exp', 'score' 'IP_s', 'LI_s'
plt.scatter(x,y, c=z, marker="s") #,vmin=3, vmax=7
print(len(x))

def fe3(v, exp, mul):
    return np.power(np.clip(v*mul, 0.0, 1.0), exp)


import matplotlib.pyplot as plt
import matplotlib.cm as cm

# create a scalar mappable object using the default colormap
sm = cm.ScalarMappable(cmap=plt.get_cmap())
# map a single value to a color from the default colormap
#value = 0.5
#color = sm.to_rgba(value)

min_z = np.min(z)
z=z-min_z
max_z = np.max(z)
z=z/max_z

v = np.arange(0, 1, 0.001)

#for xi, yi, zi in zip(x,y,z):
#    plt.plot(v*0.1+xi, fe3(v, yi, xi)*0.025+yi, c=sm.to_rgba(zi, norm=False))

plt.show()