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




x, y, z = load_datapoints(['fe_mul', 'fe_exp', 'score'], 'v5_exp_mul_scatter_2s_3sh')#'fe_mul', 'fe_exp', 'score' 'IP_s', 'LI_s' STDP_s
#plt.scatter(x,y, c=z, marker="s") #,vmin=3, vmax=7
#print(len(x))

#def fe3(v, exp, mul):
#    return np.power(np.clip(v*mul, 0.0, 1.0), exp)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

# Define some scattered data
#x = np.random.rand(50) * 10
#y = np.random.rand(50) * 10
#z = np.sin(np.sqrt(x**2 + y**2)) / np.sqrt(x**2 + y**2)

# Define a regular grid onto which to interpolate the data
xi = np.linspace(np.min(x), np.max(x), 1000)
yi = np.linspace(np.min(y), np.max(y), 1000)
xi, yi = np.meshgrid(xi, yi)


f = np.interp2d(x, y, z, kind='linear')
zi = f(xi[0,:], yi[:,0])

# Display the interpolated values as an image with a color scale
plt.imshow(zi, extent=(0, 10, 0, 10), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.show()



# Use griddata to interpolate the data onto the regular grid
zi = griddata((x, y), z, (xi, yi), method='cubic', rescale=True)

# Display the interpolated values as an image with a color scale
plt.imshow(zi, extent=(0, 10, 0, 10), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.show()