import matplotlib.pyplot as plt
import numpy as np
import random

# Import the libraries
from matplotlib import cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D




default = [3.98076923, 1.32692308, 0.88461538, 0.44230769, 0.88461538,
       0.44230769, 2.65384615, 0.44230769, 0.44230769, 2.21153846,
       0.44230769, 0.88461538, 0.44230769, 0.44230769, 1.32692308,
       0.88461538, 0.44230769, 0.44230769, 1.32692308, 0.88461538,
       0.88461538, 0.44230769, 0.44230769]

best = [3.51708333, 1.40875   , 1.0925    , 0.62291667, 0.94875   ,
       0.60375   , 2.83666667, 0.47916667, 0.44083333, 2.415     ,
       0.44083333, 0.94875   , 0.44083333, 0.42166667, 1.24583333,
       0.90083333, 0.39291667, 0.39291667, 1.2075    , 0.79541667,
       0.68041667, 0.39291667, 0.37375   ]


#plt.bar(np.arange(len(default)), default, width=0.4)
#plt.bar(np.arange(len(best))+0.4, best, width=0.4)

#plt.show()

#E 0.74
#I 24.400000000000002

#x = np.arange(0, 1, 0.001)
#plt.plot(x, np.tanh(x * 24.4))
#plt.plot(x, np.tanh(x * 16.8))
#plt.plot(x, np.tanh(x * 18.22))
#plt.show()

#neurons.linh = np.clip(1 - (o - neurons.LI_threshold) * self.strength, 0.0, 1.0)

#x = np.arange(0, 1, 0.001)

#plt.plot(x, np.clip(1 - (x - neurons.LI_threshold) * 1, 0.0, 1.0))
#plt.plot(x, np.clip(1 - (x - neurons.LI_threshold) * 1, 0.0, 1.0))

#plt.show()


#x = np.arange(0, 1, 0.001)
#plt.plot(x, np.power(np.abs(x - 0.5) * 2, 0.55) * (x > 0.5))
#plt.plot(x, np.power(np.abs(x - 0.5) * 2, 0.74) * (x > 0.5))
#plt.show()


#set_genome({'T': 0.018375060660013355, 'I': 17.642840216020584, 'L': 0.28276029930786767, 'S': 0.0018671765337737584, 'E': 0.55})
#set_genome({'T': 0.018431720759132134, 'I': 18.87445616966079, 'L': 0.3216325389993774, 'S': 0.0019649003173716696, 'E': 0.55})
#set_genome({'T': 0.019, 'I': 18.222202430254626, 'L': 0.31, 'S': 0.0019, 'E': 0.55})


originaltarget_activity = 0.01923
originalexc_output_exponent = 0.01 / originaltarget_activity + 0.22
originalinh_output_slope = 0.4 / originaltarget_activity + 3.6
originalLI_threshold = np.tanh(originalinh_output_slope * originaltarget_activity)

target_activity = 0.028
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

#x = np.arange(0, 1, 0.001)
#plt.plot(x, np.power(np.abs(x - 0.5) * 2, originalexc_output_exponent) * (x > 0.5))
#plt.plot(x, np.power(np.abs(x - 0.5) * 2, exc_output_exponent) * (x > 0.5))
#plt.plot(x, np.power(np.abs(x - 0.5) * 2, 0.55) * (x > 0.5))
#plt.show()

x = np.arange(0, 0.2, 0.001)
#plt.plot(x, np.tanh(x * originalinh_output_slope))
#plt.plot(x, np.tanh(x * inh_output_slope))
plt.plot(x, np.tanh(x * 18.222202430254626))
plt.show()



#x = np.arange(0, 1.0, 0.001)
#plt.plot(x, np.clip((x - originalLI_threshold) * 31, 0.0, 1.0))
#plt.plot(x, np.clip((x - LI_threshold) * 31, 0.0, 1.0))
#plt.plot(x, np.clip((x - 0.31) * 31, 0.0, 1.0))
#plt.show()
