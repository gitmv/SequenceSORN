import numpy as np

def SpikeTrain_ISI(x):
    result = []
    last = -1
    for i, x in enumerate(x):
        if x > 0:
            if last!=-1:
                result.append(i-last)
            last=i

    return result


s = np.random.rand(100000) > 0.9

import matplotlib.pyplot as plt

plt.hist(SpikeTrain_ISI(s), bins=50)
plt.show()
