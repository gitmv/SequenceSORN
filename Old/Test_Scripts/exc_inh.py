import matplotlib.pyplot as plt
import numpy as np


#def relu(x):
#    return np.clip((x - 0.5) * 2.0, 0.0, 1.0)

#x = np.arange(0, 1, 0.01)
#xp = np.power(x, 2)
#rp = relu(x)

#plt.plot(x, x)
#plt.plot(x, x-xp*0.5)
#plt.plot(x, x-rp*0.5)

#plt.plot(x, relu(x))
#plt.plot(x, np.power(x, 2))

#plt.show()



def exc(x):
    return x

def inh(x):
    y = (x-20)
    return y*(y > 0.0)


for i in np.arange(0, 100, 1.0):
    y = []
    a = 0.0
    for _ in range(100):
        a = i + exc(a) - inh(a)
        y.append(a)
    plt.plot(y)



#plt.plot(x, exc(x)-inh(exc(x)))

plt.show()
