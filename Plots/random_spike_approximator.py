import matplotlib.pyplot as plt
import numpy as np

def get_std(ne, p):
    variance = p*(1-p)/ne
    std_dev = np.sqrt(variance)
    return std_dev

bins = 20

x = []
for i in range(10000):
    s_e = np.random.rand(2400) > 0.5

    #w = np.random.rand(2400)
    #w /= np.sum(w)
    w = 1.0 / 2400

    x.append(np.sum(s_e*w))

plt.hist(x, bins=bins)

plt.hist(np.random.normal(0.0, np.std(x), 10000)+0.4, bins=bins)#0.005
print(np.std(x))

plt.hist(np.random.normal(0.0, get_std(2400, 0.5), 10000)+0.2, bins=bins)#0.005
print(get_std(2400, 0.5))

plt.show()


'''
def compute_std_dev(p, c):
    variance = p*(1-p)/c
    std_dev = np.sqrt(variance)
    return std_dev

#def get_std(n_ninh_neurons, target_inh):
#    p=
#    return np.sqrt(n_ninh_neurons*p(1-p))

    #sqrt(cp(1 - p))

def get_std(n_ninh_neurons, target_inh):
    return np.std(np.mean(np.random.rand(n_ninh_neurons, 10000) < target_inh, axis=0) - target_inh)

print(get_std(240, 0.2))
print(get_std(1000, 0.2))
print(get_std(10000, 0.2))

print(compute_std_dev(0.2, 240))
print(compute_std_dev(0.2, 1000))
print(compute_std_dev(0.2, 10000))

def create_random_approximator(n_ninh_neurons, target_inh):
    return np.mean(np.random.rand(n_ninh_neurons) < target_inh)-target_inh

#    #x.append(np.mean(np.random.rand(240)>0.5))

bins=20

x = []
for i in range(10000):
    x.append(create_random_approximator(240, 0.5))
plt.hist(x, bins=bins)

plt.hist(np.random.normal(0.0, np.std(x), 10000), bins=bins)#0.03
print(np.std(x))


x = []
for i in range(10000):
    x.append(create_random_approximator(1000, 0.5)+0.2)
plt.hist(x, bins=bins)

plt.hist(np.random.normal(0.0, np.std(x), 10000)+0.2, bins=bins)#0.015
print(np.std(x))

x = []
for i in range(10000):
    x.append(create_random_approximator(10000, 0.5)+0.4)
plt.hist(x, bins=bins)

plt.hist(np.random.normal(0.0, np.std(x), 10000)+0.4, bins=bins)#0.005
print(np.std(x))

plt.show()

'''