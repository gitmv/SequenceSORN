import numpy as np
import matplotlib.pyplot as plt

# Define the functions
x = np.linspace(0, 1, 1000)

def sigmoid(x, b):
    return 1 / (1 + np.exp(-b * x))

def tanh(x, b):
    return np.tanh(b * x)

def exponential(x, b):
    return (np.exp(b * x)-1)/10

def inverse(x, b):
    return x ** (1 / b)

def power(x, b):
    return x ** b

def arcsine(x):
    return np.arcsin(np.sqrt(x))

def logarithmic(x, b):
    return (-np.log(1 / x) / np.log(b))/10+1

def gaussian(x, b):
    return np.exp(-b * (x-1) ** 2)

def weibull(x, a, b):
    return 1 - np.exp(-b * x ** a)

#def beta(x, a, b):
#    return x ** (a - 1) * (1 - x) ** (b - 1) / scipy.special.beta(a, b)

# Plot the functions
plt.figure(figsize=(12, 8))

plt.plot(x, sigmoid(x, 10), label='Sigmoid')
plt.plot(x, tanh(x, 5), label='Tanh')
plt.plot(x, exponential(x, 3), label='Exponential')
plt.plot(x, inverse(x, 2), label='Inverse')
plt.plot(x, power(x, 3), label='Power')
plt.plot(x, arcsine(x), label='Arcsine')
plt.plot(x, logarithmic(x, 10), label='Logarithmic')
plt.plot(x, gaussian(x, 5), label='Gaussian')
plt.plot(x, weibull(x, 2, 5), label='Weibull')
#plt.plot(x, beta(x, 2, 3), label='Beta')

plt.legend()
plt.show()