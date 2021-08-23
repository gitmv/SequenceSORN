from PymoNNto import *



class relu_output(Behaviour):

    def relu(self, x):
        return np.clip((x - 0.5) * 2.0, 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.relu(neurons.activity)


class power_output(Behaviour):

    def power(self, x):
        return np.clip(np.power(x, 4.0), 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.power(neurons.activity)


class id_output(Behaviour):

    def id(self, x):
        return np.clip(x, 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.id(neurons.activity)


class sigmoid_output(Behaviour):

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.power(np.e, -(x - 0.5) * 15))

    def new_iteration(self, neurons):
        neurons.output = self.sigmoid(neurons.activity)


class relu_step_output(Behaviour):

    def step(self, x):
        stairs = 4
        return np.clip(np.trunc((x-0.5)*2.0*stairs+1)/stairs, 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.step(neurons.activity)




#x=np.arange(0.0,1.0,0.01)
#import matplotlib.pyplot as plt
#plt.plot(x, relu_output().relu(x))
#plt.show()
#plt.plot(x, sigmoid_output().sigmoid(x))
#plt.show()
#plt.plot(x, x)
#plt.show()
#plt.plot(x, x>0.5)
#plt.show()
#plt.plot(x, power_output().power(x))
#plt.show()
#plt.plot(x, relu_step_output().step(x))
#plt.show()
