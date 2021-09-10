from PymoNNto import *



class Threshold_Output(Behaviour):

    def set_variables(self, neurons):
        neurons.threshold = neurons.get_neuron_vec()+self.get_init_attr('threshold', 0.0, neurons)

    def new_iteration(self, neurons):
        neurons.output = (neurons.activity >= neurons.threshold)#.astype(def_dtype)



class ReLu_Output(Behaviour):

    def relu(self, x):
        return np.clip((x - 0.5) * 2.0, 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.relu(neurons.activity)


class ReLu_Output_Prob(ReLu_Output):

    def new_iteration(self, neurons):
        chance = self.relu(neurons.activity)
        neurons.output = neurons.get_neuron_vec("uniform") < chance


class Power_Output(Behaviour):

    def power(self, x):
        return np.clip(np.power(x, 4.0), 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.power(neurons.activity)


class ID_Output(Behaviour):

    def id(self, x):
        return np.clip(x, 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.id(neurons.activity)


class Sigmoid_Output(Behaviour):

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.power(np.e, -(x - 0.5) * 15))

    def new_iteration(self, neurons):
        neurons.output = self.sigmoid(neurons.activity)


class ReLu_Step_Output(Behaviour):

    def step(self, x):
        stairs = 4
        return np.clip(np.trunc((x-0.5)*2.0*stairs+1)/stairs, 0.0, 1.0)

    def new_iteration(self, neurons):
        neurons.output = self.step(neurons.activity)


class norm_output(Behaviour):

    def set_variables(self, neurons):
        self.factor = self.get_init_attr('factor', 1.0)

    def new_iteration(self, neurons):

        s=np.sum(neurons.output)
        if s>0:
            neurons.output = neurons.output/s*self.factor





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
