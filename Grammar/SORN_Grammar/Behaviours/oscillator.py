from PymoNNto import *
import matplotlib.pyplot as plt

'''
class oscillator2(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()#-1
        neurons.f = self.get_init_attr('f', 10)
        self.shift = 0

    def new_iteration(self, neurons):
        neurons.spike = np.sin(np.pi * 2.0 * neurons.iteration / neurons.f)#(neurons.iteration % 150) == 0
        add = np.sin(np.pi * 2.0 * ((neurons.iteration + self.shift) / neurons.f))
        neurons.activity += add / neurons.f #* 3.0
        neurons.activity *= 0.99
        self.shift += ((1 - neurons.activity)-neurons.spike)/10
'''

class oscillator(Behaviour):

    def set_variables(self, neurons):
        neurons.x = neurons.get_neuron_vec()
        neurons.activity = neurons.get_neuron_vec() + 1
        neurons.f = self.get_init_attr('f', 10)

    def new_iteration(self, neurons):
        #neurons.activity += np.random.rand() * 0.1
        neurons.x += (neurons.x - neurons.activity) / neurons.f
        neurons.x *= 0.97# - 0.8/neurons.f   #0.97
        neurons.activity += neurons.x

for i in [20]:#[2, 10, 30, 50, 70, 100, 150, 200]:
    net = Network()
    ng = NeuronGroup(size=1, net=net, behaviour={1: oscillator(f=i), 2: get_Recorder('n.activity')})
    net.initialize()
    net.simulate_iterations(1000)
    plt.plot(ng['n.activity', 0])

plt.show()

plt.plot([20,50,62.5,100,125,150,200,250,500],[28,43,49,62,70,76,90,101,147])

plt.show()