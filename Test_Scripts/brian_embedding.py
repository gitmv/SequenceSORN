import PymoNNto as pymonnto
import brian2 as brian2
from PymoNNto.NetworkCore.Behaviour import *

class Brian2_embedding(Behaviour):

    def set_variables(self, neurons):
        # define resolution
        brian2.defaultclock.dt = 1.0 * brian2.ms

        # define neuron model
        eqs = '''
        dv/dt=(1.0*mV-v)/tau : volt
        tau : second'''
        self.G = brian2.NeuronGroup(
            1, model=eqs, threshold='v>0.9*mV',
            reset='v=0*mV', refractory=1.0 * brian2.ms)
        self.net = brian2.Network(self.G)

        # set variables
        self.G.v = 0.0 * brian2.mV
        self.G.tau = 100.0 * brian2.ms

    def new_iteration(self, n):
        self.net.run(1 * brian2.ms)
        n.v = self.G.v / brian2.volt

net = pymonnto.Network()

My_Neurons = pymonnto.NeuronGroup(1, net=net, behaviour={
    1: Brian2_embedding()
})

net.initialize()

for i in range(1000):
    net.simulate_iteration()
    print(My_Neurons.v)