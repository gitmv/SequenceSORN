import PymoNNto as pymonnto
#from brian2 import *
#import brian2
from PymoNNto.NetworkCore.Behavior import *


import nest
#import nest.voltage_trace
import matplotlib.pyplot as plt
nest.set_verbosity("M_WARNING")
#nest.ResetKernel()


class Nest_embedding(Behavior):

    def initialize(self, neurons):
        self.add_tag('Nest_embedding')

        nest.SetKernelStatus({'resolution':1.0})

        self.neuron = nest.Create("iaf_psc_delta")
        
        nest.SetStatus(self.neuron, 'tau_m', 100.0)
        nest.SetStatus(self.neuron, 'E_L', 0.0)
        nest.SetStatus(self.neuron, 'V_th', 10.0)
        nest.SetStatus(self.neuron, 'V_m', 1.0)


    def iteration(self, neurons):
        nest.Simulate(1.0)

        neurons.v = nest.GetStatus(self.neuron, 'V_m')


My_Network = pymonnto.Network()

My_Neurons = pymonnto.NeuronGroup(net=My_Network, tag='my_neurons', size=pymonnto.get_squared_dim(100), behavior={
    1: Nest_embedding()
})

My_Network.initialize()

from PymoNNto.Exploration.Network_UI import *
my_UI_modules = get_default_UI_modules(['v'], ['W'])
Network_UI(My_Network, modules=my_UI_modules, label='My_Network_UI', group_display_count=1).show()
