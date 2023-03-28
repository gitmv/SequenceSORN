from PymoNNto import *

class Init_Neurons(Behavior):

    def initialize(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.target_activity = self.get_init_attr('target_activity', None, neurons)

    def iteration(self, neurons):
        neurons.activity.fill(0)
