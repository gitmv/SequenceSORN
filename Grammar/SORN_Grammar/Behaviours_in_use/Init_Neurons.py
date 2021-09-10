from PymoNNto import *

class Init_Neurons(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.target_activity = self.get_init_attr('target_activity', None, neurons)

    def new_iteration(self, neurons):
        neurons.activity.fill(0)
