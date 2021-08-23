from PymoNNto import *

class init_neurons(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)

        neurons.target_activity = self.get_init_attr('target_activity', 0.02, neurons)

        neurons.timescale=1

    def new_iteration(self, neurons):

        neurons.activity.fill(0)# *= 0

        #neurons.output_old=neurons.output.copy()