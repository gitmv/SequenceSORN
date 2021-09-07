from PymoNNto import *

class delayed_activity_response(Behaviour):

    def set_variables(self, neurons):
        neurons.x = neurons.get_neuron_vec()
        neurons.delay = self.get_init_attr('delay', 10)

    def new_iteration(self, neurons):

