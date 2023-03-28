from PymoNNto import *

class delayed_activity_response(Behavior):

    def initialize(self, neurons):
        neurons.x = neurons.get_neuron_vec()
        neurons.delay = self.get_init_attr('delay', 10)

    def iteration(self, neurons):
        return
