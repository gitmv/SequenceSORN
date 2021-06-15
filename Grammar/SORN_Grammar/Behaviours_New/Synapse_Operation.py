from PymoNNto import *

class init_synapses_simple(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        density = self.get_init_attr('density', None, neurons)
        for s in neurons.afferent_synapses[self.transmitter]:
            s.W = s.get_synapse_mat('ones', density=density)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            s.slow_add = s.W.dot(s.src.output)

            s.dst.activity += s.slow_add