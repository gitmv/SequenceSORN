from PymoNNto import *

class create_weights(Behavior):

    def initialize(self, synapses):
        distribution = self.get_init_attr('distribution', 'uniform(1.0,1.0)')#ones
        density = self.get_init_attr('density', 1)

        synapses.W = synapses.get_synapse_mat(distribution, density=density) * synapses.enabled

        if self.get_init_attr('update_enabled', False):
            synapses.enabled *= synapses.W > 0

        normalize = self.get_init_attr('normalize', True)
        if normalize:
            synapses.W /= np.sum(synapses.W, axis=1)[:, None]

    def iteration(self, synapses):
        synapses.W = synapses.W * synapses.enabled



#Old
class init_synapses_simple(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        density = self.get_init_attr('density', None, neurons)
        for s in neurons.afferent_synapses[self.transmitter]:
            s.W = s.get_synapse_mat('ones', density=density)

    def iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            s.slow_add = s.W.dot(s.src.activity)#output

            s.dst.activity += s.slow_add