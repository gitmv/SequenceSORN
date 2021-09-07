from PymoNNto import *

class create_weights(Behaviour):

    def set_variables(self, synapses):
        distribution = self.get_init_attr('distribution', 'uniform(1.0,1.0)')#ones
        density = self.get_init_attr('density', 1)

        synapses.W = synapses.get_synapse_mat(distribution, density=density) * synapses.enabled

        normalize = self.get_init_attr('normalize', True)
        if normalize:
            synapses.W /= np.sum(synapses.W, axis=1)[:, None]

    def new_iteration(self, synapses):
        synapses.W = synapses.W * synapses.enabled



#Old
class init_synapses_simple(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        density = self.get_init_attr('density', None, neurons)
        for s in neurons.afferent_synapses[self.transmitter]:
            s.W = s.get_synapse_mat('ones', density=density)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            s.slow_add = s.W.dot(s.src.activity)#output

            s.dst.activity += s.slow_add