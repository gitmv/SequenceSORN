from PymoNNto import *
from PymoNNto.NetworkBehavior.Basics.Normalization import *

class Out_Normalization(Behavior):

    def initialize(self, neurons):
        self.add_tag('SN')
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)

        neurons.require_synapses(self.syn_type, warning=False)#suppresses error when synapse group does not exist

        self.only_positive_synapses = self.get_init_attr('only_positive_synapses', True, neurons)

        self.behavior_norm_factor = self.get_init_attr('behavior_norm_factor', 1.0, neurons)
        neurons.weight_norm_factor = neurons.get_neuron_vec()+self.get_init_attr('neuron_norm_factor', 1.0, neurons)

        self.exec_every_x_step = self.get_init_attr('exec_every_x_step', 1)

    def iteration(self, neurons):

        if neurons.iteration % self.exec_every_x_step == 0:

            if self.only_positive_synapses:
                for s in neurons.afferent_synapses[self.syn_type]:
                    s.W[s.W < 0.0] = 0.0

            normalize_synapse_attr_efferent('W', 'W', neurons.weight_norm_factor*self.behavior_norm_factor, neurons, self.syn_type)
