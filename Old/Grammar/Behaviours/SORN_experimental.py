from PymoNNto.NetworkCore.Behavior import *

#from Grammar.Behaviors.SORN_advanced_buffer import *
from PymoNNto.NetworkBehavior.Basics.BasicHomeostasis import *






class SORN_WTA_fast_syn(SORN_signal_propagation_base):

    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag('fast_' + self.transmitter)
        self.get_init_attr('so', None, neurons)#just for error suppression

    def iteration(self, neurons):
        if last_cycle_step(neurons) and self.strength != 0:
            for s in neurons.afferent_synapses[self.transmitter]:
                s.fast_add = s.W.dot(s.src.output) * self.strength# / neurons.timescale

                s.dst.activity += s.fast_add
                if self.strength > 0:
                    s.dst.excitation += s.fast_add
                else:
                    s.dst.inhibition += s.fast_add

class SORN_WTA_iSTDP(Behavior):

    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag('WTA iSTDP')
        self.eta_iSTDP = self.get_init_attr('eta_iSTDP', 0.1, neurons)

    def iteration(self, neurons):
        if last_cycle_step(neurons):
            for s in neurons.afferent_synapses['GABA']:
                add = s.dst.activity[:, None] * s.src.activity[None, :] * self.eta_iSTDP * s.enabled
                #print(add.shape)
                s.W += add


class SORN_IP_TI_WTA(Time_Integration_Homeostasis):

    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag('IP_TI_WTA')
        self.measurement_param = self.get_init_attr('mp', 'n.output', neurons)
        self.adjustment_param = 'exhaustion_value'

        self.set_threshold(self.get_init_attr('h_ip', 0.1, neurons))
        self.adj_strength = -self.get_init_attr('eta_ip', 0.001, neurons)
        self.target_clip_min=self.get_init_attr('clip_min', -0.001, neurons)

        #target_clip_max

        neurons.exhaustion_value = 0

    def iteration(self, neurons):
        if last_cycle_step(neurons):
            super().iteration(neurons)
            neurons.activity -= neurons.exhaustion_value


class IP(Instant_Homeostasis):

    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag('IP_WTA')
        self.measurement_param = self.get_init_attr('mp', 'n.output', neurons)
        self.adjustment_param = 'exhaustion_value'

        self.set_threshold(self.get_init_attr('h_ip', 0.1, neurons))
        self.adj_strength = -self.get_init_attr('eta_ip', 0.001, neurons)

        #target_clip_max

        neurons.exhaustion_value = 0

    def iteration(self, neurons):
        super().iteration(neurons)
        #neurons.activity -= neurons.exhaustion_value

class exhaustion_same_mean(Behavior):

    def iteration(self, neurons):
        neurons.exhaustion_value = neurons.exhaustion_value - np.mean(neurons.exhaustion_value)

class IP_apply(Behavior):

    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag('SORN_IP_WTA_apply')

    def iteration(self, neurons):
        if last_cycle_step(neurons):
            neurons.activity -= neurons.exhaustion_value

'''
class SORN_Neuron_Exhaustion(Neuron_Behavior):

    def initialize(self, neurons):
        self.add_tag('exhaustion')
        neurons.exhaustion_value = neurons.get_neuron_vec()
        self.decay_factor = self.get_init_attr('decay_factor', 0.9, neurons)
        self.strength = self.get_init_attr('strength', 0.1, neurons)

    def iteration(self, neurons):
        if last_cycle_step(neurons):
            neurons.exhaustion_value *= self.decay_factor
            neurons.exhaustion_value += neurons.output

            neurons.activity -= neurons.exhaustion_value * self.strength
'''