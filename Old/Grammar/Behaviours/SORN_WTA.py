from Old.Grammar.Behaviors.SORN_advanced_buffer import *
from PymoNNto.NetworkCore.Behavior import *

class SORN_init_neuron_vars_decay(Behavior):

    def initialize(self, neurons):
        self.add_tag('init_neuron_vars')

        neurons.activity = neurons.get_neuron_vec()
        neurons.excitation = neurons.get_neuron_vec()
        neurons.inhibition = neurons.get_neuron_vec()
        neurons.input_act = neurons.get_neuron_vec()

        neurons.timescale = self.get_init_attr('timescale', 1)

    def iteration(self, neurons):
        if first_cycle_step(neurons):
            neurons.activity *= 0.9
            neurons.excitation.fill(0)# *= 0
            neurons.inhibition.fill(0)# *= 0
            neurons.input_act.fill(0)# *= 0

class refrac(Behavior):

    def initialize(self, neurons):
        self.add_tag('Refractory_A')
        neurons.refractory_counter_analog = neurons.get_neuron_vec()
        self.decayfactor = self.get_init_attr('decayfactor', 0.5, neurons)


    def iteration(self, neurons):
        #if last_cycle_step(neurons):
            #neurons.activity -= neurons.refractory_counter_analog * self.strengthfactor

        neurons.refractory_counter_analog *= self.decayfactor
        neurons.refractory_counter_analog += neurons.output

class refrac_apply(Behavior):

    def initialize(self, neurons):
        self.add_tag('Refractory_Apply')
        self.strengthfactor = self.get_init_attr('strengthfactor', 1.0, neurons)

    def iteration(self, neurons):
        if last_cycle_step(neurons):
            neurons.activity -= neurons.refractory_counter_analog * self.strengthfactor

class SynapseOperation(SORN_signal_propagation_base):

    def initialize(self, neurons):
        super().initialize(neurons)
        self.add_tag('slow_simple' + self.transmitter)
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            s.dst.activity += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)







