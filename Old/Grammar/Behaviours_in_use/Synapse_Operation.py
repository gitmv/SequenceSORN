from PymoNNto import *
from PymoNNto.NetworkBehavior.Basics.Normalization import *

class SORN_signal_propagation_base(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1.0, neurons)  # 1 or -1

        if self.transmitter==None:
            print('warning no transmitter defined')
        #elif self.transmitter is not 'GLU' and self.transmitter is not 'GABA':
        #    print('warning unknown transmitter')

        if self.transmitter=='GLU' and self.strength<0:
            print('warning glutamate strength is inhibitory')

        if self.transmitter=='GABA' and self.strength>0:
            print('warning GABA strength is excitatory')

    def iteration(self, neurons):
        print('warning: signal_propagation_base has to be overwritten')









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








#requires STDP
class LearningInhibition(SORN_signal_propagation_base):

    def iteration(self, neurons):
        neurons.linh = neurons.get_neuron_vec()
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            s.dst.linh += s.add
        buffer = neurons.buffers['output']
        buffer[1] = np.clip(buffer[1]+neurons.linh, 0.0, 1.0)


#requires STDP
class LearningInhibition_mean(Behavior):

    def initialize(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)
        self.threshold = self.get_init_attr('threshold', 0.02, neurons)
        self.use_inh = self.get_init_attr('use_inh', True, neurons)

    def iteration(self, neurons):
        if self.use_inh:
            o = np.mean(neurons.inh) - self.threshold#0.13909244787
        else:
            o = np.mean(neurons.output) - self.threshold

        neurons.linh = np.clip(o, 0, None)*self.strength
        buffer = neurons.buffers['output']
        buffer[1] = np.clip(buffer[1] - neurons.linh, 0.0, 1.0)

class LearningInhibition_GABA(Behavior):

    def initialize(self, neurons):
        self.const = np.tanh(neurons.target_activity*20)

    def iteration(self, neurons):
        neurons.linh = neurons.input_GABA < self.const
        # np.clip(neurons.input_GABA-neurons.target_activity, 0, None)*(-170)
        buffer = neurons.buffers['output']
        #buffer[1] = np.clip(buffer[1] + neurons.linh, 0.0, 1.0)
        buffer[1] = buffer[1] + neurons.linh * neurons.linh
















class Collect_Synapse_Input(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1, neurons)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class Apply_Synapse_Input(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1, neurons)
        self.input_tag = 'input_' + self.transmitter

    def iteration(self, neurons):
        neurons.activity += getattr(neurons, self.input_tag)*self.strength




class LearningInhibition_mean2(Behavior):

    def initialize(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)
        self.threshold = self.get_init_attr('threshold', 0.02, neurons)
        self.inh_attr = self.get_init_attr('inh_attr', 'input_GABA', neurons)#output or input_GABA or inh

    def iteration(self, neurons):
        o = np.mean(np.abs(getattr(neurons, self.inh_attr))) - self.threshold#mean!!!!!!!!!!!!!

        #if self.use_inh:
        #    o = np.mean(neurons.input_GABA) - self.threshold#0.13909244787
        #else:
        #    o = np.mean(neurons.output) - self.threshold

        neurons.linh = np.clip(o, 0, None)*self.strength
        buffer = neurons.buffers['output']
        buffer[1] = np.clip(buffer[1] - neurons.linh, 0.0, 1.0)