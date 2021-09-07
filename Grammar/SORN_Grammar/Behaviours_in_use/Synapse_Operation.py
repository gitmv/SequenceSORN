from PymoNNto import *
from PymoNNto.NetworkBehaviour.Basics.Normalization import *

class SORN_signal_propagation_base(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1, neurons)  # 1 or -1

        if self.transmitter==None:
            print('warning no transmitter defined')
        #elif self.transmitter is not 'GLU' and self.transmitter is not 'GABA':
        #    print('warning unknown transmitter')

        if self.transmitter=='GLU' and self.strength<0:
            print('warning glutamate strength is inhibitory')

        if self.transmitter=='GABA' and self.strength>0:
            print('warning GABA strength is excitatory')

    def new_iteration(self, neurons):
        print('warning: signal_propagation_base has to be overwritten')

class synapse_operation(SORN_signal_propagation_base):

    def set_variables(self, neurons):
        super().set_variables(neurons)
        self.add_tag('slow_simple' + self.transmitter)
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def new_iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            s.dst.activity += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)

#requires STDP
class learning_inhibition(SORN_signal_propagation_base):

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            buffer = neurons.buffers['output']
            buffer[1] = np.clip(buffer[1]+s.add, 0.0, 1.0)
