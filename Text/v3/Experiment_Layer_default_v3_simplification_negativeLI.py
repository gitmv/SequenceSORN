from PymoNNto import *
from Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *
from Helper import *

'''
class Output_Excitatory(Behaviour):

    def set_variables(self, neurons):
        self.exp = self.parameter('exp', 1.5)
        self.mul = self.parameter('mul', 2.127)
        self.act_mul = self.parameter('act_mul', 0.0)

        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)
        neurons.output_old = neurons.vector(bool)

    def activation_function(self, a):
        return a*self.mul #0.188+
        #return np.power(np.clip(a*self.mul, 0.0, 1.0), self.exp)
        #return np.power(np.abs(a - 0.5) * 2, self.exp) * (a > 0.5)

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.vector("uniform") < self.activation_function(neurons.voltage)
        neurons._voltage = neurons.voltage.copy() #for plotting
        if self.act_mul==0.0:
            neurons.voltage.fill(0.0)
        else:
            neurons.voltage *= self.act_mul
'''

class Output_Inhibitory(Behaviour):

    def set_variables(self, neurons):
        self.duration = self.parameter('duration', 2.0)
        self.avg_inh = self.parameter('avg_inh', 0.28)
        self.avg_act = 0
        self.target_activity = self.parameter('target_activity', 0.02)
        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)

    def activation_function(self, a):
        return (a*self.avg_inh)/self.target_activity

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.voltage) / (self.duration + 1)
        neurons.output = neurons.vector('uniform') < self.activation_function(self.avg_act)
        neurons._voltage = neurons.voltage.copy()  # for plotting
        neurons.voltage.fill(0)

class STDP(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.eta_stdp = self.parameter('strength', 0.005)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            src_len = np.sum(s.src.output_old)
            weight_change = s.dst.linh[s.dst.output]*self.eta_stdp
            dw = np.tile(weight_change[:,None],(1, src_len))
            s.W[np.ix_(s.dst.output, s.src.output_old)] += dw
            s.W[np.ix_(s.dst.output, s.src.output_old)] = np.clip(s.W[np.ix_(s.dst.output, s.src.output_old)], 0.0, None)

class LearningInhibition(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.parameter('strength', 1.0)
        self.avg_inh = self.parameter('avg_inh', 0.28)
        self.input_tag = 'input_' + self.parameter('transmitter', 'GABA')
        neurons.linh = neurons.vector()

    def new_iteration(self, neurons):
        inhibition = np.abs(getattr(neurons, self.input_tag))
        #neurons.linh = np.clip((neurons.LI_threshold-inhibition) * self.strength, 0.0, 1.0)
        neurons.linh = np.clip((1-inhibition/self.avg_inh) * self.strength, -gene('LIMT', 1.0), 1.0)


class CreateWeights(Behaviour):

    def set_variables(self, synapses):
        distribution = self.parameter('distribution', 'uniform(0.0,1.0)')#ones
        density = self.parameter('density', 1.0)

        synapses.W = synapses.matrix(distribution, density=density) * synapses.enabled

        self.remove_autapses = self.parameter('remove_autapses', False) and synapses.src == synapses.dst

        if self.parameter('normalize', True):
            for i in range(10):
                synapses.W /= np.sum(synapses.W, axis=0)
                synapses.W /= np.sum(synapses.W, axis=1)[:, None]
            synapses.W *= self.parameter('nomr_fac', 1.0)


    def new_iteration(self, synapses):
        if self.remove_autapses:
            np.fill_diagonal(synapses.W, 0.0)


ui = True
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10
grammar = get_random_sentences(3)

target_act = 1/n_chars(grammar)#0.019230769230769232#

#3s
set_genome({'IP_s': 0.008644124429332259, 'avg_inh': 0.33379413762020216, 'LI_s': 7.728711263682648, 'STDP_s': 0.0025713481006048714, 'fe_exp': 0.5803961466810694, 'fe_mul': 2.1850529679529758})

IP_s = gene('IP_s', 0.00757)
avg_inh = gene('avg_inh', 0.28)
LI_s = gene('LI_s', 8.158)
STDP_s = gene('STDP_s', 0.002287)
fe_exp = gene('fe_exp', 0.57165)
fe_mul = gene('fe_mul', 2.0)
#fe_mul = gene('fe_mul', 2.127)

net = Network(tag=ex_file_name())

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behaviour={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    # group output
    50: Output_InputLayer(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behaviour={
    # weight normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),
    3.1: Normalization(tag='NormFSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=IP_s, init_sensitivity=0),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=LI_s, avg_inh=avg_inh),
    41: STDP(transmitter='GLU', strength=STDP_s),

    # group output
    50: Output_Excitatory(exp=fe_exp, mul=fe_mul),
    #50: Output_Excitatory(mul=fe_mul),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh_neuros), color=red, behaviour={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(avg_inh=avg_inh, target_activity=target_act, duration=2),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behaviour={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behaviour={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1', behaviour={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1', behaviour={
    1: CreateWeights()
})

sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=60000, recovery_steps=10000, free_steps=5000, sm=sm)
