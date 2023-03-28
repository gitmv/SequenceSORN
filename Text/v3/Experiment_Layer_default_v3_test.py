from Text.v2.Behavior_Core_Modules import *
from Text.v4.Behavior_Text_Modules import *
from Helper import *

class Output_Inhibitory(Behavior):

    def initialize(self, neurons):
        self.duration = self.parameter('duration', 2.0)
        self.slope = self.parameter('slope', 14.3)
        self.avg_act = 0
        neurons.activity = neurons.vector()
        neurons.output = neurons.vector(bool)

    def activation_function(self, a):
        return a*self.slope

    def iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        neurons.output = neurons.vector('uniform') < self.activation_function(self.avg_act)#neurons.inh
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)


class Output_Excitatory(Behavior):

    def initialize(self, neurons):
        self.exp = self.parameter('exp', 1.5)
        self.mul = self.parameter('mul', 2.0)
        self.act_mul = self.parameter('act_mul', 0.0)

        neurons.activity = neurons.vector()
        neurons.output = neurons.vector(bool)
        neurons.output_old = neurons.vector(bool)

    def activation_function(self, a):
        return np.power(np.clip(a*self.mul, 0.0, 1.0), self.exp)

    def iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.vector("uniform") < self.activation_function(neurons.activity)
        neurons._activity = neurons.activity.copy() #for plotting
        if self.act_mul==0.0:
            neurons.activity.fill(0.0)
        else:
            neurons.activity *= self.act_mul


#{'I_s': 14.677840043903998, 'E_exp': 0.5716505035597882, 'LI_s': 29.25278222816641, 'LI_t': 0.2462380365675854}

I_s = gene('I_s', 14.677840043903998)
E_exp = gene('E_exp', 0.5716505035597882)
E_mul = gene('E_mul', 2.0)
LI_s = gene('LI_s', 29.25278222816641)
LI_t = gene('LI_t', 0.2462380365675854)

#I_s = gene('I_s', 14.3)
#E_exp = gene('E_exp', 0.60416)
#E_mul = gene('E_mul', 2.0)
#LI_s = gene('LI_s', 27.699)
#LI_t = gene('LI_t', 0.24685)


ui = False
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10
grammar = get_random_sentences(3)

target_act = 1/n_chars(grammar)

net = Network(tag=ex_file_name())

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=orange, behavior={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    # group output
    50: Output_InputLayer(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behavior={
    # weight normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=100),
    3.1: Normalization(tag='NormFSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=100),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.00757, init_sensitivity=-0.01),#0.49

    # learning
    40: LearningInhibition(transmitter='GABA', strength=LI_s, threshold=LI_t),
    41: STDP(transmitter='GLU', strength=0.002287),

    # group output
    50: Output_Excitatory(exp=E_exp, mul=E_mul),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh_neuros), color=red, behavior={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(slope=I_s, duration=2),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behavior={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1', behavior={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1', behavior={
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