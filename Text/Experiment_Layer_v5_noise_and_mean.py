from PymoNNto import *
from Behavior_Core_Modules_v5 import *
from Behavior_Text_Modules_v5 import *
from Helper import *


class mean_inhibition(Behavior):

    def initialize(self, neurons):
        #self.parameter('strength', 1.0)
        self.avg_act = 0
        self.duration = 2
        self.avg_inh = 0.3427857658747104
        self.target_activity = self.parameter('target_activity', 0.02)

    def get_std2(self, c, p): #number of neurons, spike chance: uniform<p
        variance = p * (1 - p) / c
        std_dev = np.sqrt(variance)
        return std_dev

    def iteration(self, neurons):
        activity = np.mean(neurons.output)#noise gets lost!!!
        self.avg_act = (self.avg_act * self.duration + activity) / (self.duration+1)
        inh = np.clip(self.avg_act*self.avg_inh/self.target_activity, 0, 1)

        std = self.get_std2(240, self.avg_inh)#self.get_std2(240, inh)

        std = std * 2.0# compensate lost noise

        neurons.input_GABA = neurons.vector()-inh+np.random.normal(0.0, std, size=neurons.size)
        neurons.voltage += neurons.input_GABA


class voltage_noise(Behavior):

    def initialize(self, neurons):
        #self.noise_fac = self.parameter('noise_fac', 0.2)
        self.std=self.get_std(240, 0.3427857658747104)


    def get_std(self, n_ninh_neurons, target_inh):
        return np.std(np.mean(np.random.rand(n_ninh_neurons, 1000) < target_inh, axis=0) - target_inh)

    def iteration(self, neurons):
        inh=np.abs(neurons.input_GABA[0])
        neurons.voltage += np.random.normal(0.0, self.get_std2(240, inh), size=neurons.size)# * self.noise_fac #neurons.vector('uniform')



ui = False
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10

grammar = get_random_sentences(3)
target_act = 1/n_chars(grammar)

net = Network(tag=ex_file_name(), settings=settings)

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behavior={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),

    # group output
    50: Output_TextActivator(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behavior={
    # weight normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),#watch out when using higher STDP speeds!
    3.1: Normalization(tag='NormFSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    #20: SynapseOperation(transmitter='GABA', strength=-1.0),

    25: mean_inhibition(target_activity=target_act),
    #26: voltage_noise(), #noise_fac=gene('nf', 0.2)

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.008735764741458582, init_sensitivity=0),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=6.450234496564654, avg_inh=0.3427857658747104, min=-0.15), #(optional)(higher sccore/risky/higher spread)
    41: STDP(transmitter='GLU', strength=0.0030597477411211885),

    # group output
    51: Output_Excitatory(exp=0.7378726012049153, mul=2.353594052973287),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behavior={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights(normalize=False)
})


sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui and not is_evolution_mode():
    from UI_Helper import *
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=60000, recovery_steps=10000, free_steps=5000, sm=sm)
