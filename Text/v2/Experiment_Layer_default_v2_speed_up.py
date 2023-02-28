from PymoNNto import *
from Text.v2.Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *
from Helper import *

ui = False

n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10

grammar = get_random_sentences(3)


net = Network(tag=ex_file_name())
net.def_dtype = np.float32

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=orange, behaviour={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    # group output
    50: Output_InputLayer(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons', size=getGrid(n_exc_neurons), color=blue, behaviour={
    # weight normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=0.018375, strength=0.00757, init_sensitivity=0.49),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=27.699, threshold=0.24685),
    41: STDP(transmitter='GLU', strength=0.002287),

    # group output
    50: Output_Excitatory(exp=0.60416),
})

NeuronGroup(net=net, tag='inh_neurons', size=getGrid(n_inh_neuros), color=red, behaviour={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(slope=15.5, duration=2),
})

SynapseGroup(net=net, tag='GLU,ES,SOMA', src='inp_neurons', dst='exc_neurons', behaviour={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='GLU,EE,DISTAL', src='exc_neurons', dst='exc_neurons', behaviour={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons', dst='inh_neurons', behaviour={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons', dst='exc_neurons', behaviour={
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