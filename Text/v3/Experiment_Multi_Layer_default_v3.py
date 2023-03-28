from Behavior_Core_Modules import *
from Text.v4.Behavior_Text_Modules import *
from Helper import *


ui = False
layer_sizes = [2400, 2400]
grammar = get_random_sentences(3)

target_act = 1/n_chars(grammar)

net = Network(tag=ex_file_name(), def_dtype=np.float32)


net = Network(tag='Multi Layer Cluster Formation Network')


NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behavior={
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    50: Output_InputLayer(),

    80: TextReconstructor_ML(tag='TextReconstructor')
    #80: TextReconstructor()
})


NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(layer_sizes[0]), color=blue, behavior={

    # normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.00757, init_sensitivity=0.0),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=29.25278222816641, threshold=0.2462380365675854),
    41: STDP(transmitter='GLU', strength=0.002287),

    # group output
    50: Output_Excitatory(exp=0.5716505035597882, mul=2.0),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(layer_sizes[0]/10), color=red, behavior={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # output
    70: Output_Inhibitory(slope=14.677840043903998, duration=2),
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


#more layers
layer_nr = 1
for layer_size in layer_sizes[1:]:
    layer_nr += 1
    current_layer = str(layer_nr)
    previous_layer = str(layer_nr-1)

    NeuronGroup(net=net, tag='exc_neurons'+current_layer, size=getGrid(layer_size), color=aqua, behavior={

        # normalization
        3.1: Normalization(tag='Norm', direction='afferent and efferent', syn_type='GLU', exec_every_x_step=200),

        # excitatory and inhibitory input
        12: SynapseOperation(transmitter='GLU', strength=1.0),
        20: SynapseOperation(transmitter='GABA', strength=-1.0),

        # stability
        30: IntrinsicPlasticity(target_activity=target_act, strength=0.00757, init_sensitivity=0.0),

        # learning
        40: LearningInhibition(transmitter='GABA', strength=29.25278222816641, threshold=0.2462380365675854),
        41: STDP(transmitter='GLU', strength=0.002287),

        # group output
        50: Output_Excitatory(exp=0.5716505035597882, mul=2.0),
    })

    NeuronGroup(net=net, tag='inh_neurons'+current_layer, size=get_squared_dim(layer_size/10), color=orange, behavior={

        # excitatory input
        60: SynapseOperation(transmitter='GLUI', strength=1.0),

        # output
        70: Output_Inhibitory(slope=14.677840043903998, duration=2),
    })


    SynapseGroup(net=net, tag='E2E,GLU', src='exc_neurons'+previous_layer, dst='exc_neurons'+current_layer, behavior={
        1: CreateWeights(normalize=False)#, nomr_fac=10
    })

    #SynapseGroup(net=net, tag='EE2,GLU', src='exc_neurons'+current_layer, dst='exc_neurons'+previous_layer, behavior={
    #    1: CreateWeights(normalize=False)
    #})

    SynapseGroup(net=net, tag='E2E2,GLU', src='exc_neurons'+current_layer, dst='exc_neurons'+current_layer, behavior={
        1: CreateWeights(normalize=False)
    })

    SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons'+current_layer, dst='inh_neurons'+current_layer, behavior={
        1: CreateWeights()
    })

    SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons'+current_layer, dst='exc_neurons'+current_layer, behavior={
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

