from Text.v2.Behavior_Core_Modules import *
from Text.v4.Behavior_Text_Modules import *
from Helper import *

ui = False

l_size = int(np.minimum(np.maximum(gene('EE2SIZE', 1000), 10.0), 2400.0))
layer_sizes=[2400, l_size]

plastic_steps = 60000
recovery_steps = 10000
text_gen_steps = 5000

grammar = get_random_sentences(3)
n_chars = len(set(''.join(grammar)))

target_activity = gene('T', 0.018375060660013355)
exc_output_exponent = gene('E', 0.55)
inh_output_slope = gene('I', 17.642840216020584)
LI_threshold = gene('L', 0.28276029930786767)


net = Network(tag='Multi Layer Cluster Formation Network')


NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=n_chars, depth=1, centered=False), color=green, behavior={
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    50: Output_InputLayer(),

    80: TextReconstructor()
})


NeuronGroup(net=net, tag='exc_neurons,exc_neurons1', size=get_squared_dim(layer_sizes[0]), color=blue, behavior={

    # normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='STDP', exec_every_x_step=10),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='FSTDP', exec_every_x_step=10),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold),

    41.1: STDP(transmitter='STDP', strength=0.0015),
    41.2: STDP(tag='FSTDP', transmitter='FSTDP', strength=0.00187),

    # output
    50: Output_Excitatory(exp=exc_output_exponent),
})

NeuronGroup(net=net, tag='inh_neurons,inh_neurons1', size=get_squared_dim(layer_sizes[0]/10), color=red, behavior={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # output
    70: Output_Inhibitory(slope=inh_output_slope, duration=2),
})

SynapseGroup(net=net, tag='GLU,ES,FSTDP', src='inp_neurons', dst='exc_neurons', behavior={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='GLU,EE,STDP', src='exc_neurons', dst='exc_neurons', behavior={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons', dst='inh_neurons', behavior={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons', dst='exc_neurons', behavior={
    1: CreateWeights()
})


#more layers
layer_nr = 1
for layer_size in layer_sizes[1:]:
    layer_nr += 1

    NeuronGroup(net=net, tag='exc_neurons'+str(layer_nr), size=get_squared_dim(layer_size), color=aqua, behavior={

        # normalization
        3.1: Normalization(tag='Norm', direction='afferent and efferent', syn_type='STDP', exec_every_x_step=10),

        # excitatory and inhibitory input
        12: SynapseOperation(transmitter='GLU', strength=1.0),
        20: SynapseOperation(transmitter='GABA', strength=-1.0),

        # stability
        30: IntrinsicPlasticity(target_activity=target_activity, strength=0.007),

        # learning
        40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
        41.1: STDP(transmitter='STDP', strength=0.0015),

        # output
        50: Output_Excitatory(exp=exc_output_exponent),
    })

    NeuronGroup(net=net, tag='inh_neurons'+str(layer_nr), size=get_squared_dim(layer_size/10), color=orange, behavior={

        # excitatory input
        60: SynapseOperation(transmitter='GLUI', strength=1.0),

        # output
        70: Output_Inhibitory(slope=inh_output_slope, duration=2),
    })


    SynapseGroup(net=net, tag='GLU,E2E,STDP', src='exc_neurons'+str(layer_nr-1), dst='exc_neurons'+str(layer_nr), behavior={
        1: CreateWeights(normalize=False)#, nomr_fac=10
    })

    SynapseGroup(net=net, tag='GLU,EE2,STDP', src='exc_neurons'+str(layer_nr), dst='exc_neurons'+str(layer_nr-1), behavior={
        1: CreateWeights(normalize=False)
    })

    SynapseGroup(net=net, tag='GLU,E2E2,STDP', src='exc_neurons'+str(layer_nr), dst='exc_neurons'+str(layer_nr), behavior={
        1: CreateWeights(normalize=False)
    })

    SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons'+str(layer_nr), dst='inh_neurons'+str(layer_nr), behavior={
        1: CreateWeights()
    })

    SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons'+str(layer_nr), dst='exc_neurons'+str(layer_nr), behavior={
        1: CreateWeights()
    })



sm = StorageManager(net.tags[0], random_nr=True)
net.initialize(info=True, storage_manager=sm)

net.exc_neurons.sensitivity += 0.49
#net.exc_neurons2.sensitivity += 0.2#49


#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='ES')
    show_UI(net, sm)
else:
    train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)
