from PymoNNto import *
from Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *
from Helper import *

ui = False
neuron_count = 2400

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D

net = Network(tag='Layered Cluster Formation Network')


NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=orange, behaviour={

    10: Text_Generator(iterations_per_char=1, text_blocks=grammar),
    11: Text_Activator(strength=1),

    50: Output_InputLayer(),

    80: Text_Reconstructor()
})


NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    # normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=10),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=10),

    # excitatory and inhibitory input
    12: Synapse_Operation(transmitter='GLU', strength=1.0),
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=0.018375, strength=0.00757, init_sensitivity=0.49),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=27.699, threshold=0.24685),
    41: STDP(transmitter='GLU', strength=0.002287),

    # output
    50: Output_Excitatory(exp=0.60416),
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Output_Inhibitory(slope=15.5, duration=2),
})

SynapseGroup(net=net, tag='GLU,ES,SOMA', src='inp_neurons', dst='exc_neurons', behaviour={
    1: create_weights(nomr_fac=10)
})

SynapseGroup(net=net, tag='GLU,EE,DISTAL', src='exc_neurons', dst='exc_neurons', behaviour={
    1: create_weights(normalize=False)
})

SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons', dst='inh_neurons', behaviour={
    1: create_weights()
})

SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons', dst='exc_neurons', behaviour={
    1: create_weights()
})

sm = StorageManager(net.tags[0], random_nr=True)#net.tag
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='ES')
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=60000, recovery_steps=10000, free_steps=5000, sm=sm)