from PymoNNto import *
from Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *
from Helper import *

ui = False
neuron_count = 2400
plastic_steps = 60000
recovery_steps = 10000
text_gen_steps = 5000

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D

print(grammar)

net = Network(tag='Layered Cluster Formation Network')

target_activity = gene('T', 0.018375060660013355)
exc_output_exponent = gene('E', 0.55)
inh_output_slope = gene('I', 17.642840216020584)
LI_threshold = gene('L', 0.28276029930786767)

NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=len(set(''.join(grammar))), depth=1, centered=False), color=orange, behaviour={

    10: Text_Generator(iterations_per_char=1, text_blocks=grammar),
    11: Text_Activator(strength=1),

    50: Output_Input_Layer(),

    80: Text_Reconstructor()
})


NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    # normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='STDP', exec_every_x_step=10),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='FSTDP', exec_every_x_step=10),

    # excitatory and inhibitory input
    12: Synapse_Operation(transmitter='GLU', strength=1.0),
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
    41: STDP(tag='STDP_EE', transmitter='EE', strength=0.0015),
    41.1: STDP(tag='STDP_ES', transmitter='ES', strength=0.0015),

    # output
    50: Output_Excitatory(exp=exc_output_exponent),
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Output_Inhibitory(slope=inh_output_slope, duration=2),
})

SynapseGroup(net=net, tag='GLU,ES,FSTDP', src='inp_neurons', dst='exc_neurons', behaviour={
    1: create_weights(nomr_fac=10)
})

SynapseGroup(net=net, tag='GLU,EE,STDP', src='exc_neurons', dst='exc_neurons', behaviour={
    1: create_weights(normalize=False)
})

SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons', dst='inh_neurons', behaviour={
    1: create_weights()
})

SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons', dst='exc_neurons', behaviour={
    1: create_weights()
})

sm = StorageManager(net.tags[0], random_nr=True)
sm.backup_execued_file()
net.initialize(info=True, storage_manager=sm)

net.exc_neurons.sensitivity += 0.49

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='ES')
    show_UI(net, sm)
else:
    train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)