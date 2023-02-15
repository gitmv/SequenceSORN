from Text.New.Behaviour_Core_Modules import *
from Text.New.Behaviour_Text_Modules import *
from Text.New.Behaviour_Input_layer_Modules import *
from Helper import *

ui = False

layer_sizes=[2400, 2400]#[1000, 1000]#

#neuron_count = 2400
#neuron_count2 = 2400#int(np.minimum(np.maximum(gene('EE2SIZE', 1000), 10.0), 2400.0))

plastic_steps = 60000
recovery_steps = 10000
text_gen_steps = 5000

grammar = get_random_sentences(3)
n_chars = len(set(''.join(grammar)))

target_activity = gene('T', 0.018375060660013355)
exc_output_exponent = gene('E', 0.55)
inh_output_slope = gene('I', 17.642840216020584)
LI_threshold = gene('L', 0.28276029930786767)


net = Network(tag='Multi layer Cluster Formation Network')


NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=n_chars, depth=1, centered=False), color=green, behaviour={

    #3: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10, norm_factor=100),

    10: Text_Generator(iterations_per_char=1, text_blocks=grammar),
    11: Text_Activator_IL(strength=1),

    50: Out(),

    80: Text_Reconstructor_IL()
})


NeuronGroup(net=net, tag='exc_neurons,exc_neurons1', size=get_squared_dim(layer_sizes[0]), color=blue, behaviour={

    # normalization
    3: Normalization(tag='Norm', syn_direction='afferent', syn_type='STDP', exec_every_x_step=10),
    3.1: Normalization(tag='Norm_FSTDP', syn_direction='afferent', syn_type='FSTDP', exec_every_x_step=10),
    3.3: Normalization(tag='Norm', syn_direction='efferent', syn_type='STDP', exec_every_x_step=10),
    # 3.4: Normalization(tag='Norm_EE2',syn_direction='afferent', syn_type='EE2', exec_every_x_step=10),
    # 3.5: Normalization(tag='Norm_EE2',syn_direction='efferent', syn_type='EE2', exec_every_x_step=10),

    # excitatory input
    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold),

    41.1: STDP(transmitter='STDP', strength=0.0015),
    41.2: STDP(tag='FSTDP', transmitter='FSTDP', strength=0.00187),

    # output
    50: Generate_Output(exp=exc_output_exponent),
})

NeuronGroup(net=net, tag='inh_neurons,inh_neurons1', size=get_squared_dim(layer_sizes[0]/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2),
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


#more layers
layer_nr = 1
for layer_size in layer_sizes[1:]:
    layer_nr += 1

    NeuronGroup(net=net, tag='exc_neurons'+str(layer_nr), size=get_squared_dim(layer_size), color=aqua, behaviour={

        #normalization
        3.1: Normalization(tag='Norm', syn_direction='afferent', syn_type='STDP', exec_every_x_step=10),
        3.2: Normalization(tag='Norm', syn_direction='efferent', syn_type='STDP', exec_every_x_step=10),

        # excitatory input
        12: Synapse_Operation(transmitter='GLU', strength=1.0),

        # inhibitory input
        20: Synapse_Operation(transmitter='GABA', strength=-1.0),

        # stability
        30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007),

        # learning
        40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
        41.1: STDP(transmitter='STDP', strength=0.0015),

        # output
        50: Generate_Output(exp=exc_output_exponent),
    })

    NeuronGroup(net=net, tag='inh_neurons'+str(layer_nr), size=get_squared_dim(layer_size/10), color=orange, behaviour={

        # excitatory input
        60: Synapse_Operation(transmitter='GLUI', strength=1.0),

        # output
        70: Generate_Output_Inh(slope=inh_output_slope, duration=2),
    })


    SynapseGroup(net=net, tag='GLU,E2E,STDP', src='exc_neurons'+str(layer_nr-1), dst='exc_neurons'+str(layer_nr), behaviour={
        1: create_weights(normalize=False)#, nomr_fac=10
    })

    SynapseGroup(net=net, tag='GLU,EE2,STDP', src='exc_neurons'+str(layer_nr), dst='exc_neurons'+str(layer_nr-1), behaviour={
        1: create_weights(normalize=False)
    })

    SynapseGroup(net=net, tag='GLU,EE,STDP', src='exc_neurons'+str(layer_nr), dst='exc_neurons'+str(layer_nr), behaviour={
        1: create_weights(normalize=False)
    })

    SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons'+str(layer_nr), dst='inh_neurons'+str(layer_nr), behaviour={
        1: create_weights()
    })

    SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons'+str(layer_nr), dst='exc_neurons'+str(layer_nr), behaviour={
        1: create_weights()
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




    #for i in range(20):
    #    train_ES_and_get_distribution_score(net, 5000, deactivateEE=True)

    #train_ES_and_get_distribution_score(net, 40000)#30000 #int(gene('T', 20000))


#different h parameters for experiment
#target_activity = 0.05
#target_activity = 0.0125

#target_activity = 0.00625
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#LI_threshold = np.tanh(inh_output_slope * target_activity)

#target_activity = 1.0 / len(''.join(grammar))
#print(target_activity)#0.019230769230769232

#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#print(exc_output_exponent)
#print(inh_output_slope)

#LI_threshold = np.tanh(inh_output_slope * target_activity)

#print(LI_threshold) 0.3122864360921645