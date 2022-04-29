from Helper import *
from UI.UI_Helper import *
from Behaviour_Modules import *
from Behaviour_Text_Modules import *

ui = True
neuron_count = 2400
plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

net = Network(tag='Grammar Learning Network')

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={

    # excitatory input
    10: Text_Generator(text_blocks=get_default_grammar(3)),
    11: Text_Activator(input_density=0.04, strength=1.0),#0.04
    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=0.02, speed=0.007),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=0.377),
    41: STDP_simple(transmitter='GLU', eta_stdp=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=0.614),

    # reconstruction
    80: Text_Reconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=20, duration=2),

})

SynapseGroup(net=net, tag='EE,GLU', src='exc_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons', dst='inh_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0, normalize=True)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0, normalize=True)
})

sm = StorageManager(net.tags[0], random_nr=True)
net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    show_UI(net, sm)
else:
    train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)
