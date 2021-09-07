from PymoNNto import *
from Grammar.SORN_Grammar.Behaviours_in_use import *

ui = True
neuron_count = 2400
plastic_steps = 30000

SORN = Network(tag='SORN_Layer')

input_neurons = NeuronGroup(net=SORN, tag='input_neurons', size=None, behaviour={
    #init
    1: init_neurons(),

    #input
    11: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.'], set_network_size_to_alphabet_size=True),
    12: Text_Activator_Simple(),
    13: synapse_operation(transmitter='GLU', strength='1.0'),
    13.5: char_cluster_compensation(),
    14: SORN_generate_output_K_WTA(K=1),

    #learning
    41: buffer_variables(),#for STDP
    42: STDP_complex(transmitter='GLU', eta_stdp='0.00015', STDP_F={-1: 1}),#{-1: 0.2, 1: -1}
    45: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),

    #reconstruction
    50: Text_Reconstructor_Simple()
})

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behaviour={
    #init
    1: init_neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    16: input_synapse_operation(input_density=0.04, strength=0.75),#0.5 #0.04
    18: synapse_operation(transmitter='GLU', strength=1.0),
    #19: synapse_operation(transmitter='GABA', strength=-0.3),#-0.1

    #stability
    21: ip_new(sliding_window='0', speed='0.007'),#21: ip_new(sliding_window='[100#sw]', speed='[0.01#sp]'),#0.01#uniform(0.01,0.05)
    22: Refractory_D(steps=4.0),
    #23: NOX_Diffusion(th_nox=0.0, strength=1.0),
    #24: isi_reaction_module(strength=2.0, target_exp=-1.0),
    #25: random_activity_simple(rate=0.001),

    #output
    #30: threshold_output(threshold=0.5),
    #30: relu_output(),
    30: relu_output_probablistic(),
    #30: power_output(),
    #30: relu_step_output(),
    #30: id_output(),
    #30: sigmoid_output(),

    #learning
    41: buffer_variables(),#for STDP

    41.5: learning_inhibition(transmitter='GABA', strength=-10),

    42: STDP_complex(transmitter='GLU', eta_stdp=0.00015, STDP_F={-1: 1}),
    45: Normalization(syn_type='GLU'),

    100: STDP_analysis(),

})

inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
    #init
    2: init_neurons(),

    #input!
    11: synapse_operation(transmitter='GLU', strength=17),#2.0

    #output!
    #15: threshold_output(threshold='uniform(0.1,0.9)'),
    15: relu_output()
})

SynapseGroup(net=SORN, src=input_neurons, dst=exc_neurons, tag='Input_GLU,syn', behaviour={})#weights created by input_synapse_operation

SynapseGroup(net=SORN, src=exc_neurons, dst=input_neurons, tag='GLU,syn', behaviour={
    3: create_weights()
})

SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,syn', behaviour={
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto'),
    3: create_weights(distribution='lognormal(1.0,0.6)', density=0.9)#uniform(0.1,1.0)
})

SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,syn', behaviour={
    #init
    3: create_weights(distribution='uniform(0.5,1.0)', density=0.2)#0.05
})

SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,syn', behaviour={
    #init
    3: create_weights(distribution='uniform(0.5,1.0)', density=0.5)
})



sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)

SORN.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    exc_neurons.color = blue
    inh_neurons.color = red
    input_neurons.color = yellow
    show_UI(SORN, sm, 3)





#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_mechanisms('STDP')
SORN.deactivate_mechanisms('Text_Activator')

#recovery phase
SORN.simulate_iterations(5000, 100)

#text generation
SORN['Text_Reconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
recon_text = SORN['Text_Reconstructor', 0].reconstruction_history
print(recon_text)

#scoring
score = SORN['Text_Generator', 0].get_text_score(recon_text)
set_score(score, sm)





#SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU_cluster,syn', behaviour={
#    1: Box_Receptive_Fields(range=18, remove_autapses=True),
#    2: Partition(split_size='auto')
#})

#3: init_afferent_synapses(transmitter='GLU', density='90%', distribution='uniform(0.1,1.0)', normalize=True),

#0.3: init_synapses_simple(transmitter='GLU'),

#15: STDP(transmitter='GLU', eta_stdp=0.00015),

# 3.1: init_afferent_synapses(transmitter='GLU_cluster', density='90%', distribution='uniform(0.1,1.0)', normalize=True),

# 14.1: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
# 14.2: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

# 14.3: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
# 14.4: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

# 14.5: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
# 14.6: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

# 21.2: STDP_complex(transmitter='GLU_cluster', eta_stdp='[0.00015#STDP_eta]', STDP_F={0: 2.0}),
# 22.2: Normalization(syn_type='GLU_cluster', behaviour_norm_factor=0.3),