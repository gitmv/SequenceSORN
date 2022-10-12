from Old.Grammar._common import *

ui = False
neuron_count = 2400
plastic_steps = 30000
recovery_steps = 5000

SORN = Network(tag='WTA_SORN')

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={
    #init
    1: Init_Neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    15: Text_Generator(text_blocks=get_default_grammar(3)),#], ' parrots can fly.', 'the fish swims' #
    16: Text_Activator(input_density=0.04, strength=1.0),
    18: Synapse_Operation(transmitter='GLU', strength=1.0),

    #stability
    21: IP(sliding_window=0, speed=0.007),
    #22: Refractory_D(steps=4.0),

    #output
    30: K_WTA_output_local(partition_size=7, K=0.02),

    #learning
    41: Buffer_Variables(),#for STDP
    42: STDP_C(transmitter='GLU', eta_stdp=0.00015, STDP_F={-1: 1}),
    45: Normalization(syn_type='GLU'),

    #reconstruction
    50: Text_Reconstructor(),
})

SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,EE', behaviour={
    #1: Box_Receptive_Fields(range=18, remove_autapses=True),
    #2: Partition(split_size='auto'),
    3: create_weights(distribution='uniform(0.1,1.0)', density=get_gene('sd', 1.0))#lognormal(1.0,0.6)
})

sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)
SORN.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    show_UI(SORN, sm)
else:
    train_and_generate_text(SORN, plastic_steps, recovery_steps, sm=sm)






















'''
#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_mechanisms('STDP')
SORN.deactivate_mechanisms('Text_Activator')

#recovery phase
SORN.simulate_iterations(5000, 100)

#text generation
tr = SORN['Text_Reconstructor', 0]
tr.reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print(tr.reconstruction_history)

#scoring
score = SORN['Text_Generator', 0].get_text_score(tr.reconstruction_history)
set_score(score, info={'text': tr.reconstruction_history, 'simulated_iterations':SORN.iteration})

'''







#SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU_cluster,syn', behaviour={
#    1: Box_Receptive_Fields(range=18, remove_autapses=True),
#    2: Partition(split_size='auto')
#})

#15: STDP(transmitter='GLU', eta_stdp=0.00015),

# 2: init_neuron_variables(),
# 3: init_afferent_synapses(transmitter='GLU', density='full', distribution='lognormal(1.0,0.6)', normalize=True),#uniform(0.1,1.0)

# 11: IP_apply(),
# 11.1: ip_new_apply(),
# 12: refrac_apply(strengthfactor='[0.1#rs]'),

# 18: refrac(decayfactor=0.5),
# 20: IP(h_ip='lognormal_real_mean([0.02#k], [0.2944#IP_sigma])', eta_ip='[0.007#eta_ip]'),#

# 20.1: exhaustion_same_mean(),

# 3.1: init_afferent_synapses(transmitter='GLU_cluster', density='90%', distribution='uniform(0.1,1.0)', normalize=True),

# 14.1: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
# 14.2: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

# 14.3: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
# 14.4: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

# 14.5: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
# 14.6: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

# 21.2: STDP_complex(transmitter='GLU_cluster', eta_stdp='[0.00015#STDP_eta]', STDP_F={0: 1}),#{0: 2.0}
# 22.2: Normalization(syn_type='GLU_cluster', behaviour_norm_factor=0.3),