from Grammar._common import *

ui = False
neuron_count = 2400#2400
plastic_steps = 30000#30000
recovery_steps = 5000

SORN = Network(tag='SORN_Layer_cluster')

input_neurons = NeuronGroup(net=SORN, tag='input_neurons', size=None, color=yellow, behaviour={
    #init
    1: Init_Neurons(),

    #input
    11: Text_Generator(text_blocks=get_default_grammar(3), set_network_size_to_alphabet_size=True),
    12: Text_Activator_Simple(),
    13: Synapse_Operation(transmitter='GLU', strength='1.0'),
    13.5: Char_Cluster_Compensation(strength=1.0),
    14: SORN_generate_output_K_WTA(K=1),

    #learning
    41: Buffer_Variables(),#for STDP
    42: STDP_C(transmitter='GLU', eta_stdp='0.00015', STDP_F={-1: 1}),#{-1: 0.2, 1: -1}
    45: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),

    #reconstruction
    50: Text_Reconstructor_Simple()
})

class reset_act(Behaviour):
    def new_iteration(self, neurons):
        neurons.activity.fill(0)

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={
    #init
    1: Init_Neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    16: input_synapse_operation(input_density=0.04, strength=0.75),#0.5 #0.04 #0.75 #1.0
    18: Synapse_Operation(transmitter='GLU', strength=1.0),
    #19: Synapse_Operation(transmitter='GABA', strength=-1.0),#-0.1


    21: IP2(sliding_window='0', speed='0.007'),
    22: Refractory_D(steps=4.0),
    30: ReLu_Output_Prob(),

    31: reset_act(),
    32: Synapse_Operation(transmitter='GLU_cluster', strength='0.3'),
    21: IP(sliding_window='0', speed='0.007'),
    34: ReLu_Output_Prob(),

    #20.3: reset_act(),
    #20.4: Synapse_Operation(transmitter='GLU_cluster', strength='0.3'),
    #20.5: ReLu_Output_Prob(),

    #20.6: reset_act(),
    #20.7: Synapse_Operation(transmitter='GLU_cluster', strength='0.3'),
    #20.8: ReLu_Output_Prob(),

    #learning
    41: Buffer_Variables(),#for STDP
    #41.5: Learning_Inhibition(transmitter='GABA', strength=-2),
    41.5: Learning_Inhibition_mean(strength=-200),
    42: STDP_C(transmitter='GLU', eta_stdp=0.0015, STDP_F={-1: 1}),#0.00015
    45: Normalization(syn_type='GLU'),

    43: STDP_C(transmitter='GLU_cluster', eta_stdp=0.00015, STDP_F={0: 2.0}),
    46: Normalization(syn_type='GLU_cluster'),

    #100: STDP_Analysis(),


})

#exc_neurons.visualize_module()

#inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={
    #init
#    2: Init_Neurons(),

    #input!
#    31: Synapse_Operation(transmitter='GLU', strength=30),#2.0

    #output!
    #15: threshold_output(threshold='uniform(0.1,0.9)'),
#    32: ReLu_Output(),
    #32: ReLu_Output_Prob(),
    #32: ID_Output(),
    #32: Power_Output(),
#})

SynapseGroup(net=SORN, src=input_neurons, dst=exc_neurons, tag='Input_GLU,EInp', behaviour={})#weights created by input_synapse_operation

SynapseGroup(net=SORN, src=exc_neurons, dst=input_neurons, tag='GLU,InpE', behaviour={
    3: create_weights()
})

SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,EE', behaviour={
    #init
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto'),
    3: create_weights(distribution='lognormal(1.0,0.6)', density=0.9)#uniform(0.1,1.0)
})

#SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,IE', behaviour={
    #init
#    3: create_weights(distribution='uniform(0.9,1.0)', density=0.5)#0.05
#})

#SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,EI', behaviour={
    #init
#    3: create_weights(distribution='uniform(0.9,1.0)', density=0.9)
#})

SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU_cluster,syn', behaviour={
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto'),
    3: create_weights(distribution='uniform(0.1,1.0)', density=0.9)
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
SORN.deactivate_mechanisms('Normalization')
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
set_score(score, sm, info={'text': tr.reconstruction_history, 'simulated_iterations':SORN.iteration})

'''



# 23: NOX_Diffusion(th_nox=0.0, strength=1.0),
# 24: isi_reaction_module(strength=2.0, target_exp=-1.0),
# 25: random_activity_simple(rate=0.001),

# 30: Threshold_Output(threshold=0.5),
# 30: ReLu_Output(),

#21: ip_new(sliding_window='[100#sw]', speed='[0.01#sp]'),#0.01#uniform(0.01,0.05) #

'''

import matplotlib.pyplot as plt
SORN.simulate_iteration()
W = SORN['GLU', 1].W
X = 5
Y = 5
fig, axs = plt.subplots(X, Y)
for x in range(X):
    for y in range(Y):
        neuron_syn = W[(y * Y + x)*80, :].reshape(50, 48)
        axs[x, y].matshow(neuron_syn)
plt.show()

'''

