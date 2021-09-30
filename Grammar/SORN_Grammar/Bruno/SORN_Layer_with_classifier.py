from PymoNNto import *
from Grammar.SORN_Grammar.Behaviours_in_use import *

from Grammar.SORN_Grammar.Bruno.Logistic_Regression_Reconstruction import *

ui = False
neuron_count = 2400#2400
plastic_steps = 30000#30000

SORN = Network(tag='SORN_Layer')

input_neurons = NeuronGroup(net=SORN, tag='input_neurons', size=None, behaviour={
    #init
    1: Init_Neurons(),

    #input
    11: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.'], set_network_size_to_alphabet_size=True),
    12: Text_Activator_Simple(),
    13: Synapse_Operation(transmitter='GLU', strength='1.0'),
    13.5: Char_Cluster_Compensation(strength=1.0),
    14: SORN_generate_output_K_WTA(K=1),

    #learning
    41: Buffer_Variables(),#for STDP
    42: STDP_C(transmitter='GLU', eta_stdp='0.00015', STDP_F={-1: 1}),#{-1: 0.2, 1: -1} #, 1:-1
    45: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),

    #reconstruction
    50: Text_Reconstructor_Simple()
})

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behaviour={
    #init
    1: Init_Neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    16: input_synapse_operation(input_density=0.04, strength=0.75),#0.5 #0.04 #0.75 #1.0
    18: Synapse_Operation(transmitter='GLU', strength=1.0),
    #19: Synapse_Operation(transmitter='GABA', strength=-1.0),#-0.1

    17: Classifier_Text_Reconstructor(),

    #stability
    21: IP(sliding_window='0', speed='0.007'),
    22: Refractory_D(steps=4.0),

    #output
    30: ReLu_Output_Prob(),

    #learning
    41: Buffer_Variables(),#for STDP
    #41.5: Learning_Inhibition(transmitter='GABA', strength=-2),
    41.5: Learning_Inhibition_mean(strength=-200),
    42: STDP_C(transmitter='GLU', eta_stdp=0.0015, STDP_F={-1: 1}),#0.00015
    45: Normalization(syn_type='GLU'),

    #100: STDP_Analysis(),

})

#inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
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

sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)

SORN.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    exc_neurons.color = blue
    #inh_neurons.color = red
    input_neurons.color = yellow
    show_UI(SORN, sm, 2)



#learning
#input_neurons['STDP_C', 0].behaviour_enabled = False
#input_neurons['Normalization', 0].behaviour_enabled = False
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
exc_neurons['STDP_C', 0].behaviour_enabled = False
exc_neurons['Normalization', 0].behaviour_enabled = False

#input_neurons['STDP_C', 0].behaviour_enabled = True
#input_neurons['Normalization', 0].behaviour_enabled = True

SORN['Classifier_Text_Reconstructor', 0].start_recording()
SORN.simulate_iterations(10000, 100)
SORN.deactivate_mechanisms('Text_Activator')
input_neurons['STDP_C', 0].behaviour_enabled = False
input_neurons['Normalization', 0].behaviour_enabled = False
SORN['Classifier_Text_Reconstructor', 0].train()#starts activating after training/stops recording automatically
SORN['Classifier_Text_Reconstructor', 0].activate_predicted_char = False
c = SORN['Classifier_Text_Reconstructor', 0].readout_layer.coef_.copy()
w = SORN['InpE', 0].W.copy()

#recovery phase
SORN.simulate_iterations(5000, 100)

#text generation
#layer
SORN['Text_Reconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print('normal layer', SORN['Text_Reconstructor', 0].reconstruction_history)

#layer swapped
SORN['Text_Reconstructor', 0].reconstruction_history = ''
SORN['InpE', 0].W = c
SORN.simulate_iterations(5000, 100)
print('swapped layer', SORN['Text_Reconstructor', 0].reconstruction_history)


#classifier
SORN.deactivate_mechanisms('input_synapse_operation')
SORN['Classifier_Text_Reconstructor', 0].activate_predicted_char = True

SORN['Classifier_Text_Reconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print('classifier', SORN['Classifier_Text_Reconstructor', 0].reconstruction_history)

SORN['Classifier_Text_Reconstructor', 0].reconstruction_history = ''
SORN['Classifier_Text_Reconstructor', 0].readout_layer.coef_ = w
SORN.simulate_iterations(5000, 100)
print('swapped classifier', SORN['Classifier_Text_Reconstructor', 0].reconstruction_history)



#scoring
score = SORN['Text_Generator', 0].get_text_score(SORN['Text_Reconstructor', 0].reconstruction_history)
set_score(score, sm)

import matplotlib.pyplot as plt
plt.matshow(SORN['Classifier_Text_Reconstructor', 0].classifier.coef_[:, 0:200])
# with bias:
# np.hstack((clf.intercept_[:,None], clf.coef_))
plt.show()

import matplotlib.pyplot as plt
plt.matshow(SORN['InpE', 0].W[:, 0:200])
plt.show()





'''
#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_mechanisms('STDP')
SORN.deactivate_mechanisms('Normalization')
#SORN.deactivate_mechanisms('Text_Activator')

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