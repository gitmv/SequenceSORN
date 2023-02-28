from Old.Grammar.Behaviours_in_use import *

ui = False
neuron_count = 2400#1600
plastic_steps = 30000#100000
train_steps = 5000#10000
spont_steps = 1000#10000

SORN = Network(tag='Bruno SORN')

input_neurons = NeuronGroup(net=SORN, tag='input_neurons', size=None, behaviour={
    #init
    1: Init_Neurons(),

    #input
    11: TextGenerator(text_blocks=grammar_text_blocks_simple(), set_network_size_to_alphabet_size=True),#[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.'], ' parrots can fly.' , 'the fish swims'
    12: TextActivator_Simple(),
    13: SynapseOperation(transmitter='GLU', strength='1.0'),
    13.5: Char_Cluster_Compensation(strength=1.0),
    14: SORN_generate_output_K_WTA(K=1),

    #learning
    41: Buffer_Variables(),#for STDP
    42: STDP_C(transmitter='GLU', eta_stdp='0.00015', STDP_F={-1: 1}),#{-1: 0.2, 1: -1}
    45: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),

    #reconstruction
    50: TextReconstructor_Simple()
})

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behaviour={
    #init
    1: Init_Neurons(target_activity=0.1),

    #input
    #15: TextGenerator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.']),
    #16: TextActivator(input_density=neuron_count/60, strength=1.0, char_weighting=False),
    16: input_SynapseOperation(input_density=0.04, strength=0.75),

    17: ClassifierTextReconstructor(),

    18: SynapseOperation(transmitter='GLU', strength=1.0),
    19: SynapseOperation(transmitter='GABA', strength=-1.0),

    #stability
    21: IP(sliding_window='0', speed='0.001'),

    #output
    30: Threshold_Output(threshold='uniform(0.0, 0.5)'),

    #learning
    41: Buffer_Variables(),
    42: STDP_C(transmitter='GLU', eta_stdp=0.005, STDP_F={-1: 1, 1: -1}),
    45: Normalization(syn_type='GLU'),

    #100: STDP_Analysis(),
})

inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
    #init
    2: Init_Neurons(),
    #input
    31: SynapseOperation(transmitter='GLU', strength=1.0),
    #output
    32: Threshold_Output(threshold='uniform(0.0, 0.5)'),
})


SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,syn,EE', behaviour={
    3: create_weights(distribution='uniform(0.0,1.0)', density=10/neuron_count, update_enabled=True)
})

SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,syn', behaviour={
    3: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,syn', behaviour={
    3: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})



SynapseGroup(net=SORN, src=input_neurons, dst=exc_neurons, tag='Input_GLU,EInp', behaviour={})#weights created by input_SynapseOperation

SynapseGroup(net=SORN, src=exc_neurons, dst=input_neurons, tag='GLU,InpE', behaviour={
    3: create_weights()
})


sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)

SORN.initialize(info=True, storage_manager=sm)


#User interface
if __name__ == '__main__' and ui:
    exc_neurons.color = blue
    inh_neurons.color = red
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

SORN['ClassifierTextReconstructor', 0].start_recording()
SORN.simulate_iterations(10000, 100)
SORN.deactivate_behaviours('TextActivator')
input_neurons['STDP_C', 0].behaviour_enabled = False
input_neurons['Normalization', 0].behaviour_enabled = False
SORN['ClassifierTextReconstructor', 0].train()#starts activating after training/stops recording automatically
SORN['ClassifierTextReconstructor', 0].activate_predicted_char = False
c = SORN['ClassifierTextReconstructor', 0].readout_layer.coef_.copy()
w = SORN['InpE', 0].W.copy()

#recovery phase
SORN.simulate_iterations(5000, 100)

#text generation
#layer
SORN['TextReconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print('normal layer', SORN['TextReconstructor', 0].reconstruction_history)

#layer swapped
SORN['TextReconstructor', 0].reconstruction_history = ''
SORN['InpE', 0].W = c
SORN.simulate_iterations(5000, 100)
print('swapped layer', SORN['TextReconstructor', 0].reconstruction_history)


#classifier
SORN.deactivate_behaviours('input_SynapseOperation')
SORN['ClassifierTextReconstructor', 0].activate_predicted_char = True

SORN['ClassifierTextReconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print('classifier', SORN['ClassifierTextReconstructor', 0].reconstruction_history)

SORN['ClassifierTextReconstructor', 0].reconstruction_history = ''
SORN['ClassifierTextReconstructor', 0].readout_layer.coef_ = w
SORN.simulate_iterations(5000, 100)
print('swapped classifier', SORN['ClassifierTextReconstructor', 0].reconstruction_history)



#scoring
score = SORN['TextGenerator', 0].get_text_score(SORN['TextReconstructor', 0].reconstruction_history)
set_score(score)

import matplotlib.pyplot as plt
plt.matshow(SORN['ClassifierTextReconstructor', 0].classifier.coef_[:, 0:200])
# with bias:
# np.hstack((clf.intercept_[:,None], clf.coef_))
plt.show()

import matplotlib.pyplot as plt
plt.matshow(SORN['InpE', 0].W[:, 0:200])
plt.show()










'''
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
exc_neurons['STDP_C', 0].behaviour_enabled = False
exc_neurons['Normalization', 0].behaviour_enabled = False

input_neurons['STDP_C', 0].behaviour_enabled = False
input_neurons['Normalization', 0].behaviour_enabled = False

SORN['ClassifierTextReconstructor', 0].start_recording()
#input_neurons['STDP_C', 0].behaviour_enabled = True
#input_neurons['Normalization', 0].behaviour_enabled = True
SORN.simulate_iterations(10000, 100)
SORN.deactivate_mechanisms('TextActivator')
SORN['ClassifierTextReconstructor', 0].train()#starts activating after training/stops recording automatically

#SORN.deactivate_mechanisms('input_SynapseOperation')
SORN['ClassifierTextReconstructor', 0].activate_predicted_char = False

#recovery phase
SORN.simulate_iterations(5000, 100)

#text generation
SORN['TextReconstructor', 0].reconstruction_history = ''
SORN['ClassifierTextReconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print(SORN['TextReconstructor', 0].reconstruction_history)
print(SORN['ClassifierTextReconstructor', 0].reconstruction_history)

#scoring
score = SORN['TextGenerator', 0].get_text_score(SORN['TextReconstructor', 0].reconstruction_history)
set_score(score)

#import matplotlib.pyplot as plt
#plt.matshow(SORN['Classifier_TextReconstructor', 0].classifier.coef_[:, 0:200])
# with bias:
# np.hstack((clf.intercept_[:,None], clf.coef_))
#plt.show()

#import matplotlib.pyplot as plt
#plt.matshow(SORN['InpE', 0].W[:, 0:200])
#plt.show()
'''



'''
#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_mechanisms('STDP')
SORN.deactivate_mechanisms('Normalization')
#SORN.deactivate_mechanisms('TextActivator')
SORN['Classifier_TextReconstructor', 0].start_recording()

SORN.simulate_iterations(train_steps, 100)
SORN.deactivate_mechanisms('TextActivator')
SORN['Classifier_TextReconstructor', 0].train()#starts activating after training/stops recording automatically

SORN.deactivate_mechanisms('input_SynapseOperation')
#SORN['Classifier_TextReconstructor', 0].activate_predicted_char=False

import matplotlib.pyplot as plt
plt.matshow(SORN['Classifier_TextReconstructor', 0].classifier.coef_[:, 0:200])
# with bias:
# np.hstack((clf.intercept_[:,None], clf.coef_))
plt.show()

import matplotlib.pyplot as plt
plt.matshow(SORN['InpE', 0].W[:, 0:200])
plt.show()


SORN.simulate_iterations(spont_steps, 100)
print(SORN['Classifier_TextReconstructor', 0].reconstruction_history)

#scoring
#score = SORN['TextGenerator', 0].get_text_score(recon_text)
#set_score(score)

'''