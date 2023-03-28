from Old.Grammar_old.Bruno.Logistic_Regression_Reconstruction import *
from UI_Helper import *

ui = False
neuron_count = 2400#900#1600
plastic_steps = 10000#100000
train_steps = 5000#10000
spont_steps = 1000#10000

SORN = Network(tag='Bruno SORN')

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behavior={
    #init
    1: Init_Neurons(target_activity=0.1),

    #input
    15: TextGenerator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.']),
    16: TextActivator(input_density=neuron_count/60, strength=1.0, char_weighting=False),

    17: Classifier_TextReconstructor(),

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

    100: STDP_Analysis(),
})

inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behavior={
    #init
    2: Init_Neurons(),
    #input
    31: SynapseOperation(transmitter='GLU', strength=1.0),
    #output
    32: Threshold_Output(threshold='uniform(0.0, 0.5)'),
})


SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,syn,EE', behavior={
    3: create_weights(distribution='uniform(0.0,1.0)', density=10/neuron_count, update_enabled=True)
})

SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,syn', behavior={
    3: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,syn', behavior={
    3: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)

SORN.initialize(info=True, storage_manager=sm)


#User interface
if __name__ == '__main__' and ui:
    exc_neurons.color = blue
    inh_neurons.color = red
    show_UI(SORN, sm)


#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_behaviors('STDP')
SORN.deactivate_behaviors('Normalization')
#SORN.deactivate_behaviors('TextActivator')
SORN['ClassifierTextReconstructor', 0].start_recording()

SORN.simulate_iterations(train_steps, 100)
SORN.deactivate_behaviors('TextActivator')
SORN['ClassifierTextReconstructor', 0].train()#starts activating after training/stops recording automatically

#import matplotlib.pyplot as plt
#plt.matshow(SORN['ClassifierTextReconstructor', 0].classifier.coef_[:, 0:200])
# with bias:
# np.hstack((clf.intercept_[:,None], clf.coef_))
#plt.show()

SORN.simulate_iterations(spont_steps, 100)
print(SORN['ClassifierTextReconstructor', 0].reconstruction_history)

#scoring
#score = SORN['TextGenerator', 0].get_text_score(recon_text)
#set_score(score)


'''
import matplotlib.pyplot as plt

for i in range(100):
    SORN.simulate_iterations(100)

    SORN.simulate_iteration()
    W = SORN['EE', 0].W
    X = 4
    Y = 4
    fig, axs = plt.subplots(X, Y)
    for x in range(X):
        for y in range(Y):
            neuron_syn = W[(y * Y + x)*80, :].reshape(50, 48)
            axs[x, y].matshow(neuron_syn)
    plt.savefig(str(i)+'k.png', dpi=500)
    plt.cla()
    plt.clf()
    plt.close()
    #plt.show()
'''