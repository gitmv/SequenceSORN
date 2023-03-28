from Behavior_Modules import *
from recurrent_unsupervised_stdp_learning.Exp_helper import *
from Old.Grammar.Behaviors_in_use.Behavior_Bar_Activator import *

#set_genome({'TA': 3.988351708556492, 'EXP': 0.5004156607096197, 'S': 15.274553117050054})#{'TA': 3.617607251026159, 'EXP': 0.526794854151458, 'S': 15.517577277732101}#{'TA': 3.617607251026159, 'EXP': 0.526794854151458, 'S': 15.517577277732101})
#{'a': 0.7922513600022001, 'c': 6.195336615158932, 'd': 0.37000460400309465, 'S': 0.13848295213418618}

#set_genome({'a': 1.6291153700760483, 'c': 14.899524287829388, 'd': 0.6797303080873359, 'S': 0.06484569893919856})

ui = True
neuron_count = 2400#1500#2400
plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

net = Network(tag='Grammar Learning Network')


target_activity = 0.05#0.02#0.1
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

print(LI_threshold)

#20 10
#28*3, height=28*2
NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behavior={#60 30get_squared_dim(neuron_count)NeuronDimension(width=20, height=10, depth=1)

    12: SynapseOperation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_activity, strength=0.007), #0.02 '[0.03#TA]'

    # learning#0.5226654296858209
    #40: LearningInhibition_test(transmitter='GABA', a='[0.5#a]', b=0.0, c='[50.0#c]', d='[1.5#d]'),#'[0.6410769611853464#a]'
    40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    41: STDP(transmitter='GLU', strength='[0.0015#S]'),#0.005
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=1),##################
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=1),##################

    # output
    50: Generate_Output(exp=exc_output_exponent),  #'[0.614#EXP]'

    # reconstruction
    #80: TextReconstructor()

    90: Recorder(variables=['np.mean(n.output)'])
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behavior={#net['exc_neurons',0].size*0.5

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'

})

SynapseGroup(net=net, tag='EE,GLU', src='exc_neurons', dst='exc_neurons', behavior={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons', dst='inh_neurons', behavior={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons', dst='exc_neurons', behavior={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

sm = StorageManager(net.tags[0], random_nr=True)
net.initialize(info=True, storage_manager=sm)

Classifier_Weights_Pre(net['exc_neurons', 0]),
Classifier_Weights_Post(net['exc_neurons', 0]),

#User interface
if __name__ == '__main__' and ui:
    show_UI_2(net, sm)
else:
    set_score(measure_stability_score(net), info={'simulated_iterations': net.iteration})


