#Experiment_Time_after_evo_ng.py old name
from Text.v0.Behavior_Core_Modules import *
from Gabor.Behavior_STDP_Modules import *
from Text.v4.Behavior_Text_Modules import *
from Helper import *


#set_genome({'EE': 1.1976511689225946, 'IS': 45.18634856759456, 'I': 0.007171712492137172, 'R': 0.17320492857467587})
#set_genome({'EE': 1.5797593883794177, 'IS': 48.181668522381685, 'I': 0.009834236226210418, 'R': 0.4216672798960566})
#set_genome({'TA': 0.008250553182900128, 'EE': 2.273120313515198, 'IS': 34.118460954814815, 'GA': 0.26506483499717143, 'ST': 0.00020582873930264994, 'AM': 0.9591359460872247})

#best evo so far (1s):
#set_genome({'TA': 0.00879144796169291, 'EE': 2.746396203406996, 'IS': 32.85207616706578, 'GA': 0.23898400019898508, 'ST': 0.00016217145756623595, 'AM': 0.9307506691609726})
#set_genome({'TA': 0.009073883159464285, 'EE': 2.5929150213125673, 'IS': 35.42250549715638, 'GA': 0.2905456608942841, 'ST': 0.00015901604176955198, 'AM': 0.9378024118569637})
set_genome({'TA': 0.008880324496678695, 'EE': 2.5270280356460484, 'IS': 35.031450802782366, 'GA': 0.2215478014927156, 'ST': 0.0001673451128277578, 'AM': 0.9246457244833817})

ui = True
neuron_count = 1400

input_steps = 30000
recovery_steps = 10000
free_steps = 5000

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(1)    #Experiment D

#grammar = [' boy drinks juice.']

input_density = 0.92

target_activity = get_gene('TA', 0.008311759039259207)#0.008311759039259207#1.0 / len(''.join(grammar))
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6

#print('ex', exc_output_exponent)
#print('in', inh_output_slope)

exc_output_exponent=get_gene('EE',2.328663832385507)#get_gene('EE', 1.4231147622021607)
inh_output_slope=get_gene('IS',32.672539804654434)#get_gene('IS', 51.72459048808643)


GABA_strength = get_gene('GA', 0.25579281223890693)#0.1#0.3#!!!!!!!!!!!!!!!!!!!!!!!!!
LI_threshold = np.tanh(inh_output_slope * target_activity)*GABA_strength#0.12#0.43757758126023955

#target_activity = get_gene('T', 0.1)
#exc_output_exponent=get_gene('E', 0.32)
#inh_output_slope=get_gene('I', 7.6)
#LI_threshold=np.tanh(inh_output_slope * target_activity)#0.6410769611853464

net = Network(tag='Text Learning Network')

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behavior={

    #9: Exception_Activator(), # use for manual text input with GUI code tab...

    # excitatory input
    10: TextGenerator(text_blocks=grammar, iterations_per_char=10),
    11: TextActivator(input_density=input_density, strength=0.1),#remove for non input tests
    12: SynapseOperation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: SynapseOperation(transmitter='GABA', strength=-GABA_strength),#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! changes LI_threshold

    # stability
    30: IntrinsicPlasticity(target_activity=target_activity, strength=0.009400924413930374),#get_gene('I', 0.005967758933694278)
    31: Refrac_New(exh_add=0.4886497796272904),#get_gene('R', 0.10700595454949832)

    # learning
    40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
    41: Complex_STDP(transmitter='GLU', strength=get_gene('ST',0.0002),
                     LTP=np.array([+0.0, +0.0, +0.0, +0.0, 0.0, 0.0, 0.0, -0.0, 0.116, 0.278, 0.4, 0.421, 0.79, 0.515, 0.539]) * 2.686458,#2.0
                     LTD = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]) * 0.00617,  # *1.5
                     #LTP=np.array([+0.0, +0.0, +0.0, +0.0, +0.1, +0.2, +0.6, +1.0, +1.0, +1.0, +0.8, +0.7, +0.5, +0.4, +0.2]),
                     #LTD=np.array([-0.1, -0.3, -0.4, -0.5, -0.6, -0.6, -0.7, -0.6, -0.6, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]) * 0.0,  # *1.5
                     # LTP=np.array([+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.5, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0]),
                     # LTD=np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2])*0.0,#*0.6,
                     plot=True),

    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent, act_mul=get_gene('AM', 0.9151677506224606)), #'[0.614#EXP]'
    51: Complex_STDP_Buffer(),

    # reconstruction
    80: TextReconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behavior={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2),

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

#net.exc_neurons.sensitivity += 0.3

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=sm)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

