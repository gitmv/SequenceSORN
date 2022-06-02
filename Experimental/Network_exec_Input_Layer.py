from Behaviour_Modules import *
from Experimental.Exp_helper import *
from Old.Grammar.Behaviours_in_use.MNISTActivator import *
from Old.Grammar.Behaviours_in_use.LineActivator import *



#set_genome({'TA': 3.988351708556492, 'EXP': 0.5004156607096197, 'S': 15.274553117050054})#{'TA': 3.617607251026159, 'EXP': 0.526794854151458, 'S': 15.517577277732101}#{'TA': 3.617607251026159, 'EXP': 0.526794854151458, 'S': 15.517577277732101})

ui = True
neuron_count = 2000#1500#2400
plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

net = Network(tag='Grammar Learning Network')

#h = 0.05
#b = 0.7179041071061821 + h*100 * -0.04398732037836847
#c = 25.220023435454422 + h*100 * -2.0929686503913683

#h = 0.05
#b = 0.01 / h + 0.22
#c = 0.4 / h + 3.6
#th = np.tanh(c * h)

target_activity = 0.025#5#0.05
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

#set_genome({'TA': h, 'EXP': b, 'S': c})

#set_genome({'TA': 3.988351708556492, 'EXP': 0.5004156607096197, 'S': 15.274553117050054})

#grammar=get_default_grammar(3)#get_char_sequence(90)##get_long_text()split_into_words()#get_long_text()get_char_sequence(56)get_long_text()

#calculate_parameters(0.99, grammar, True)#get_char_sequence(56)

class Out(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)
        neurons.linh=1.0

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.activity>0.0
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)


NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=50, height=20, depth=1), color=yellow, behaviour={
    #10: Line_Patterns(center_x=50, center_y=[0,1,1,1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], degree=0, line_length=100, random_order=True),
    10: Line_Patterns(center_x=50, center_y=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], degree=0, line_length=100, random_order=True),

    #12: Synapse_Operation(transmitter='GLU', strength=1.0),

    #41: STDP(transmitter='GLU', strength=0.0015),

    50: Out()
})

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007), #0.02

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    41: STDP(transmitter='GLU', strength=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    #44: Normalization(syn_direction='afferent', syn_type='ES', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent),#[0.614#EXP]

    # reconstruction
    #80: Text_Reconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'

})

SynapseGroup(net=net, tag='ES,GLU', src='inp_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='SE,GLU', src='exc_neurons', dst='inp_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EE,GLU', src='exc_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons', dst='inh_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

sm = StorageManager(net.tags[0], random_nr=True)
net.initialize(info=True, storage_manager=sm)

CWP(net['exc_neurons', 0])

#User interface
if __name__ == '__main__' and ui:
    show_UI_2(net, sm)
else:
    print('TODO implement')
    #train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)#remove
