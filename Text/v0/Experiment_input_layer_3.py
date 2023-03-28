from Text.v0.Behavior_Core_Modules import *
from Text.v0.Behavior_Text_Modules import *
from Helper import *

class TextActivator_IL(Behavior):

    def initialize(self, neurons):
        self.add_tag('TextActivator')
        self.TextGenerator = neurons['TextGenerator', 0]
        self.strength = self.get_init_attr('strength', 1, neurons)

    def iteration(self, neurons):
        neurons.input_grammar = (neurons.y == neurons.current_char_index)*self.strength
        neurons.activity += neurons.input_grammar
        #neurons.output = neurons.activity>0
        #print(neurons.input_grammar)



class TextReconstructor_IL(Behavior):

    def initialize(self, neurons):
        self.add_tag('TextReconstructor')
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def iteration(self, neurons):
        if neurons['TextActivator_IL', 0] is not None:

            neurons.rec_act = neurons.get_neuron_vec()
            for s in neurons.efferent_synapses['GLU']:
                s.src.rec_act += s.W.T.dot(s.dst.output)

            if np.sum(neurons.rec_act)==0:
                self.current_reconstruction_char_index = -1
                self.current_reconstruction_char = '#'
            else:
                index_act = np.sum(neurons.rec_act.reshape((neurons.height, neurons.width)), axis=1)
                self.current_reconstruction_char_index = np.argmax(index_act)
                self.current_reconstruction_char = neurons['TextActivator_IL', 0].TextGenerator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char



class Out(Behavior):

    def initialize(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)
        neurons.linh=1.0

    def iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.activity>0.0
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)

#set_genome({'E': '0.4887617702816654', 'I': '20.939463630168248', 'L': '0.18462995264097187', 'S': '0.0021527580600582286'})

#set_genome({'T': 0.019, 'I': 18.222202430254626, 'L': 0.31, 'S': 0.0019, 'E': 0.55})
#set_genome({'T': 0.018431720759132134, 'I': 18.87445616966079, 'L': 0.3216325389993774, 'S': 0.0019649003173716696, 'E': 0.55})
set_genome({'T': 0.018375060660013355, 'I': 17.642840216020584, 'L': 0.28276029930786767, 'S': 0.0018671765337737584, 'E': 0.55})

ui = False
neuron_count = 2400
plastic_steps = 30000+30000
recovery_steps = 10000
text_gen_steps = 5000

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D

#print(len(''.join(grammar)))

net = Network(tag='Cluster Formation Network')

#different h parameters for experiment
#target_activity = 0.05
#target_activity = 0.0125

#target_activity = 0.00625
#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#LI_threshold = np.tanh(inh_output_slope * target_activity)

#target_activity = 1.0 / len(''.join(grammar))
target_activity = gene('T', 0.017)
#print(target_activity)#0.019230769230769232

#exc_output_exponent = 0.01 / target_activity + 0.22
#inh_output_slope = 0.4 / target_activity + 3.6
#print(exc_output_exponent)
#print(inh_output_slope)

exc_output_exponent = gene('E', 0.55)
inh_output_slope = gene('I', 16.8)

#LI_threshold = np.tanh(inh_output_slope * target_activity)

LI_threshold = gene('L', 0.2)#0.25

#print(LI_threshold) 0.3122864360921645

NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=len(set(''.join(grammar))), depth=1, centered=False), color=orange, behavior={

    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator_IL(strength=1),

    #42.1: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10, norm_factor=100),

    50: Out(),

    80: TextReconstructor_IL()
})


NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behavior={#60 30#NeuronDimension(width=10, height=10, depth=1)

    12: SynapseOperation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_activity, strength=0.007), #0.02

    # learning
    40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    41: STDP(tag='STDP_EE', transmitter='EE', strength=0.0015),#0.0015#gene('S1',0.0015)
    41.1: STDP(tag='STDP_ES', transmitter='ES', strength=gene('S',0.0015)),#0.0015

    42: Normalization(tag='Norm_EE', syn_direction='afferent', syn_type='EE', exec_every_x_step=10),
    42.1: Normalization(tag='Norm_ES',syn_direction='afferent', syn_type='ES', exec_every_x_step=10),#3#10, norm_factor=gene('N',1)
    43: Normalization(tag='Norm_EE', syn_direction='efferent', syn_type='EE', exec_every_x_step=10),

    #42: Normalization(syn_direction='afferent', syn_type='EE', exec_every_x_step=10),
    #43: Normalization(syn_direction='efferent', syn_type='EE', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent),
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behavior={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'
})

SynapseGroup(net=net, tag='ES,GLU', src='inp_neurons', dst='exc_neurons', behavior={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0, nomr_fac=10)
})

#SynapseGroup(net=net, tag='SE,GLU', src='exc_neurons', dst='inp_neurons', behavior={
#    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
#})

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

net.exc_neurons.sensitivity += 0.49

#net.exc_neurons.sensitivity += 0.3

#net['TextGenerator',0].plot_char_distribution()

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='ES')
    show_UI(net, sm)
else:
    #for i in range(20):
    #    train_ES_and_get_distribution_score(net, 5000, deactivateEE=True)

    #train_ES_and_get_distribution_score(net, 40000)#30000 #int(gene('T', 20000))
    train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)
