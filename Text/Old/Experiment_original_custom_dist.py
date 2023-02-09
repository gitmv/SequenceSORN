from PymoNNto import *
from Helper import *
from UI_Helper import *
from Text.New2.Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *

class Text_Activator_custom(Text_Activator):

    def set_variables(self, neurons):
        self.add_tag('Text_Activator')
        self.text_generator = neurons['Text_Generator', 0]

        input_density = self.get_init_attr('input_density', 0.5)
        activation_size = np.floor((neurons.size * input_density) / len(self.text_generator.alphabet)) #average char cluster size

        neurons.mean_network_activity = activation_size / neurons.size  # optional/ can be used by other (homeostatic) modules

        if self.get_init_attr('char_weighting', True):
            cw = self.text_generator.char_weighting
        else:
            cw = None

        print(cw)

        ccw = self.get_init_attr('custom_cw', None)
        if ccw is not None:
            ccw = np.array(ccw)
            i = [0, 6, 9, 1, 14, 18, 2, 4, 11, 15, 19, 20, 3, 5, 7, 8, 10, 12, 13, 16, 17, 21, 22]#resort sorted distribution
            cw[i] = ccw
            cw = cw/np.sum(cw)*len(self.text_generator.alphabet)

        print(cw)

        neurons.Input_Weights = self.one_hot_vec_to_neuron_mat(len(self.text_generator.alphabet), neurons.size, activation_size, cw)
        neurons.Input_Mask = np.sum(neurons.Input_Weights, axis=1) > 0

        neurons.input_grammar = neurons.get_neuron_vec()

        self.strength = self.get_init_attr('strength', 1, neurons)



ui = True
neuron_count = 2400

input_steps = 30000
recovery_steps = 10000
free_steps = 5000

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D

input_density=0.92
target_activity = 1.0 / len(''.join(grammar))
exc_output_exponent = 0.01 / target_activity + 0.22 #gene('E', 0.55)
inh_output_slope = 0.4 / target_activity + 3.6 #gene('I', 18.222202430254626)
LI_threshold = np.tanh(inh_output_slope * target_activity) #gene('L', 0.31)

net = Network(tag='Text Learning Network')

custom_cw = [367, 296, 252, 147, 130, 126, 114,  99,  99,  94,  83,  71,  65, 63,  50,  46,  46,  46,  44,  41,  41,  41,  39]

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={

    #9: Exception_Activator(), # use for manual text input with GUI code tab...



    # excitatory input
    10: Text_Generator(text_blocks=grammar),
    11: Text_Activator_custom(input_density=input_density, strength=1.0, custom_cw=custom_cw),#remove for non input tests
    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
    41: STDP(transmitter='GLU', strength=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent), #'[0.614#EXP]'

    # reconstruction
    80: Text_Reconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),

    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2),

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

#User interface
if __name__ == '__main__' and ui:
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    #add_all_analysis_modules(net['exc_neurons', 0])
    show_UI(net, sm)
else:
    net.exc_neurons.add_behaviour(200, Recorder(variables=['np.mean(n.output)']))
    train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=sm)
    plot_output_trace(net['np.mean(n.output)', 0], input_steps, recovery_steps, net.exc_neurons.target_activity)
