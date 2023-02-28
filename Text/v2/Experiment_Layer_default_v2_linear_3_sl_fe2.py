from PymoNNto import *
from Text.v2.Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *
from Helper import *

class Output_Inhibitory_linear(Output_Inhibitory):

    def set_variables(self, neurons):
        self.duration = self.get_init_attr('duration', 2.0)
        self.slope = self.get_init_attr('slope', 14.3)

        self.avg_act = 0
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)

    def activation_function(self, a):
        return a*self.slope


class Output_Excitatory(Behaviour):

    def set_variables(self, neurons):
        self.exp = self.get_init_attr('exp', 1.5)
        self.mul = self.get_init_attr('mul', 2.0)
        self.act_mul = self.get_init_attr('act_mul', 0.0)
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)

    def activation_function(self, a):
        return np.power(np.clip(a*self.mul, 0.0, 1.0), self.exp)

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.get_neuron_vec("uniform") < self.activation_function(neurons.activity)
        neurons._activity = neurons.activity.copy() #for plotting
        if self.act_mul==0.0:
            neurons.activity.fill(0.0)
        else:
            neurons.activity *= self.act_mul

    def get_UI_Preview_Plots(self):
        x=np.arange(0,1,0.01)
        return [[x,self.activation_function(x)]]

ui = False
neuron_count = 2400

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D


target_act = 1/n_chars(grammar)

net = Network(tag='Layered Cluster Formation Network')


#because of 2x f_E scale synapse operation may be changed to "2x"
#I_s should stay the same
#first try without syn op adjustment
I_s = gene('I_s', 14.3)
E_exp = gene('E_exp', 0.60416)
LI_s = gene('LI_s', 27.699)
LI_t = gene('LI_t', 0.24685)

#no e_exp optimization:
#{'I_s': 4.753438112128659, 'LI_s': 48.22602872990575, 'LI_t': 0.07816609714212222} E_exp=0.60416
#=> much worse
#=> may need to add another

NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=orange, behaviour={
    #text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    #spike output
    50: Output_InputLayer(),

    #text interpretation
    80: TextReconstructor()
})


NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={

    # normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=10),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=10),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.00757, init_sensitivity=0.0),#0.49

    # learning
    40: LearningInhibition(transmitter='GABA', strength=LI_s, threshold=LI_t),
    41: STDP(transmitter='GLU', strength=0.002287),

    # output
    50: Output_Excitatory_2(exp=E_exp),
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    #65: IntrinsicPlasticity(target_activity=0.5, strength=0.00001, init_sensitivity=0.0),

    # output
    70: Output_Inhibitory_linear(slope=I_s, duration=2),
})

SynapseGroup(net=net, tag='GLU,ES,SOMA', src='inp_neurons', dst='exc_neurons', behaviour={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='GLU,EE,DISTAL', src='exc_neurons', dst='exc_neurons', behaviour={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='GLUI,IE', src='exc_neurons', dst='inh_neurons', behaviour={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='GABA,EI', src='inh_neurons', dst='exc_neurons', behaviour={
    1: CreateWeights()
})

sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='EE')
    Weight_Classifier_Pre(net.exc_neurons, syn_tag='ES')
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=60000, recovery_steps=10000, free_steps=5000, sm=sm)