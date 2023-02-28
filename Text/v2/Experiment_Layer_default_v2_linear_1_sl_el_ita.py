from PymoNNto import *
from Text.v2.Behaviour_Core_Modules import *
from Text.Behaviour_Text_Modules import *
from Helper import *





def f_i(x, ci):
    return np.tanh(x * ci)

def f_i_derivative(x, ci):
    return (4 * ci) / np.power(np.exp(-ci * x) + np.exp(ci * x), 2)

def fi_2(x, ci, px):
    fx = f_i(px, ci)
    fdx = f_i_derivative(px, ci)
    print(fx-px*fdx)
    print(fdx)
    return fx + (x - px) * fdx

def fi_3(x, ci, px):
    fdx = f_i_derivative(px, ci)
    return x * fdx

def fi_4(x):
    return x * 15.5

def fe(x, ce):
    return np.power(np.abs(x - 0.5) * 2, ce) * (x > 0.5)


def f_e_derivative(x, ce):
    return 2 * ce * np.power(2 * x - 1, ce - 1)


#print(f_i(0.019230769230769232, 15.5))

#x=np.arange(0,0.1,0.0001)
#plt.plot(x, fi_2(x, ci, px))
#plt.plot(x, fi_4())
#plt.plot(x, fe((x*5+0.5), 0.60416))
#plt.plot(x, f_i(x, 15.5))
#plt.plot(x, fi_2(x, 15.5, 0.018375))
#plt.plot(x, x*14.3+0.0144499)
#plt.plot(x, fi_3(x, 15.5, 0.018375))
#plt.show()


#x=np.arange(0,1.0,0.0001)


#for xp in x:
#    print(xp, f_e_derivative(xp, 0.60416)*(xp-0.5)/(0.01923-0))

#e=0.01445
#plt.plot(x, x*14.3+e)

#e=0.3 - 14.3 * 0.01923
#plt.plot(x, x*14.3+e)

#e=0.3 - 14.3 * 0.04
#plt.plot(x, x*14.3+e)

#plt.show()

class Output_Inhibitory_linear(Output_Inhibitory):

    def set_variables(self, neurons):
        self.duration = self.get_init_attr('duration', 2.0)
        self.slope = self.get_init_attr('slope', 14.3)
        self.elevation = self.get_init_attr('elevation', 0.01445)

        self.exc_target_activity = self.get_init_attr('exc_target_activity', 0.01923)

        self.elevation = 0.3 - self.slope * self.exc_target_activity #ensures that f_i(h) = 0.3 (without changing the slope or the excitatory activation function)

        self.avg_act = 0
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)

    def activation_function(self, a):
        return a*self.slope+self.elevation
        #with elevation

ui = True
neuron_count = 2400

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D


target_act = 1/n_chars(grammar)
#print(1/n_chars(grammar)) #0.019230769230769232

net = Network(tag='Layered Cluster Formation Network')

#{'I_s': 17.241606021270513, 'I_e': 0.014088011010074652, 'E_exp': 0.6652364100123466, 'LI_s': 29.74304996287139, 'LI_t': 0.26162831683840376}
#{'I_s': 16.212649140371113, 'I_e': 0.016899117716635513, 'E_exp': 0.5792594327877636, 'LI_s': 26.152757012758293, 'LI_t': 0.25888164188164786}

I_s = gene('I_s', 14.3)
I_e = gene('I_e', 0.0144499)
E_exp = gene('E_exp', 0.60416)
LI_s = gene('LI_s', 27.699)
LI_t = gene('LI_t', 0.24685)


NeuronGroup(net=net, tag='inp_neurons', size=NeuronDimension(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=orange, behaviour={

    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    50: Output_InputLayer(),

    80: TextReconstructor()
})


NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    # normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=10),
    3.1: Normalization(tag='Norm_FSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=10),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.00757, init_sensitivity=0.49),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=LI_s, threshold=LI_t),
    41: STDP(transmitter='GLU', strength=0.002287),

    # output
    50: Output_Excitatory(exp=E_exp),
})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    #65: IntrinsicPlasticity(target_activity=0.5, strength=0.00001, init_sensitivity=0.0),

    # output
    70: Output_Inhibitory_linear(slope=I_s, elevation=I_e, duration=2, exc_target_activity=target_act),
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