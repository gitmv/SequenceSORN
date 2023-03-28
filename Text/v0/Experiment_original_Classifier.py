from PymoNNto import *
from Helper import *
from UI_Helper import *
from Behavior_Modules import *
from Behavior_Text_Modules import *
from Old.Grammar_old.Bruno.Logistic_Regression_Reconstruction import *

ui = False
neuron_count = 2400

plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

train_steps = 5000

def grammar_text_blocks_simple():
    verbs = ['eats ',
                  'drinks ']

    subjects = [
                     ' boy ',
                     ' cat ',
                     ' dog ',
                     ' fox ']


    objects_eat = ['meat.',
                        'bread.',
                        'fish.',]

    objects_drink = ['milk.',
                          'water.',
                          'juice.',]

    results=[]
    for s in subjects:
        for oe in objects_eat:
            results.append(s + verbs[0] + oe)

        for od in objects_drink:
            results.append(s + verbs[1] + od)

    return results

#grammar = get_char_sequence(5)     #A
#grammar = get_char_sequence(23)    #B
#grammar = get_long_text()          #C
#grammar = get_default_grammar(3)    #D
grammar = grammar_text_blocks_simple()
#grammar = get_reber_text(10)

input_density=0.92#7#92
target_activity = 0.02#1.0 / len(''.join(grammar))# * input_density 0.066666#
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

net = Network(tag='Grammar Learning Network')

class Exception_Activator(Behavior):

    #activate with following command
    #neurons['Exception_Activator', 0].txt='abcdefg'
    #neurons['Exception_Activator', 0].text_position = 0

    def initialize(self, neurons):
        self.txt = ' exception exception. exception exception. exception exception.'
        self.text_position = -1

    def iteration(self, neurons):
        if self.text_position>=0 and self.text_position<len(self.txt):
            neurons['TextGenerator', 0].next_char = self.txt[self.text_position]
            self.text_position += 1

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behavior={

    9: ExceptionActivator(),

    # excitatory input
    10: TextGenerator(text_blocks=grammar),
    11: TextActivator(input_density=input_density, strength=1.0),#0.04 #92%

    11.5: ClassifierTextReconstructor(),

    12: SynapseOperation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_activity, strength=0.007), #0.02 #'[0.02#TA]'

    # learning
    40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.02))
    41: STDP(transmitter='GLU', strength=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: GenerateOutput(exp=exc_output_exponent), #'[0.614#EXP]'

    # reconstruction
    80: TextReconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behavior={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # output
    70: GenerateOutput_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'

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

#User interface
if __name__ == '__main__' and ui:
    show_UI(net, sm)
else:
    #net.add_behaviors_to_object({200: Recorder(variables=['np.mean(n.output)'])}, net.exc_neurons)

    #train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)

    #plot_output_trace(net['np.mean(n.output)', 0], plastic_steps, recovery_steps, net.exc_neurons.target_activity)

    # learning
    net.simulate_iterations(plastic_steps, 100)

    # deactivate STDP and Input
    net.deactivate_behaviors('STDP')
    net.deactivate_behaviors('Normalization')
    # SORN.deactivate_mechanisms('TextActivator')
    net['ClassifierTextReconstructor', 0].start_recording()

    net.simulate_iterations(train_steps, 100)
    net.deactivate_behaviors('TextActivator')
    net['ClassifierTextReconstructor', 0].train()  # starts activating after training/stops recording automatically

    # import matplotlib.pyplot as plt
    # plt.matshow(SORN['ClassifierTextReconstructor', 0].classifier.coef_[:, 0:200])
    # with bias:
    # np.hstack((clf.intercept_[:,None], clf.coef_))
    # plt.show()

    net.simulate_iterations(text_gen_steps, 100)
    print(net['ClassifierTextReconstructor', 0].reconstruction_history)

    # scoring
    # score = SORN['TextGenerator', 0].get_text_score(recon_text)
    # set_score(score)
