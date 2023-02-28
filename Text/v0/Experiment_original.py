from Helper import *
#from UI_Helper import *
from Text.v0.Behaviour_Core_Modules import *
from Text.v0.Behaviour_Text_Modules import *

ui = False
neuron_count = 2400

input_steps = 60000#30000
recovery_steps = 10000
free_steps = 5000

#grammar = get_char_sequence(5)     #Experiment A
#grammar = get_char_sequence(23)    #Experiment B
#grammar = get_long_text()          #Experiment C
grammar = get_random_sentences(3)    #Experiment D

input_density=0.92
target_activity = 1.0 / len(''.join(grammar))
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)

print("T", target_activity)
print("E", exc_output_exponent)
print("I", inh_output_slope)

net = Network(tag='Text Learning Network')

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={

    #9: Exception_Activator(), # use for manual text input with GUI code tab...

    # excitatory input
    10: TextGenerator(text_blocks=grammar),
    11: TextActivator(input_density=input_density, strength=1.0),#remove for non input tests
    12: SynapseOperation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_activity, strength=0.007),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=31, threshold=LI_threshold),
    41: STDP(transmitter='GLU', strength=0.0015),
    42: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    43: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent), #'[0.614#EXP]'

    # reconstruction
    80: TextReconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={

    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

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
    #net.exc_neurons.add_behaviour(200, Recorder(variables=['np.mean(n.output)']))
    #net.exc_neurons.add_behaviour(200, Recorder(variables=['np.mean(n.sensitivity)', 'n.iteration']))
    train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=sm)
    #plt.plot(net['n.iteration', 0], net['np.mean(n.sensitivity)', 0])
    #plt.show()
    #plot_output_trace(net['np.mean(n.output)', 0], input_steps, recovery_steps, net.exc_neurons.target_activity)

    '''
    vars = ['n.sensitivity', 'n._activity', 'n.output', 'n.input_GLU', 'n.input_GLUI', 'n.input_GABA', 'n.input_grammar']

    re=net.exc_neurons.add_behaviour(200, Recorder(variables=vars))
    ri=net.inh_neurons.add_behaviour(201, Recorder(variables=vars))

    
    for i in range(300+150):

        if i==300:
            net.deactivate_behaviours('STDP')
            net.deactivate_behaviours('Normalization')
            net.deactivate_behaviours('TextActivator')

        net.simulate_iterations(100)

        for v in vars:
            sm.save_np('exc_'+v+'_'+str(i), net.exc_neurons[v, 0])
            sm.save_np('inh_'+v+'_'+str(i), net.inh_neurons[v, 0])

        sm.save_np('EE_'+str(i), net['EE', 0].W)
        sm.save_np('IE_'+str(i), net['IE', 0].W)
        sm.save_np('EI_'+str(i), net['EI', 0].W)

        re.clear()
        ri.clear()
    

    #net.simulate_iterations(input_steps, 100)

    # deactivate Input
    #net.deactivate_behaviours('STDP')
    #net.deactivate_behaviours('Normalization')
    #net.deactivate_behaviours('TextActivator')

    #net.simulate_iterations(recovery_steps, 100)

    for i in range(45000):

        re.clear()
        ri.clear()

        if i==30000:
            net.deactivate_behaviours('STDP')
            net.deactivate_behaviours('Normalization')
            net.deactivate_behaviours('TextActivator')

        net.simulate_iteration()

        for v in vars:
            sm.save_np('exc_'+v+'_'+str(i), net.exc_neurons[v, 0])
            sm.save_np('inh_'+v+'_'+str(i), net.inh_neurons[v, 0])

        txt = net['TextReconstructor', 0].reconstruction_history[-20:]
        sm.save_param('txt_'+str(i), txt)

        #sm.save_np('EE_'+str(i), net['EE', 0].W)
        #sm.save_np('IE_'+str(i), net['IE', 0].W)
        #sm.save_np('EI_'+str(i), net['EI', 0].W)
    '''


