from PymoNNto import *
from Grammar.SORN_Grammar.Behaviours_in_use import *

#set_genome({'GABAE': 1.1057403610397172, 'IP': 0.007223298098309225, 'LIM': 195.9931369624642, 'STDP': 0.001025817650992625, 'GLUI': 17.450647503029778, 'PO': 1.0507630664072505, 'IED': 0.48737926313028035, 'gen': 38.0, 'score': 9.39619585134594})

ui = False
neuron_count = 2400
plastic_steps = 30000#50000
recovery_steps = 10000

SORN = Network(tag='SORN')

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behaviour={
    #init
    1: Init_Neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    15: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']),#, ' man drives car.', ' plant loves rain.', ' parrots can fly.', 'the fish swims' #
    16: Text_Activator(input_density=0.04, strength=0.75),#1.0
    18: Synapse_Operation(transmitter='GLU', strength=1.0),
    19: Synapse_Operation(transmitter='GABA', strength='-[1.0#GABAE]'),

    #stability
    21: IP(sliding_window=0, speed='[0.007#IP]'),
    22: Refractory_D(steps=4.0),

    #output
    30: ReLu_Output_Prob(),

    #learning
    41: Buffer_Variables(),#for STDP
    #41.5: Learning_Inhibition(transmitter='GABA', strength=-2),
    41.5: Learning_Inhibition_mean(strength='-[200#LIM]'),
    42: STDP_C(transmitter='GLU', eta_stdp='[0.0015#STDP]', STDP_F={-1: 1}),
    45: Normalization(syn_type='GLU', exec_every_x_step=1),

    #reconstruction
    50: Text_Reconstructor(),
})

inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
    2: Init_Neurons(),
    31: Synapse_Operation(transmitter='GLU', strength='[10.0#GLUI]'),#approximately: (mean_e+oscillation_e)*10.0=(0.02+0.06)*10=0.8 (nearly 1)
    32: Power_Output(exp='[2.0#PO]'),
    #32: Power_Output_Prob(exp='[2.0#PO]'),
    #32: ID_Output(),
    #32: ReLu_Output(),
})

SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,IE', behaviour={
    3: create_weights(distribution='uniform(0.9,1.0)', density='[0.5#IED]')
})

SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,EI', behaviour={
    3: create_weights(distribution='uniform(0.9,1.0)', density=1.0)#0.9
})


#inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
    #init
#    2: Init_Neurons(),

    #input!
#    31: Synapse_Operation(transmitter='GLU', strength=30),

    #output!
    #14: Threshold_Output(threshold='uniform(0.1,0.9)'),
#    32: ReLu_Output(),
#})

SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,EE', behaviour={
    #init
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto'),
    3: create_weights(distribution='lognormal(1.0,0.6)', density=0.9)
})

#SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,IE', behaviour={
#    3: create_weights(distribution='uniform(0.9,1.0)', density=0.5)
#})

#SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,EI', behaviour={
#    3: create_weights(distribution='uniform(0.9,1.0)', density=0.9)
#})


sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)

SORN.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    exc_neurons.color = blue
    inh_neurons.color = red
    show_UI(SORN, sm, 2)


#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_mechanisms('STDP')
SORN.deactivate_mechanisms('Text_Activator')

#recovery phase
SORN.simulate_iterations(recovery_steps, 100)

#text generation
SORN['Text_Reconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print(SORN['Text_Reconstructor', 0].reconstruction_history)

#scoring
score = SORN['Text_Generator', 0].get_text_score(SORN['Text_Reconstructor', 0].reconstruction_history)
set_score(score, sm)



#SORN.simulate_iterations(1500, 100)

#score = 1-np.mean(np.abs(SORN['np.mean(n.output)', 0, 'np'][1000:]-0.02))

#set_score(score, sm)

#30: Threshold_Output(threshold=0.5),

# 23: NOX_Diffusion(th_nox=0.0, strength=1.0),
# 24: isi_reaction_module(strength=0.1),
# 25: random_activity_simple(rate=0.001),
# 100: STDP_Analysis(),

# 100: Recorder(tag='avg_rec', variables=['np.mean(n.output)']),

#SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU_cluster,syn', behaviour={
#    1: Box_Receptive_Fields(range=18, remove_autapses=True),
#    2: Partition(split_size='auto')
#})


#set_genome({'gs': 0.381261212447333, 'IP_sigma': 0.24850875372613956, 'eta_ip': 3.554892196403195, 'eta_stdp': 4.4771428322595506e-05, 'igs': 9.23610086304308, 'it': 0.06842236713342817, 'gen': 163.0, 'score': 0.9977216666666666})
    #{'gs': 0.45843287191962945, 'IP_sigma': 0.40237164995106556, 'eta_ip': 1.696521778172585, 'eta_stdp': 7.45870734695293e-05, 'igs': 14.305182506267165, 'it': 0.07825416873911004, 'gen': 129.0, 'score': 0.9968291666666667})
    #{'gs': 0.43051966037057793, 'IP_sigma': 1.0905837218902503, 'eta_ip': 0.5723874590899439, 'eta_stdp': 9.584398083167788e-05, 'igs': 10.113515129658017, 'it': 0.04290661366908447, 'gen': 89.0, 'score': 0.9903166666666666})


# 23: IP(h_ip='lognormal_real_mean(0.02, [0.2944#IP_sigma])', eta_ip='0.007'),#[0.07#eta_ip]

# 3: init_afferent_synapses(transmitter='GLU', density='1%', distribution='uniform([0.1#iglumin],1.0)', normalize=True),
# 8: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.']),#, ' parrots can fly.', 'the fish swims'
# 9: Text_Activator(input_density=0.04),#0.03#0.04#0.043#0.015#0.043#0.015
# 15: buffer_variables(),#for STDP
# 18: refrac(decayfactor=0.5),
# 20: IP(h_ip='lognormal_real_mean([0.02#k], [0.2944#IP_sigma])', eta_ip='[0.007#eta_ip]'),#
# 21: STDP_complex(transmitter='GLU', eta_stdp='[0.00015#eta_stdp]', STDP_F={-1: 1}),#{-1: 0.2, 1: -1} #, 1: -1

#2: init_neuron_variables(),
# 3: init_afferent_synapses(transmitter='GLU', density='full', distribution='lognormal(1.0,0.6)', normalize=True),
# 3.1: init_afferent_synapses(transmitter='GABA', density='full', distribution='uniform(0.9,1.0)', normalize=True),

#learning
#SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
#SORN.deactivate_mechanisms('STDP')
#SORN.deactivate_mechanisms('Text_Activator')

#recovery phase
#SORN.simulate_iterations(5000, 100)

#text generation
#SORN['Text_Reconstructor', 0].reconstruction_history = ''
#SORN.simulate_iterations(5000, 100)
#recon_text = SORN['Text_Reconstructor', 0].reconstruction_history
#print(recon_text)

#scoring
#score = SORN['Text_Generator', 0].get_text_score(recon_text)
#set_score(score, sm)


