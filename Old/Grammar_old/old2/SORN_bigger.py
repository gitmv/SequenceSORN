from Old.Grammar._common import *

#set_genome({'GABAE': 1.853888950370442, 'IP': 0.00907738308945769, 'LIM': 341.17623451186614, 'STDP': 0.0006437913343690012, 'GLUI': 23.512299265794823, 'PO': 4.270647892248294, 'IED': 0.5524304942357359, 'gen': 199.0, 'score': 6.954308498976853})

set_genome({'IS': 0.8665002381794471, 'IP': 0.006654017643968699, 'S': 5.025925897276355, 'D': 1.5961401737386358, 'exp': 0.81688717995202, 'LIM': 176.98109110411784, 'STDP': 0.0015246122509244239, 'gen': 20.0, 'score': 6.9706985526873595})


ui = True
neuron_count = 3600
plastic_steps = 60000#50000
recovery_steps = 10000#10000#1000

load_learned_state = False

SORN = Network(tag='SORN_big')

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={
    #init
    1: Init_Neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    #15: Line_Patterns(center_x=20, center_y=np.arange(10,30), degree=0, line_length=30),

    #15: MNIST_Patterns_On_Off(center_x=30, center_y=30, pattern_count=10),

    15: TextGenerator(text_blocks=get_default_grammar(3)),
    16: TextActivator(input_density=96, strength=0.75),#1.0
    18: SynapseOperation(transmitter='GLU', strength=1.0),
    #19: SynapseOperation(transmitter='GABA', strength='-[1.0#GABAE]'),

    #stability
    21: IP(sliding_window=0, speed='[0.007#IP]'),
    22: Refractory_D(steps=4.0),

    # 23: inhibition_test(strength=4.5),
    24: inhibition_test_long(strength='[4.0#S]', duration='[2#D]'),

    #output
    #30: ReLu_Output_Prob(),
    30: variable_slope_relu_exp(exp='[1.0#exp]'),
    #30: Mem_Noise_Output_Prob_Triangular(tr_left=-0.5),

    #learning
    41: Buffer_Variables(),#for STDP
    #41.5: LearningInhibition(transmitter='GABA', strength=-2),
    41.5: LearningInhibition_mean(strength='-[200#LIM]'),
    42: STDP_C(transmitter='GLU', eta_stdp='[0.0015#STDP]', STDP_F={-1: 1}),
    45: Normalization(syn_type='GLU'),#, exec_every_x_step=1

    #reconstruction
    50: TextReconstructor(),
})

#inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={
#    2: Init_Neurons(),
#    31: SynapseOperation(transmitter='GLU', strength='[10.0#GLUI]'),#approximately: (mean_e+oscillation_e)*10.0=(0.02+0.06)*10=0.8 (nearly 1)
#    32: Power_Output(exp='[2.0#PO]'),
#    #32: Power_Output_Prob(exp='[2.0#PO]'),
#    #32: ID_Output(),
#    #32: ReLu_Output(),
#})

#SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,IE', behaviour={
#    #3: create_weights(distribution='uniform(0.9,1.0)', density='[0.5#IED]')
#    3: create_weights(distribution='uniform(1.0,1.0)', density='[1.0#IED]')
#})

#SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,EI', behaviour={
#    #3: create_weights(distribution='uniform(0.9,1.0)', density=1.0)#0.9
#    3: create_weights(distribution='uniform(1.0,1.0)', density=1.0)#0.9
#})


#inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
    #init
#    2: Init_Neurons(),

    #input!
#    31: SynapseOperation(transmitter='GLU', strength=30),

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
    show_UI(SORN, sm)
else:
    train_and_generate_text(SORN, plastic_steps, recovery_steps, sm=sm)































#set_genome({'GABAE': 1.1057403610397172, 'IP': 0.007223298098309225, 'LIM': 195.9931369624642, 'STDP': 0.001025817650992625, 'GLUI': 17.450647503029778, 'PO': 1.0507630664072505, 'IED': 0.48737926313028035, 'gen': 38.0, 'score': 9.39619585134594})
#set_genome({'GABAE': 1.171293350006476, 'IP': 0.007284819283376444, 'LIM': 159.03562916675344, 'STDP': 0.001081502318114489, 'GLUI': 11.505736726725017, 'PO': 1.6664744313778532, 'IED': 0.45052399449750585, 'gen': 23.0, 'score': 6.866036947462384})
#{'GABAE': 1.0, 'IP': 0.007, 'LIM': 200.0, 'STDP': 0.0015, 'GLUI': 10.0, 'PO': 2.0, 'IED': 0.5}
#set_genome({'GABAE': 1.118021772738199, 'IP': 0.008224797756181379, 'LIM': 208.7539498796273, 'STDP': 0.0008528586868285881, 'GLUI': 17.460488750338744, 'PO': 2.2872364608557265, 'IED': 0.43959733787779914, 'gen': 62.0, 'score': 6.955774245055846})


'''

if load_learned_state:
    subfolder='60ks_stable'

    folder = get_data_folder(create_when_not_found=True)+'/'+subfolder+'/'

    set_partitioned_synapse_matrix(SORN['exc_neurons',0], 'EE', 'W', np.load(folder+'_EEw.npy'))
    set_partitioned_synapse_matrix(SORN['exc_neurons',0], 'EE', 'enabled', np.load(folder+'_EEe.npy'))
    #network['EE',0].W = np.load(folder+'_EEw.npy')
    #network['EE',0].enabled = np.load(folder+'_EEe.npy')
    #SORN['IE',0].W = np.load(folder+'_IEw.npy')
    #SORN['IE',0].enabled = np.load(folder+'_IEe.npy')
    #SORN['EI',0].W = np.load(folder+'_EIw.npy')
    #SORN['EI',0].enabled = np.load(folder+'_EIe.npy')
    SORN['exc_neurons', 0].target_activity = np.load(folder+'_ENt.npy')
    SORN['exc_neurons', 0].Input_Weights = np.load(folder+'_ENw.npy')
    SORN['exc_neurons', 0].Input_Mask = np.load(folder+'_ENm.npy')
    SORN['exc_neurons', 0].sensitivity = np.load(folder+'_ENs.npy')

    # deactivate STDP and Input
    SORN.deactivate_mechanisms('STDP')
    SORN.deactivate_mechanisms('Normalization')
    SORN.deactivate_mechanisms('TextActivator')
    
    

#plot_corellation_matrix(SORN)

if not load_learned_state:
    # learning
    SORN.simulate_iterations(plastic_steps, 100)

    # deactivate STDP and Input
    SORN.deactivate_mechanisms('STDP')
    SORN.deactivate_mechanisms('Normalization')
    SORN.deactivate_mechanisms('TextActivator')

#recovery phase
SORN.simulate_iterations(recovery_steps, 100)

#text generation
tr = SORN['TextReconstructor', 0]
tr.reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
print(tr.reconstruction_history)

#scoring
score = SORN['TextGenerator', 0].get_text_score(tr.reconstruction_history)
set_score(score, info={'text': tr.reconstruction_history, 'simulated_iterations':SORN.iteration})




'''









#import matplotlib.pyplot as plt

#plt.matshow(get_partitioned_synapse_matrix(exc_neurons, 'EE', 'enabled'))
#plt.show()

#np.save('all_mat_test', get_combined_partition_matrix(exc_neurons, 'EE', 'W'))
#np.save('neuron_mat_test', get_single_neuron_combined_partition_matrix(exc_neurons, 'EE', 'W', 1000))

#all = np.load('all_mat_test.npy')
#neu = np.load('neuron_mat_test.npy')

#set_partitioned_synapse_matrix(exc_neurons, 'EE', 'W', all)
#set_partitioned_single_neuron_weights(exc_neurons, 'EE', 'W', 1000, neu)

#plt.matshow(get_partitioned_synapse_matrix(exc_neurons, 'EE', 'W'))
#plt.show()



#SORN.simulate_iterations(1500, 100)

#score = 1-np.mean(np.abs(SORN['np.mean(n.output)', 0, 'np'][1000:]-0.02))

#set_score(score)

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
# 8: TextGenerator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.']),#, ' parrots can fly.', 'the fish swims'
# 9: TextActivator(input_density=0.04),#0.03#0.04#0.043#0.015#0.043#0.015
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
#SORN.deactivate_mechanisms('TextActivator')

#recovery phase
#SORN.simulate_iterations(5000, 100)

#text generation
#SORN['TextReconstructor', 0].reconstruction_history = ''
#SORN.simulate_iterations(5000, 100)
#recon_text = SORN['TextReconstructor', 0].reconstruction_history
#print(recon_text)

#scoring
#score = SORN['TextGenerator', 0].get_text_score(recon_text)
#set_score(score)


