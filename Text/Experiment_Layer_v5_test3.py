from PymoNNto import *
from Behavior_Core_Modules_v5 import *
from Behavior_Text_Modules_v5 import *
from Helper import *

ui = False
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10

grammar = get_random_sentences(3)
target_act = gene('h', 0.01923)#1/n_chars(get_random_sentences(3))

#third
#target_act = 0.006410256410256411#1/n_chars(grammar)/3
#set_genome({'IP_s': 0.020896679906752823, 'avg_inh': 0.26240211240528, 'LI_s': 4.217760789551765, 'STDP_s': 0.005206438619706144, 'fe_exp': 0.47499215627917557, 'fe_mul': 8.694893109383168})

#fifth
#target_act = 1/n_chars(grammar)/5
#set_genome({'IP_s': 0.024927716649829512, 'avg_inh': 0.19976460087122255, 'LI_s': 7.843941000823148, 'STDP_s': 0.006737234118296163, 'fe_exp': 0.2782225219900681, 'fe_mul': 20.521902971962692})


#curved 3s
IP_s = gene('IP_s', 0.008735764741458582)
avg_inh = gene('avg_inh', 0.3427857658747104)#gene('avg_inh', 0.5)#gene('avg_inh', 0.15) #gene('avg_inh', 0.25)
LI_s = gene('LI_s', 6.450234496564654)
STDP_s = gene('STDP_s', 0.0030597477411211885)#important!#can be higher
fe_exp = gene('fe_exp', 0.7378726012049153)#important!
fe_mul = gene('fe_mul', 2.353594052973287)#important!

net = Network(tag=ex_file_name(), settings=settings)

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behavior={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),

    # group output
    50: Output_TextActivator(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behavior={
    # weight normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),#watch out when using higher STDP speeds!
    3.1: Normalization(tag='NormFSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=IP_s, init_sensitivity=0),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=LI_s, avg_inh=avg_inh, min=-0.15), #(optional)(higher sccore/risky/higher spread)
    41: STDP(transmitter='GLU', strength=STDP_s),

    # group output
    51: Output_Excitatory(exp=fe_exp, mul=fe_mul),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh_neuros), color=red, behavior={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(avg_inh=avg_inh, target_activity=target_act, duration=2),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behavior={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1', behavior={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights()
})

sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#for i in range(1000):
#    result = net.simulate_iteration(True)
#print(result)

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=100000, recovery_steps=30000, free_steps=5000, sm=sm)
