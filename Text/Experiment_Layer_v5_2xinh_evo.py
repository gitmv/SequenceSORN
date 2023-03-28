import matplotlib.pyplot as plt
from PymoNNto import *
from Behavior_Core_Modules_v5 import *
from Behavior_Text_Modules_v5 import *
from Helper import *

ui = False
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/gene('inh', 10)

grammar = get_random_sentences(3)
target_act = 1/n_chars(grammar)

#/5
#set_genome({'IP_s': 0.009320036486471873, 'avg_inh': 0.33016831291904974, 'LI_s': 7.193192999273828, 'STDP_s': 0.0032803247609018847, 'fe_exp': 0.6997323566882255, 'fe_mul': 2.397297327230222})

#/2
set_genome({'IP_s': 0.010773019350417714, 'avg_inh': 0.3128008390762195, 'LI_s': 11.537244801880927, 'STDP_s': 0.0032778007751636216, 'fe_exp': 0.6010524568195746, 'fe_mul': 2.717563437525031})

#curved 3s
IP_s = gene('IP_s', 0.008735764741458582)
avg_inh = gene('avg_inh', 0.3427857658747104)
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

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=60000, recovery_steps=15000, free_steps=5000, sm=sm)

    '''
    net.deactivate_behaviors('TextReconstructor')

    net.exc_neurons1.add_behavior(1111, Recorder('np.mean(output)'), True)
    net.inh_neurons1.add_behavior(1112, Recorder('np.mean(output)'), True)

    net.simulate_iterations(60000)

    eo = net.exc_neurons1['np.mean(output)', 0, 'np']
    io = net.inh_neurons1['np.mean(output)', 0, 'np']

    ey = []
    iy = []
    for i in range(net.iteration-1000):
        ey.append(np.var(eo[i:i + 1000]))
        iy.append(np.var(io[i:i + 1000]))

    ey = np.array(ey)
    iy = np.array(iy)

    np.save('ey10.npy', arr=ey)
    np.save('iy10.npy', arr=iy)
    '''

