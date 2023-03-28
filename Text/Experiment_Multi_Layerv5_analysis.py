from Behavior_Core_Modules_v5 import *
from Behavior_Text_Modules_v5 import *
from Helper import *


ui = False
layer_sizes = [2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400]

grammar = get_random_sentences(3)
target_act = 1/n_chars(grammar)


def add_layer(net, layer, layer_size):
    e_color = (0.0, 0.0, 255.0-(layer-1)*70, 255.0)
    i_color = (255.0-(layer-1)*70, 0.0, 0.0, 255.0)

    NeuronGroup(net=net, tag='exc_neurons'+str(layer), size=getGrid(layer_size), color=e_color, behavior={
        # normalization
        3: Normalization(direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),
    }|({3.1: Normalization(direction='afferent', syn_type='SOMA', exec_every_x_step=200)} if layer==1 else {})|{ #only add behavior for first layer

        # excitatory and inhibitory input
        12: SynapseOperation(transmitter='GLU', strength=1.0),
        20: SynapseOperation(transmitter='GABA', strength=-1.0),

        # stability
        30: IntrinsicPlasticity(target_activity=target_act, strength=0.0087357, init_sensitivity=0),

        # learning
        40: LearningInhibition(transmitter='GABA', strength=6.450234, avg_inh=0.342785, min=-0.15),# (optional)(higher sccore/risky/higher spread)
        41: STDP(transmitter='GLU', strength=0.00305974),

        # group output
        50: Output_Excitatory(exp=0.7378726, mul=2.35359),
    })

    NeuronGroup(net=net, tag='inh_neurons'+str(layer), size=get_squared_dim(layer_size/10), color=i_color, behavior={
        # excitatory input
        60: SynapseOperation(transmitter='GLUI', strength=1.0),

        # group output
        70: Output_Inhibitory(avg_inh=0.3427857, target_activity=target_act, duration=2),
    })

    SynapseGroup(net=net, tag='GLU,DISTAL', src='exc_neurons' + str(layer), dst='exc_neurons' + str(layer), behavior={
        1: CreateWeights(normalize=False)
    })

    SynapseGroup(net=net, tag='GLUI', src='exc_neurons' + str(layer), dst='inh_neurons' + str(layer), behavior={
        1: CreateWeights()
    })

    SynapseGroup(net=net, tag='GABA', src='inh_neurons' + str(layer), dst='exc_neurons' + str(layer), behavior={
        1: CreateWeights()
    })


def add_connection(net, src_nr, dst_nr):
    SynapseGroup(net=net, tag='GLU,DISTAL', src='exc_neurons' + str(src_nr), dst='exc_neurons' + str(dst_nr), behavior={
        1: CreateWeights(normalize=False)  # , nomr_fac=10
    })



net = Network(tag=ex_file_name(), settings=settings)


#add main layers and connect each to the previous one
for i, size in enumerate(layer_sizes):
    add_layer(net, i+1, size)

    if i>0:
        add_connection(net, i, i + 1) #forward
        #add_connection(net, i + 1, i) #backward

#add 1->4 and 2->4 connections
for i in range(len(layer_sizes)-2):
    add_connection(net, i + 1, 4) #forward
    #add_connection(net, 4, i + 1) #backward



#add input layer
NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behavior={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),

    # group output
    50: Output_TextActivator(),

    # text reconstruction
    80: TextReconstructor()
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behavior={
    1: CreateWeights(nomr_fac=10)
})


sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui:
    from UI_Helper import *
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=500000, recovery_steps=10000, free_steps=5000, sm=sm)

from PymoNNto.Exploration.HelperFunctions.Save_Load import *

save_network(net, 'ff_multi_7_layer_test_500k')


