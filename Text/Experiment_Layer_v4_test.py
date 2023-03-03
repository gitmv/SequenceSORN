from PymoNNto import *
from Behaviour_Core_Modules_v4 import *
from Text.Behaviour_Text_Modules import *
from Helper import *

ui = False
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10

#grammar = get_random_sentences(4)
#target_act = 1/n_chars(grammar)

#curved 3s
#IP_s = gene('IP_s', 0.008735764741458582)
#avg_inh = gene('avg_inh', 0.3427857658747104)
#LI_s = gene('LI_s', 6.450234496564654)
#STDP_s = gene('STDP_s', 0.0030597477411211885)#important!#can be higher
#fe_exp = gene('fe_exp', 0.7378726012049153)#important!
#fe_mul = gene('fe_mul', 2.353594052973287)#important!

##curved 4s
##IP_s = gene('IP_s', 0.009263675397284607)
##avg_inh = gene('avg_inh', 0.3755245533873526)
##LI_s = gene('LI_s', 9.137740897721274)6.137740897721274
##STDP_s = gene('STDP_s', 0.006146842011975385)#important!
#fe_exp = gene('fe_exp', 0.5226260394007497)#important!
#fe_mul = gene('fe_mul', 1.849497255127404)#important!

#curved abcde._ / 7
grammar = ['abcde. ']
target_act = gene('T', 1/n_chars(grammar)/7)# #0.02
IP_s = gene('IP_s', 0.008973931059651436)
avg_inh = gene('avg_inh', 0.3231724475180935)
LI_s = gene('LI_s', 14.924004407474971)
STDP_s = gene('STDP_s', 0.018129208184866318)#important!
fe_exp = gene('fe_exp', 0.38183660298411154)#important!
fe_mul = gene('fe_mul', 1.1403125408175965)#important!
#{'IP_s': 0.008120058543820047, 'avg_inh': 0.34673989744399814, 'LI_s': 12.626213735551083, 'STDP_s': 0.01548284674215687, 'fe_exp': 0.38292203973626115, 'fe_mul': 1.1460953005896783}

#IP_s = 0.008735764741458582
#avg_inh = 0.3427857658747104
#LI_s = 10.0
#STDP_s = 0.018 #0.006 #0.018 is worse with 200norm! works with 50norm. works with 50norm


##fe_exp = gene('fe_exp', 0.7378726012049153)
##fe_mul = gene('fe_mul', 2.353594052973287)


net = Network(tag=ex_file_name(), settings=settings)

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=n_unique_chars(grammar), depth=1, centered=False), color=green, behaviour={
    # text input
    10: TextGenerator(iterations_per_char=1, text_blocks=grammar),
    11: TextActivator(strength=1),

    # group output
    50: Output_InputLayer(),

    # text reconstruction
    80: TextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behaviour={
    # weight normalization
    3: Normalization(tag='Norm', direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=50),#watch out when using higher STDP speeds or higher target activities!
    3.1: Normalization(tag='NormFSTDP', direction='afferent', syn_type='SOMA', exec_every_x_step=50),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=IP_s, init_sensitivity=0),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=LI_s, avg_inh=avg_inh, min=0.0), #-0.15 #(optional)(higher sccore/risky/higher spread)
    41: STDP(transmitter='GLU', strength=STDP_s),

    # group output
    50: Output_Excitatory(exp=fe_exp, mul=fe_mul),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh_neuros), color=red, behaviour={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(avg_inh=avg_inh, target_activity=target_act, duration=2),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behaviour={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behaviour={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1', behaviour={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1', behaviour={
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
    train_and_generate_text(net, input_steps=60000, recovery_steps=10000, free_steps=5000, sm=sm)
