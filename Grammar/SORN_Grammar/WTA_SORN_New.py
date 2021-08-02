import numpy as np
import random

#np.random.seed(1)
#random.seed(1)

from Grammar.SORN_Grammar.Behaviours.SORN_experimental import *
from Grammar.SORN_Grammar.Behaviours.SORN_WTA import *
from Grammar.SORN_Grammar.Behaviours_New.Text_Generation import *
from Grammar.SORN_Grammar.Behaviours_New.Text_Activator import *
from Grammar.SORN_Grammar.Behaviours_New.Text_Reconstructor import *
from Grammar.Common.Grammar_Helper import *


ui = False
neuron_count = 2400
plastic_steps = 30000

sm = StorageManager('test', random_nr=True, print_msg=True)

SORN = Network()

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behaviour={
    2: init_neuron_variables(),
    3: init_afferent_synapses(transmitter='GLU', density='full', distribution='lognormal(1.0,0.6)', normalize=True),#uniform(0.1,1.0)

    8: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.']),#, ' parrots can fly.', 'the fish swims'
    9: Text_Activator(input_density=0.04),#0.03#0.04#0.043#0.015#0.043#0.015

    10: synapse_operation(transmitter='GLU', strength='1.0'),
    11: IP_apply(),
    #12: refrac_apply(strengthfactor='[0.1#rs]'),
    13: K_WTA_output_local(partition_size=7, K='[0.02#k]'),#0.03 #0.01

    15: buffer_variables(),#for STDP

    #18: refrac(decayfactor=0.5),
    20: IP(h_ip='lognormal_real_mean([0.02#k], [0.2944#IP_sigma])', eta_ip='[0.007#eta_ip]'),#
    20.1: exhaustion_same_mean(),
    21: STDP_complex(transmitter='GLU', eta_stdp='[0.00015#eta_stdp]', STDP_F={-1: 1}),#{-1: 0.2, 1: -1} #, 1: -1
    22: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),

    40: Text_Reconstructor(),

     #3.1: init_afferent_synapses(transmitter='GLU_cluster', density='90%', distribution='uniform(0.1,1.0)', normalize=True),

     #14.1: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
     #14.2: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

     #14.3: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
     #14.4: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

     #14.5: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
     #14.6: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

     #21.2: STDP_complex(transmitter='GLU_cluster', eta_stdp='[0.00015#STDP_eta]', STDP_F={0: 1}),#{0: 2.0}
     #22.2: Normalization(syn_type='GLU_cluster', behaviour_norm_factor=0.3),
})

SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU,syn', behaviour={
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto')
})

#SynapseGroup(net=SORN, src=exc_neurons, dst=exc_neurons, tag='GLU_cluster,syn', behaviour={
#    1: Box_Receptive_Fields(range=18, remove_autapses=True),
#    2: Partition(split_size='auto')
#})

SORN.initialize(info=True, storage_manager=sm)


#print(SORN['Text_Generator',0].get_max_score())



#User interface
if __name__ == '__main__' and ui:
    from PymoNNto.Exploration.Network_UI import *
    from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
    exc_neurons.color = get_color(0, 1)
    my_modules = get_default_UI_modules(['output', 'exhaustion_value', 'weight_norm_factor'])+get_my_default_UI_modules()# , 'nox', 'refractory_counter'
    my_modules[0] = UI_sidebar_activity_module(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})
    my_modules[8] = single_group_plot_tab({'activity': (0, 0, 0), 'excitation': (0, 0, 255), 'inhibition': (255, 0, 0), 'input_act': (255, 0, 255),'exhaustion_value': (0, 255, 0)})
    Network_UI(SORN, modules=my_modules, label='SORN K_WTA', storage_manager=sm, group_display_count=1, reduced_layout=False).show()


#learning
SORN.simulate_iterations(plastic_steps, 100)

#deactivate STDP and Input
SORN.deactivate_mechanisms('STDP')
SORN.deactivate_mechanisms('Text_Activator')

#recovery phase
SORN.simulate_iterations(5000, 100)

#text generation
SORN['Text_Reconstructor', 0].reconstruction_history = ''
SORN.simulate_iterations(5000, 100)
recon_text = SORN['Text_Reconstructor', 0].reconstruction_history
print(recon_text)

#scoring
score = SORN['Text_Generator', 0].get_text_score(recon_text)
set_score(score, sm)


