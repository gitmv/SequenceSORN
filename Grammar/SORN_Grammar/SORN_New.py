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
from Grammar.SORN_Grammar.Behaviours_New.Nox import *
from Grammar.SORN_Grammar.Behaviours_New.Refractory import *

from Grammar.SORN_Grammar.Behaviour_New_2NGS.threshold_output import *

ui = True
neuron_count = 2400
plastic_steps = 30000

#set_genome({'gs': 0.381261212447333, 'IP_sigma': 0.24850875372613956, 'eta_ip': 3.554892196403195, 'eta_stdp': 4.4771428322595506e-05, 'igs': 9.23610086304308, 'it': 0.06842236713342817, 'gen': 163.0, 'score': 0.9977216666666666})
    #{'gs': 0.45843287191962945, 'IP_sigma': 0.40237164995106556, 'eta_ip': 1.696521778172585, 'eta_stdp': 7.45870734695293e-05, 'igs': 14.305182506267165, 'it': 0.07825416873911004, 'gen': 129.0, 'score': 0.9968291666666667})
    #{'gs': 0.43051966037057793, 'IP_sigma': 1.0905837218902503, 'eta_ip': 0.5723874590899439, 'eta_stdp': 9.584398083167788e-05, 'igs': 10.113515129658017, 'it': 0.04290661366908447, 'gen': 89.0, 'score': 0.9903166666666666})

sm = StorageManager('SORN_New', random_nr=True, print_msg=True)

SORN = Network()

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(neuron_count), behaviour={
    2: init_neuron_variables(),
    3: init_afferent_synapses(transmitter='GLU', density='full', distribution='lognormal(1.0,0.6)', normalize=True),
    3.1: init_afferent_synapses(transmitter='GABA', density='full', distribution='uniform(0.1,1.0)', normalize=True),

    10: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']),#, ' man drives car.', ' plant loves rain.', ' parrots can fly.', 'the fish swims' #
    11: Text_Activator(input_density=0.04),#0.03#0.04#0.043#0.015#0.043#0.015

    20: synapse_operation(transmitter='GLU', strength='1.0'),
    #21: synapse_operation(transmitter='GABA', strength='-[0.3#gs]'),

    30: IP(h_ip='lognormal_real_mean(0.02, [0.2944#IP_sigma])', eta_ip='0.07'),#[0.07#eta_ip]
    #31: NOX_Diffusion(th_nox=0.0, strength=1.0),
    32: Refractory_D(steps=3.0),

    40: threshold_output(threshold=0.5),

    50: buffer_variables(),#for STDP
    51: STDP_complex(transmitter='GLU', eta_stdp='0.00015', STDP_F={-1: 1}),#[0.00015#eta_stdp]{-1: 0.2, 1: -1} #, 1: -1
    52: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),


    99: Text_Reconstructor(),

    100: Recorder(tag='avg_rec', variables=['np.mean(n.output)']),
})


inh_neurons = NeuronGroup(net=SORN, tag='inh_neurons', size=get_squared_dim(neuron_count/10), behaviour={
    2: init_neuron_variables(),
    3: init_afferent_synapses(transmitter='GLU', density='[10#igld]%', distribution='uniform([0.1#iglumin],1.0)', normalize=True),

    #8: Text_Generator(text_blocks=[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.']),#, ' parrots can fly.', 'the fish swims'
    #9: Text_Activator(input_density=0.04),#0.03#0.04#0.043#0.015#0.043#0.015

    41: synapse_operation(transmitter='GLU', strength='[5.0#igls]'),
    #11: IP_apply(),
    #12: refrac_apply(strengthfactor='1'),#
    #13: K_WTA_output_local(partition_size=7, K='[0.02#k]'),#0.03 #0.01

    42: threshold_output(threshold='uniform(0.1,0.9)'),#[0.03#it]

    #15: buffer_variables(),#for STDP

    #18: refrac(decayfactor=0.5),
    #20: IP(h_ip='lognormal_real_mean([0.02#k], [0.2944#IP_sigma])', eta_ip='[0.007#eta_ip]'),#
    #21: STDP_complex(transmitter='GLU', eta_stdp='[0.00015#eta_stdp]', STDP_F={-1: 1}),#{-1: 0.2, 1: -1} #, 1: -1
    43: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),
})

SynapseGroup(net=SORN, src=exc_neurons, dst=inh_neurons, tag='GLU,syn', behaviour={})
SynapseGroup(net=SORN, src=inh_neurons, dst=exc_neurons, tag='GABA,syn', behaviour={})



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
    inh_neurons.color = get_color(1, 1)#, 'input_GABA', 'exhaustion_value'
    my_modules = get_default_UI_modules(['output', 'activity', 'input_GLU', 'input_GABA', 'nox', 'refrac_ct', 'exhaustion_value'])+get_my_default_UI_modules()# , 'nox', 'refractory_counter'
    my_modules[0] = UI_sidebar_activity_module(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})
    #my_modules[8] = single_group_plot_tab({'activity': (0, 0, 0), 'excitation': (0, 0, 255), 'inhibition': (255, 0, 0), 'input_act': (255, 0, 255),'exhaustion_value': (0, 255, 0)})
    Network_UI(SORN, modules=my_modules, label='SORN K_WTA', storage_manager=sm, group_display_count=2, reduced_layout=False).show()



#SORN.simulate_iterations(1500, 100)

#score = 1-np.mean(np.abs(SORN['np.mean(n.output)', 0, 'np'][1000:]-0.02))

#set_score(score, sm)


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


