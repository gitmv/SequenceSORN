import numpy as np
import random

np.random.seed(1)
random.seed(1)

from PymoNNto import *

from Grammar.SORN_Grammar.Behaviours_in_use.SORN_experimental import *
from Grammar.SORN_Grammar.Behaviours_in_use.SORN_WTA import *
from Input_Behaviours.Text.TextActivator import *

from Grammar.Common.Grammar_Helper import *


#set_genome({'rs': 0.09232235756570088, 'k': 0.021807067918146617, 'IP_mean': 0.023220131242858984, 'IP_sigma': 0.21982802505096785, 'IP_eta': 0.00815087665686908, 'STDP_eta': 0.00013768152104864887, 'STDP_eta_c': 0.00014898516237234308, 'snf': 0.24012377371983357, 'gen': 25.0, 'score': 0.4770888756503841})
#set_genome({'rs': 0.09778653264537968, 'k': 0.025169953249611018, 'IP_mean': 0.019455150509071956, 'IP_sigma': 0.23481957924540256, 'IP_eta': 0.008174544080912168, 'STDP_eta': 0.00017537585575069396, 'STDP_eta_c': 0.00017149686127492485, 'snf': 0.23159342538253455})

ui = False
so = True
print_info = True
name = get_gene('name', 'my_evo')
N_e = 1400
plastic_steps = 30000

sm = StorageManager(name, random_nr=True, print_msg=print_info)

class FewSentencesGrammar2(TextActivator_New):
    def get_text_blocks(self):
        return [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']# ' deer lives in forest.', ' parrots can fly.' ' the fish swims.' , ' deer lives in the forest.',  , ]#, ' penguin.' #,  , ' the fish swims.' #, 'the fish swims.'

source = FewSentencesGrammar2(tag='grammar_act', output_size=N_e, random_blocks=True, input_density=0.015, frequency_adjustment=True)#21

SORN = Network()

e_ng = NeuronGroup(net=SORN, tag='PC_{},prediction_source'.format(1), size=get_squared_dim(N_e), behaviour={
    2: init_neuron_variables(timescale=1),
    3: init_afferent_synapses(transmitter='GLU', density='1%', distribution='uniform(0.1,1.0)', normalize=True),

    #10.0: SORN_slow_syn
    10.0: synapse_operation(transmitter='GLU', strength='1.0'), #todo: SORN_slow_syn_simple??????
    10.1: IP_apply(),
    10.15: refrac_apply(strengthfactor='[0.1#rs]'),#0.1 #attrs['refrac']
    10.2: SORN_generate_output_K_WTA(K='[0.02#k]'),

    15: buffer_variables(random_temporal_output_shift=False),

    18: refrac(decayfactor=0.5),
    20: IP(h_ip='lognormal_real_mean([0.02#IP_mean], [0.2944#IP_sigma])', eta_ip='[0.007#IP_eta]', target_clip_min=None, target_clip_max=None), #-1.0 #1.0 #0.007
    21: STDP_complex(transmitter='GLU', eta_stdp='[0.00015#STDP_eta]', STDP_F={-1: 0.2, 1: -1}),#, 0: 1 #[0.00015#7] #0.5, 0: 3.0
    22: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),
})

SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU,syn', behaviour={})


e_ng.add_behaviour(9, SORN_external_input(strength=1.0, pattern_groups=[source]))

SORN.initialize(info=True, storage_manager=sm)


if __name__ == '__main__' and ui:

    from PymoNNto.Exploration.Network_UI import *
    from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *

    e_ng.color = get_color(0, 1)

    my_modules = get_default_UI_modules(['output', 'exhaustion_value', 'weight_norm_factor'])+get_my_default_UI_modules()
    my_modules[0] = UI_sidebar_activity_module(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})
    #my_modules[1] = multi_group_plot_tab(['output', 'exhaustion_value', 'weight_norm_factor'])  # , 'nox', 'refractory_counter'
    my_modules[8] = single_group_plot_tab({'activity': (0, 0, 0), 'excitation': (0, 0, 255), 'inhibition': (255, 0, 0), 'input_act': (255, 0, 255),'exhaustion_value': (0, 255, 0)})
    Network_UI(SORN, modules=my_modules, label='SORN UI K_WTA', storage_manager=sm, group_display_count=1, reduced_layout=False).show()


score = get_max_text_score(SORN, plastic_steps, 5000, 5000, display=True, stdp_off=True, return_key='right_word_square_score')

set_score(score, sm)
