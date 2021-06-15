import numpy as np
import random

np.random.seed(1)
random.seed(1)

from PymoNNto import *
from PymoNNto.Exploration.Evolution.Interface_Functions import *

from Grammar.SORN_Grammar.Behaviours.SORN_experimental import *
from Grammar.SORN_Grammar.Behaviours.SORN_WTA import *
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
        return ['das ist ein test um zu sehen, ob der algorithmus texte auswendig lernen kann. ']#[' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']# ' deer lives in forest.', ' parrots can fly.' ' the fish swims.' , ' deer lives in the forest.',  , ]#, ' penguin.' #,  , ' the fish swims.' #, 'the fish swims.'

source = FewSentencesGrammar2(tag='grammar_act', output_size=N_e, random_blocks=True, input_density=0.015, frequency_adjustment=True)#21

#source = NewGrammar(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, input_density=0.015, frequency_adjustment=True)
#source = FewSentencesContextGrammar(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, input_density=0.015, frequency_adjustment=True)#
#source = FewLongSentencesGrammar(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, input_density=0.015, frequency_adjustment=True)
#txt='guin eats fish. parrot eats mets. pex eits box ests bea esfish. perrod eats mets. penguin eat. fish. pareod eats mets. bex ests brrod eats mets. box eats brx ests bryod eats mets. pey uin eats fish. parrot eats metd. penguin eats fish. parrot eats mets. pey uin eats fish. pareot eats muts. penguin eats fish. perrot eats mets. pey uingeany fiseanf fnx . ts buin eat. fish. parrot eats mets. bey eats brxod eats bets. beyguin eatf fish. parrot eats bets. penguin eats fish. parrot eats metd. box e ts brx ests bea . fox ests brx edts bgxin eatbefish. perrot eats mets. peyguin eats fish. parrot eats muts. beyguin eats fish. parrotneats mets. feyguin eats fish. parrot eats muts. beyguin eats fish. parrot eats mets. bay uingeanf ftshf sh. per ot  ats mets. box eins boe e ts box eats brxddy fnseingoin eath. fah. d  buinueats fish. parrot eats mets. pay uin uats fish. parrot eats mets. perguinguat eais . pen uin eats fish. perrot eats mets. boyguinsuin eat. boe d  box euin eat  fish. panrod eats metd. beyguins bnx eats brx t d. soxead. fox eats brx ests bradd eats mets. poy uin eatg fish. parrot eats mets. pey uin eats fish. parrot eats mets. pey uin eaes fish. parrot eats mets. ponguin eats fish. perrodneats meas. box eats brx d.ts mguin eats fish. paneod eats meas. payguin aat  fish. pareot eats mets. peyguin eats fish. panguineuat bfish. parrod eats mets. pox eits brx eats brx eats brx eats brx eats box eats brx eats bead. ponguineeats fish. pareot eats mets. pexguin eats fish. parrot eats mets. pedguin eats fish. porrot eats mets. box eits box eats bex d forrot eats muts. bey uin uins atsh. fh. od eatn eath.ffeh. parrot eats mets. box uin uing atshf sh. pargod eats mets. pan uin eats fish. perrod eats mets. box eats boa eats fguin eats fish. parrot eats mets. poyguin eats fish. porrot eats mets. box eits bre eats brx eats breot eats bet . box ein uins atsh. fh. pdnguin eat  fish. parrot eats meas. box eats breot eats mets. fesh. par ot eats metd. box eiss besufish. parrot eats mets. payguin  ans fish. parrot eats mets. poyguin eats fish. parrot eats mets. box eats brx eats brx eats breod eats meat. box dats brx eatsnfiin eat. fo h. pod.uin eats fish. parrot eats mets. penguin eats fish. perrot eats mets. penguin eats fish. porrot eats bets. penguin eats fish. parrot eats mets. parguin eats fish. parrot eats metd. box eats bea eatsnbuing ats fish. parrot eats mead. bon eits bead. box ein eats fish. pareot eats mets. peyguin eats fish. parrot eats bets. bey eass brx eats breod eats mets. poyguin aats fish. porrot eats mets. penguin eats fish. parroineeats fish. par ot eafs mets. box eats brx eats bre eats bgxin eats fish. parrot eats buts. panguin eats fish. poneoi. eat. box eats bre e. box eans ats fish. forrot eats mets. box eits brx eats breot  ats metd. box eats braot eats mets. perguin eats fish. parrot eats mets. poy ein eats fish. parrot eats mets. penguin eats fish. parrot eats mett. ead. box eats breod eats meas. box eats brx eats bread eats eats fish. perrot eats mead. box eats braod eats mets. poyguin eats fish. parrot eats beat. fox . ts fuin eats fish. parrot eats mets. penguin eats fish. barrot eats meas. box eats bread. fongod eaas mfish. parrot eats mets. pex eins mea . sa. uin eats fish. borroineeats fish. parrot eats mets. pey eits mea f bhn uinruin eats fish. parrot eats muts. peyguin eats fish. barrot eats mead. box eats brx eats brx eats brx eats breaeats me. . nguin eats fish. parrot eats mets. pey efn  inuinneats fish. panguin eats fish. perrot eats mets. penguin eats fish. parrod. fox eats bre eats bre eats mets. penguin eats fish. perrot eats mea . box eats bre eats uin eats fish. parrot eats meas. feygu n eats fish. parrod eats mets. poy uins brx eats box eats bre eats  ead. forrod eats muts. penguin eats fish. penrod. fat. fox eats brx d. eat. box eats bradd. bad  eat. fash. ngnguin eats fish. parrot eats mets. poy uins aeatfibo. eats bets. perguin uinseatshfish. p rrot eats mets. penguin eats fish. parrot eats mets. penguin eats fish. barrot eats mets. penguin eats fish. parrot eats mets. penguin eats fish. box eanh ata f fo. pansod eats mets. poyguin umea fish. parrot eats mets. payguineaa.  on uinseats fish. por ea e me seape buineats braad. erxutn. pe meas. penguin eats fish. parrot eats mets. box eats bre eats breod eats mead. por eats bre eats bre eats nuti. pat.ufng . p eguinfeat. fish. porrot eats mets. poy eats bea eats brx d eats aets. panguin eats fish. parrot eats mets. poy eats bread. box eats brx eats beat.  fog. ats uin eats fish. parrot eats mets. poy uan eins atsh. fh. od eats mets. box eits breod eats mets. pey ean aats fish. parrot eats mets. poy uin eats fish. parrot eats mets. box eats brx eatsnfeat  fish. parrot eats mets. pey eanseats fish. parrot eats bets. box eats braod eats meat. foxh. borgoineeas  uts. pox eins mea . panguin eats fish. parrot eats bets. boy eins ang ats fiah. b reoi. eats fish. parrot eats mets. peyguin eats fish. parrot eats mets. poy ed eats m'
#html = source.mark_with_grammar(txt)
#show_html(html)

#source = SingleWordGrammar(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, input_density=0.015)
#source = FewLongSentencesGrammar(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, input_density=0.015)
#source.plot_char_frequency_histogram(20)



SORN = Network()

e_ng = NeuronGroup(net=SORN, tag='PC_{},prediction_source'.format(1), size=get_squared_dim(N_e), behaviour={
    2: init_neuron_variables(timescale=1),
    3: init_afferent_synapses(transmitter='GLU', density='90%', distribution='uniform(0.1,1.0)', normalize=True),#20%#lognormal(0,[0.95#1]) #[13#0]% #, partition_compensation=True , partition_compensation=True #lognormal(0,0.95)
    #4: SORN_init_afferent_synapses(transmitter='GABA', density='[30#1]%', distribution='uniform(0.0,1.0)', normalize=True),
    5: init_afferent_synapses(transmitter='GLU_cluster', density='90%', distribution='uniform(0.1,1.0)', normalize=True),

    #10.0: SORN_slow_syn
    10.0: synapse_operation(transmitter='GLU', strength='1.0'), #todo: SORN_slow_syn_simple??????
    10.1: IP_apply(),
    10.15: refrac_apply(strengthfactor='[0.1#rs]'),#0.1 #attrs['refrac']
    10.2: K_WTA_output_local(partition_size=7, K='[0.02#k]'),

    10.3: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
    10.4: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

    10.5: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
    10.6: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

    10.7: synapse_operation(transmitter='GLU_cluster', strength='1.0'),
    10.8: K_WTA_output_local(partition_size=7, K='[0.02#k]', filter_temporal_output=False),

    # 12.1: SORN_WTA_iSTDP(eta_iSTDP='[0.00015#2]'),
    # 12.2: SORN_SN(syn_type='GABA'),
    #13.4: SORN_generate_output_K_WTA_partitioned(K='[0.12#4]'),
    #13.5: SORN_WTA_fast_syn(transmitter='GABA', strength='-[0.5#5]', so=so),#[0.1383#2]SORN_fast_syn
    #14: WTA_refrac(),
    #, filter_temporal_output=True

    15: buffer_variables(random_temporal_output_shift=False),

    18: refrac(decayfactor=0.5),
    20: IP(h_ip='lognormal_real_mean([0.02#IP_mean], [0.2944#IP_sigma])', eta_ip='[0.007#IP_eta]', target_clip_min=None, target_clip_max=None), #-1.0 #1.0 #0.007
    21.1: STDP_complex(transmitter='GLU', eta_stdp='[0.00015#STDP_eta]', STDP_F={-1: 0.2, 1: -1}),#, 0: 1 #[0.00015#7] #0.5, 0: 3.0
    21.2: STDP_complex(transmitter='GLU_cluster', eta_stdp='[0.00015#STDP_eta_c]', STDP_F={0: 2.0}),  #[0.00015#7]
    22: Normalization(syn_type='GLU', behaviour_norm_factor=1.0),
    23: Normalization(syn_type='GLU_cluster', behaviour_norm_factor='[0.3#snf]'),#0.1
})

SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU,syn', behaviour={
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto')
})

SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU_cluster,syn', behaviour={
    1: Box_Receptive_Fields(range=18, remove_autapses=True),
    2: Partition(split_size='auto')
})


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

#for i in range(1):
#    #sm = StorageManager(attrs['name'] + '[{:03d}]'.format(i + 1), random_nr=True, print_msg=print_info)
#    #sm.save_param_dict(attrs)
#    sm=None
#    score += train_and_generate_text(SORN, plastic_steps, 5000, 1000, display=print_info, stdp_off=True, same_timestep_without_feedback_loop=True, steps_recovery=1000, storage_manager=sm, return_key='right_sentences_square_score')

#print(train_and_generate_text(SORN, plastic_steps, 5000, 1000, display=print_info, stdp_off=True, same_timestep_without_feedback_loop=True, steps_recovery=5000, storage_manager=sm, return_key='right_sentences_square_score'))
#print('Output:')
#print(predict_text_max_source_act(SORN, plastic_steps, 5000, 5000, display=True, stdp_off=True))#plastic_steps, 5000, 1000

#print('score=', score)

score = get_max_text_score(SORN, plastic_steps, 5000, 5000, display=True, stdp_off=True, return_key='right_word_square_score')

set_score(score, sm)
