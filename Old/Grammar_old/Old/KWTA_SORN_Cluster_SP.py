import sys
sys.path.append("..")

from PymoNNto.NetworkBehavior.Logic.SORN.SORN_experimental import *
from PymoNNto.NetworkBehavior.Logic.SORN.SORN_WTA import *
from PymoNNto.NetworkBehavior.Input.Text.TextActivator import *

if __name__ == '__main__':
    from PymoNNto.Exploration.Network_UI import *
    from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
#    from PymoNNto.Exploration.Network_UI.Network_UI import *
#    from PymoNNto.Exploration.Network_UI.DefaultTabs import *

from PymoNNto.NetworkBehavior.Input.Images.Lines import *

class FewSentencesGrammar2(TextActivator_New):

    def get_text_blocks(self):
        return [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.']# ' deer lives in forest.', ' parrots can fly.' ' the fish swims.' , ' deer lives in the forest.',  , ]#, ' penguin.' #,  , ' the fish swims.' #, 'the fish swims.'



def run(attrs={'name': 'KWTA', 'ind': [], 'N_e': 900, 'plastic': 15000}):
    so = True

    print_info = attrs.get('print', True)

    if print_info:
        print(attrs)

    sm = StorageManager(attrs['name'], random_nr=True, print_msg=print_info)
    sm.save_param_dict(attrs)

    #source = FewSentencesGrammar2(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, input_density=0.015, frequency_adjustment=True)#21

    source = Line_Patterns(tag='image_act', group_possibility=1.0, grid_width=30, grid_height=30, center_x=list(range(40)), center_y=30 / 2, degree=90, line_length=60)

    SORN = Network()

    e_ng = NeuronGroup(net=SORN, tag='PC_{},prediction_source'.format(1), size=get_squared_dim(attrs['N_e']), behavior={
                2: SORN_init_neuron_vars(timescale=1),
                3: SORN_init_afferent_synapses(transmitter='GLU', density='40%', distribution='uniform(0.1,1.0)', normalize=True),#20%#lognormal(0,[0.95#1]) #[13#0]% #, partition_compensation=True , partition_compensation=True #lognormal(0,0.95)
                #4: SORN_init_afferent_synapses(transmitter='GABA', density='[30#1]%', distribution='uniform(0.0,1.0)', normalize=True),
                5: SORN_init_afferent_synapses(transmitter='GLU_cluster', density='40%', distribution='uniform(0.1,1.0)', normalize=True),

                #10.0: SORN_slow_syn
                10.0: SORN_slow_syn_simple(transmitter='GLU', strength='1.0', so=so), #todo: SORN_slow_syn_simple??????
                10.1: SORN_IP_WTA_apply(),
                10.15: WTA_refrac_apply(strengthfactor='[0.1#0]'),#0.1 #attrs['refrac']
                10.2: SORN_generate_output_K_WTA_partitioned(partition_size=7, K='[0.02#1]'),

                10.3: SORN_slow_syn_simple(transmitter='GLU_cluster', strength='1.0'),
                10.4: SORN_generate_output_K_WTA_partitioned(partition_size=7, K='[0.02#1]', filter_temporal_output=False),

                10.5: SORN_slow_syn_simple(transmitter='GLU_cluster', strength='1.0'),
                10.6: SORN_generate_output_K_WTA_partitioned(partition_size=7, K='[0.02#1]', filter_temporal_output=False),

                10.7: SORN_slow_syn_simple(transmitter='GLU_cluster', strength='1.0'),
                10.8: SORN_generate_output_K_WTA_partitioned(partition_size=7, K='[0.02#1]', filter_temporal_output=False),

                # 12.1: SORN_WTA_iSTDP(eta_iSTDP='[0.00015#2]'),
                # 12.2: SORN_SN(syn_type='GABA'),
                #13.4: SORN_generate_output_K_WTA_partitioned(K='[0.12#4]'),
                #13.5: SORN_WTA_fast_syn(transmitter='GABA', strength='-[0.5#5]', so=so),#[0.1383#2]SORN_fast_syn
                #14: WTA_refrac(),
                #, filter_temporal_output=True

                15: SORN_buffer_variables(random_temporal_output_shift=False),

                18: WTA_refrac(decayfactor=0.5),
                20: SORN_IP_WTA(h_ip='lognormal_real_mean([0.02#1], [0.2944#2])', eta_ip='[0.007#3]', target_clip_min=None, target_clip_max=None), #-1.0 #1.0 #0.007
                21.1: SORN_STDP(transmitter='GLU', eta_stdp='[0.00015#4]', STDP_F={-1: 0.2, 1: -1}),#, 0: 1 #[0.00015#7] #0.5, 0: 3.0
                21.2: SORN_STDP(transmitter='GLU_cluster', eta_stdp='[0.00015#5]', STDP_F={0: 2.0}),  #[0.00015#7]

                30: SORN_structural_plasticity(syn_type='syn', step_ct=1),
                #31: SORN_weight_noise(max_noise=0.0001),

                41: SORN_SN(syn_type='GLU', behavior_norm_factor=1.0),
                42: SORN_SN(syn_type='GLU_cluster', behavior_norm_factor='[0.3#6]'),  # 0.1
            })


    # SynapseGroup(net=SORN, src=retina, dst=e_ng, tag='GLU,GLU_same', connectivity='(s_id!=d_id)*in_box(10)', partition=True)

    receptive_field = 18
    #receptive_field = int(18*math.sqrt(attrs['N_e']/1400))
    print(receptive_field)

    SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU,syn')#, connectivity='(s_id!=d_id)*in_box({})'.format(receptive_field))#, partition=True)#14
    SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU_cluster,syn')#, connectivity='(s_id!=d_id)*in_box({})'.format(receptive_field))#, partition=True)
    #SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GABA,GABA_same', connectivity='(s_id!=d_id)*in_box(10)', partition=True)

    e_ng.add_behavior(9, SORN_external_input(strength=1.0, pattern_groups=[source]))

    if __name__ == '__main__' and attrs.get('UI', False):
        e_ng.color = get_color(0, 1)

    SORN.set_marked_variables(info=print_info, storage_manager=sm)
    SORN.initialize(info=False)

    #print(e_ng['GLU'])
    #print(SORN.SynapseGroups)

    ###################################################################################################################

    if __name__ == '__main__' and attrs.get('UI', False):
        my_modules = get_default_UI_modules(['output', 'exhaustion_value', 'weight_norm_factor'])+get_my_default_UI_modules()
        my_modules[0] = UI_sidebar_activity_module(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})
        #my_modules[1] = multi_group_plot_tab(['output', 'exhaustion_value', 'weight_norm_factor'])  # , 'nox', 'refractory_counter'
        my_modules[8] = single_group_plot_tab({'activity': (0, 0, 0), 'excitation': (0, 0, 255), 'inhibition': (255, 0, 0), 'input_act': (255, 0, 255),'exhaustion_value': (0, 255, 0)})
        Network_UI(SORN, modules=my_modules, label='SORN UI K_WTA', storage_manager=sm, group_display_count=1, reduced_layout=False).show()

    score = 0
    plastic_steps = attrs.get('plastic', 20000)

    #for i in range(1):
    #    #sm = StorageManager(attrs['name'] + '[{:03d}]'.format(i + 1), random_nr=True, print_msg=print_info)
    #    #sm.save_param_dict(attrs)
    #    sm=None
    #    score += train_and_generate_text(SORN, plastic_steps, 5000, 1000, display=print_info, stdp_off=True, same_timestep_without_feedback_loop=True, steps_recovery=1000, storage_manager=sm, return_key='right_sentences_square_score')

    score = 0

    #print(train_and_generate_text(SORN, plastic_steps, 5000, 1000, display=print_info, stdp_off=True, same_timestep_without_feedback_loop=True, steps_recovery=5000, storage_manager=sm, return_key='right_sentences_square_score'))
    #print('Output:')
    #print(predict_text_max_source_act(SORN, plastic_steps, 5000, 5000, display=True, stdp_off=True))#plastic_steps, 5000, 1000

    #print('score=', score)

    return get_max_text_score(SORN, plastic_steps, 5000, 5000, display=True, stdp_off=True, return_key='right_word_square_score')


if __name__ == '__main__':
    #ind = [0.1024607932656874, 0.017593238652155188, 0.3082856525780059, 0.007677918919646546, 0.00015438098687883516, 0.0001579431193983243, 0.33978128099023547]
    ind = []

    print('score', run(attrs={'name': 'adsfdsfsdf', 'ind': ind, 'N_e': 1400, 'TS': [1], 'UI': True, 'plastic': 30000}))#30000 #50p log just exc 0.04






    #for refrac in [0.1, 0.5, 1.0]:
    #    print('score', run(attrs={'name': 'refrac_param{}'.format(refrac), 'ind': ind, 'N_e': 1400, 'TS': [1], 'UI': False, 'plastic': 30000, 'refrac': refrac}))


    #ind = []#[72.7255592339286, 29.458175070582683, 0.00015704485382051904, 0.006763629096129458, 0.11926664091843557, 0.5031562876644946, 0.035971677467027625, 0.00014754543834789394]#[]
    #ind = [91.28948124066251, 25.70311259727637, 0.00011958137352069863, 0.007103970516674569, 0.08764929945250621, 0.45626446218228583, 0.03506439143087593, 0.00012216591033475477]

    #[0.1, 0.02, 0.2944, 0.007, 0.00015, 0.00015, 0.3]
    # [0.020226353270632835, 0.9169259179752178, 1.1798519990521394, 0.94937075251603, 0.33040671196411336, 0.00664305122252098, 0.00012529660504072136, 0.0001550599521133631]

    #ind = [0.10942671202813736, 0.01969114446391008, 0.28822586745531886, 0.007612200554889048, 0.00015979023347970065, 0.00015715374718117852, 0.28771687148885944]

    #ind = [0.10584534961360437, 0.018824864124830556, 0.31792383056480966, 0.006868328437853531, 0.0001486108734181946, 0.00016336663822907974, 0.3097974393619824]#




# 17: SORN_IP_TI_WTA(h_ip='lognormal_real_mean([0.04#6], [0.2944#7])', eta_ip='[0.006#6];+-50%', integration_length='[15#7];+-50%', clip_min=None),
# 17: SORN_IP_WTA(h_ip='lognormal_real_mean([0.04#6], [0.2944#7])', eta_ip='[0.006#6]', clip_min=None),  # lognormal_real_mean([0.04#6], [0.2944#7]) #[0.0006#6];+-50% ;+-50%
#16: SORN_Neuron_Exhaustion(decay_factor='[0.9#3]', strength='[0.1#4]'),
#17: SORN_IP_TI_WTA(h_ip='[0.04#5]', eta_ip='[0.0006#6];+-50%', integration_length='[15#7];+-50%', clip_min=None),  # lognormal_real_mean([0.04#6], [0.2944#7])
#18: SORN_generate_output_K_WTA(K='[0.04#5]'),
#18: SORN_generate_output(init_TH='0.1;+-100%'),
#20: SORN_Refractory_Digital(factor='0.5;+-50%', threshold=0.1),
#20: SORN_Refractory_Analog(factor='0.5;+-50%'),