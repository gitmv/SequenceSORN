import sys

sys.path.append('../../')

from PymoNNto.NetworkBehaviour.Logic.SORN.SORN_advanced_buffer import *

from PymoNNto.NetworkBehaviour.Logic.SORN.SORN_experimental import *
from PymoNNto.NetworkBehaviour.Logic.SORN.SORN_WTA import *

if __name__ == '__main__':
    from PymoNNto.Exploration.Network_UI import *
    from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *


def run(attrs={'name': 'maze', 'ind': [], 'N_e': 900, 'TS': [1], 'ff': True, 'fb': True, 'plastic': 15000}):
    so = True

    print_info = attrs.get('print', True)

    if print_info:
        print(attrs)

    sm = StorageManager(attrs, random_nr=True, print_msg=print_info)
    sm.save_param_dict(attrs)

    #source = LongDelayGrammar(tag='grammar_act', output_size=attrs['N_e'], random_blocks=True, mode=['simple'], input_density=0.01)
    maze = Maze(level='default', same_color=False)

    SORN = Network()#behaviour={maze.get_network_behaviour()}

    SORN.maze = maze

    e_location = NeuronGroup(net=SORN, tag='PC_{},prediction_source'.format(1), size=maze.get_location_neuron_dimension(), behaviour={
                2: SORN_init_neuron_vars(timescale=1),
                3: SORN_init_afferent_synapses(transmitter='GLU', density='90%', distribution='uniform(0.1,1.0)', normalize=True),#20%#lognormal(0,[0.95#1]) #[13#0]% #, partition_compensation=True , partition_compensation=True #lognormal(0,0.95)
                #4: SORN_init_afferent_synapses(transmitter='GABA', density='[30#1]%', distribution='uniform(0.0,1.0)', normalize=True),
                5: SORN_init_afferent_synapses(transmitter='GLU_cluster', density='90%', distribution='uniform(0.1,1.0)', normalize=True),

                8: maze.get_vision_neuron_behaviour(),
                9: maze.get_location_neuron_behaviour(),


                #10.0: SORN_slow_syn
                10.0: SORN_slow_syn_simple(transmitter='GLU', strength='1.0', so=so), #todo: SORN_slow_syn_simple??????
                10.1: SORN_IP_WTA_apply(),
                #10.15: WTA_refrac_apply(strengthfactor='[0.1#0]'),#0.1 #attrs['refrac']
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
                22: SORN_SN(syn_type='GLU', behaviour_norm_factor=1.0),
                23: SORN_SN(syn_type='GLU_cluster', behaviour_norm_factor='[0.3#6]'),#0.1

    })

    '''
        e_action = NeuronGroup(net=SORN, tag='E_action_{}'.format(timescale), size=maze.get_action_neuron_dimension(), behaviour={
                2: SORN_init_neuron_vars(timescale=timescale),
                3: SORN_init_afferent_synapses(transmitter='GLU', density='full', distribution='uniform(0.1,0.11)', normalize=True, partition_compensation=True),  # 0.89 lognormal(0,[0.5#0])

                12: SORN_slow_syn(transmitter='GLU', strength='[0.1383#2]', so=so),
                13: SORN_slow_syn(transmitter='GABA', strength='-[0.1698#3]', so=False),
                17: SORN_fast_syn(transmitter='GABA', strength='-[0.1#4]', so=False),  # 0.11045
                18: SORN_generate_output(init_TH='0.01'),
                19: SORN_buffer_variables(),

                20: SORN_Refractory_Analog(factor='0.7;+-50%'),
                21: SORN_STDP(eta_stdp='[0.0015#5]', STDP_F={-1: 1, 1:0}, weight_attr='W_temp'),#bigger #todo!!!!
                #22: SORN_SN(syn_type='GLU', clip_max=None, behaviour_norm_factor=5.0),
                22: SORN_temporal_synapses(syn_type='GLU', behaviour_norm_factor=5.0),

                23: SORN_IP_TI(h_ip='0.04', eta_ip='0.001', integration_length='[15#18];+-50%', clip_min=None),
                #25: SORN_NOX(mp='self.partition_sum(n)', eta_nox='[0.5#9];+-50%'),
                #26: SORN_SC_TI(h_sc='lognormal_real_mean([0.015#10], [0.2944#11])', eta_sc='[0.1#12];+-50%',integration_length='1'),  # 60;+-50% #0.05
                #27: SORN_iSTDP(h_ip='same(SCTI, th)', eta_istdp='[0.0001#13]')

                29: SORN_dopamine(),

                30: maze.get_action_neuron_behaviour()
            })


29: SORN_dopamine(),

30: maze.get_action_neuron_behaviour()
    '''

    '''
        e_reward = NeuronGroup(net=SORN, tag='E_reward_{}'.format(timescale), size=maze.get_reward_neuron_dimension(), behaviour={
                2: SORN_init_neuron_vars(timescale=timescale),
                3: SORN_init_afferent_synapses(transmitter='GLU', density='full', distribution='lognormal(0,[0.95#0])', normalize=True, partition_compensation=True),

                9: maze.get_reward_neuron_behaviour(),

                12: SORN_slow_syn(transmitter='GLU', strength='[0.1383#2]', so=so),
                18: SORN_generate_output(init_TH='0.01'),
                19: SORN_buffer_variables(),

                #20: SORN_Refractory(factor='0.5;+-50%'),
                21: SORN_STDP(eta_stdp='[0.00015#5]'),
                22: SORN_SN(syn_type='GLU', clip_max=None, behaviour_norm_factor=0.1),

                #23: SORN_IP_TI(h_ip='0.04', eta_ip='0.001', integration_length='[15#18];+-50%', clip_min=None),
                #25: SORN_NOX(mp='self.partition_sum(n)', eta_nox='[0.5#9];+-50%'),
                #26: SORN_SC_TI(h_sc='lognormal_real_mean([0.015#10], [0.2944#11])', eta_sc='[0.1#12];+-50%',integration_length='1'),  # 60;+-50% #0.05
                #27: SORN_iSTDP(h_ip='same(SCTI, th)', eta_istdp='[0.0001#13]')
            })
    '''


    '''
    e_punishment = NeuronGroup(net=SORN, tag='E_punishment_{}'.format(timescale), size=maze.get_punishment_neuron_dimension(), behaviour={
            2: SORN_init_neuron_vars(timescale=timescale),
            3: SORN_init_afferent_synapses(transmitter='GLU', density='full', distribution='lognormal(0,[0.95#0])', normalize=True, partition_compensation=True),

            9: maze.get_punishment_neuron_behaviour(),

            12: SORN_slow_syn(transmitter='GLU', strength='[0.1383#2]', so=so),
            18: SORN_generate_output(init_TH='0.01'),
            19: SORN_buffer_variables(),

            #20: SORN_Refractory(factor='0.5;+-50%'),
            21: SORN_STDP(eta_stdp='[0.00015#5]'), #todo!!!!
            22: SORN_SN(syn_type='GLU', clip_max=None, behaviour_norm_factor=0.1),

            #23: SORN_IP_TI(h_ip='0.04', eta_ip='0.001', integration_length='[15#18];+-50%', clip_min=None),
            #25: SORN_NOX(mp='self.partition_sum(n)', eta_nox='[0.5#9];+-50%'),
            #26: SORN_SC_TI(h_sc='lognormal_real_mean([0.015#10], [0.2944#11])', eta_sc='[0.1#12];+-50%',integration_length='1'),  # 60;+-50% #0.05
            #27: SORN_iSTDP(h_ip='same(SCTI, th)', eta_istdp='[0.0001#13]')
        })
    '''

    SynapseGroup(net=SORN, src=e_location, dst=e_location, tag='GLU,GLU_same')
    SynapseGroup(net=SORN, src=e_location, dst=e_location, tag='GLU_cluster,syn')

    #SynapseGroup(net=SORN, src=e_vision, dst=e_location, tag='GLU', partition=True)
    #SynapseGroup(net=SORN, src=e_location, dst=e_action, tag='GLU', partition=True)

    #SynapseGroup(net=SORN, src=e_location, dst=e_reward, tag='GLU', partition=True)
    #SynapseGroup(net=SORN, src=e_reward, dst=e_location, tag='DOP+', partition=True)
    #SynapseGroup(net=SORN, src=e_reward, dst=e_action, tag='DOP+', partition=True)

    #SynapseGroup(net=SORN, src=e_location, dst=e_punishment, tag='GLU', partition=True)
    #SynapseGroup(net=SORN, src=e_punishment, dst=e_location, tag='DOP-', partition=True)
    #SynapseGroup(net=SORN, src=e_punishment, dst=e_action, tag='DOP-', partition=True)

    if __name__ == '__main__' and attrs.get('UI', False):
        e_location.color = (0, 0, 255, 255)
        #e_action.color = (255, 255, 0, 255)
        #e_vision.color = (0, 255, 255, 255)
        #e_reward.color = (100, 255, 100, 255)
        #e_punishment.color = (255, 100, 100, 255)

    SORN.set_marked_variables(attrs['ind'], info=print_info, storage_manager=sm)
    SORN.initialize(info=False)

###################################################################################################################

    if __name__ == '__main__' and attrs.get('UI', False):
        #default_modules.insert(0, maze_tab())
        #Network_UI(SORN, label='SORN UI default setup', storage_manager=sm, group_display_count=4, reduced_layout=True).show()
        Network_UI(SORN, modules=get_default_UI_modules(neuron_parameters=['activity'])+[maze_tab()], label='SORN UI K_WTA', storage_manager=sm, group_display_count=1,reduced_layout=False).show()


    return 0


if __name__ == '__main__':
    print('score', run(attrs={'name': 'test', 'ind': [], 'N_e': 900, 'TS': [1], 'UI': True}))
