from PymoNNto import *
from PymoNNto.NetworkBehaviour.Basics.Normalization import *

class buffer_reqirement:
    def __init__(self, length=-1, variable=''):
        self.length = length
        self.variable = variable

def get_buffer(neurons, variable):
    return neurons.mask_var(neurons.buffers[variable])

class Buffer_Variables(Behaviour):#has to be executed AFTER intra, inter...behaviours

    def set_variables(self, neurons):
        self.add_tag('buffer')

        n=neurons#for compile...
        neurons.get_buffer = get_buffer#just for simpler access

        neurons.precompiled_vars = {}
        neurons.buffers = {}

        post_syn_req = neurons.connected_NG_param_list('afferent_buffer_requirement', syn_tag='All', efferent_NGs=True, search_behaviours=True)
        own_req = neurons.connected_NG_param_list('own_buffer_requirement', same_NG=True, search_behaviours=True)
        pre_syn_req = neurons.connected_NG_param_list('efferent_buffer_requirement', syn_tag='All', afferent_NGs=True, search_behaviours=True)
        all_requirments = post_syn_req+own_req+pre_syn_req

        for req in all_requirments:
            if '.' in req.variable or '(' in req.variable:
                neurons.precompiled_vars[req.variable] = compile(req.variable, '<string>', 'eval')

            if req.variable not in neurons.buffers:
                neurons.buffers[req.variable] = {}

            current = len(neurons.buffers[req.variable])

            if req.length > current:
                neurons.buffers[req.variable] = neurons.get_neuron_vec_buffer(req.length)

    def new_iteration(self, neurons):
        n = neurons  # for compile...

        for variable in neurons.buffers:
            if variable in neurons.precompiled_vars:
                new = eval(variable)
            else:
                new = getattr(neurons, variable)
            neurons.buffer_roll(neurons.buffers[variable], new)


#STDP Complex
class STDP_C(Behaviour):

    def get_STDP_Function(self):
        return self.get_init_attr('STDP_F', {-1: 1, 1: -1})

    def afferent_buffer_requirement(self, neurons):
        self.STDP_F = self.get_STDP_Function()
        self.data = np.array([[t, s] for t, s in self.STDP_F.items()])
        length = int(np.maximum(np.max(self.data[:, 0]), 1)+1)
        return buffer_reqirement(length=length, variable='output')

    def own_buffer_requirement(self, neurons):
        self.STDP_F = self.get_STDP_Function()
        self.data = np.array([[t, s] for t, s in self.STDP_F.items()])
        length = int(np.maximum(np.min(self.data[:, 0])*-1, 1)+1)
        return buffer_reqirement(length=length, variable='output')

    def set_variables(self, neurons):
        self.add_tag('STDP')

        self.weight_attr = self.get_init_attr('weight_attr', 'W', neurons)

        self.STDP_F = self.get_STDP_Function()# left(negative t):pre->post right(positive t):post->pre

        self.pre_post_mask = np.array([t in self.STDP_F for t in range(self.afferent_buffer_requirement(neurons).length)])
        self.pre_post_mul = np.array([self.STDP_F[t] for t in range(self.afferent_buffer_requirement(neurons).length) if t in self.STDP_F])

        self.post_pre_mask = np.array([-t in self.STDP_F for t in range(self.own_buffer_requirement(neurons).length)])
        self.post_pre_mul = np.array([self.STDP_F[-t] for t in range(self.own_buffer_requirement(neurons).length) if -t in self.STDP_F])

        self.transmitter = self.get_init_attr('transmitter', 'GLU', neurons)

        neurons.eta_stdp = self.get_init_attr('eta_stdp', 0.005, neurons)
        neurons.last_output = neurons.get_neuron_vec()
        for s in neurons.afferent_synapses[self.transmitter]:
            s.src_act_sum_old = np.zeros(s.src.size)

        if self.get_init_attr('plot', False):
            import matplotlib.pyplot as plt
            self.data = np.array([[x, y] for x, y in self.STDP_F.items()])
            plt.bar(self.data[:, 0], self.data[:, 1], 1.0)
            plt.axhline(0, color='black')
            plt.axvline(0, color='black')
            plt.show()

    def new_iteration(self, neurons):

        for s in neurons.afferent_synapses[self.transmitter]:

            post_act = get_buffer(s.dst, 'output')
            pre_act = get_buffer(s.src, 'output')

            if not hasattr(s, 'pre_post_mask'):
                s.pre_post_mask = self.pre_post_mask.copy()
                if len(post_act) > len(s.pre_post_mask):
                    s.pre_post_mask = np.concatenate([s.pre_post_mask, np.array([False for _ in range(len(post_act) - len(s.pre_post_mask))])])

            if not hasattr(s, 'post_pre_mask'):
                s.post_pre_mask = self.post_pre_mask.copy()
                if len(pre_act) > len(s.post_pre_mask):
                    s.post_pre_mask = np.concatenate([s.post_pre_mask, np.array([False for _ in range(len(pre_act) - len(s.post_pre_mask))])])

            summed_up_dact = np.sum(post_act[s.pre_post_mask]*self.pre_post_mul[:, None], axis=0)
            summed_up_sact = np.sum(pre_act[s.post_pre_mask]*self.post_pre_mul[:, None], axis=0)

            dw_pre_post = summed_up_dact[:, None] * pre_act[0][None, :]
            dw_post_pre = post_act[0][:, None] * summed_up_sact[None, :]

            s.dw = neurons.eta_stdp * (dw_pre_post+dw_post_pre)

            setattr(s, self.weight_attr, getattr(s, self.weight_attr)+s.dw)

class STDP_Analysis(Behaviour):

    def new_iteration(self, neurons):
        neurons.total_dw = neurons.get_neuron_vec()
        for s in neurons.afferent_synapses['GLU']:
            s.dst.total_dw += np.sum(s.dw, axis=1)

class Normalization(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('SN')
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)

        neurons.require_synapses(self.syn_type, warning=False)#suppresses error when synapse group does not exist

        self.only_positive_synapses = self.get_init_attr('only_positive_synapses', True, neurons)

        self.behaviour_norm_factor = self.get_init_attr('behaviour_norm_factor', 1.0, neurons)
        neurons.weight_norm_factor = neurons.get_neuron_vec()+self.get_init_attr('neuron_norm_factor', 1.0, neurons)

    def new_iteration(self, neurons):

        if self.only_positive_synapses:
            for s in neurons.afferent_synapses[self.syn_type]:
                s.W[s.W < 0.0] = 0.0

        normalize_synapse_attr('W', 'W', neurons.weight_norm_factor*self.behaviour_norm_factor, neurons, self.syn_type)

