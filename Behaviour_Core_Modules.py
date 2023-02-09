import matplotlib.pyplot as plt
import numpy as np
from PymoNNto import *
from PymoNNto.NetworkBehaviour.Basics.Normalization import *

class Generate_Output(Behaviour):

    def set_variables(self, neurons):
        self.exp = self.get_init_attr('exp', 1.5)
        self.act_mul = self.get_init_attr('act_mul', 0.0)
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)

    def activation_function(self, a):
        return np.power(np.abs(a - 0.5) * 2, self.exp) * (a > 0.5)

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.get_neuron_vec("uniform") < self.activation_function(neurons.activity)
        neurons._activity = neurons.activity.copy() #for plotting
        neurons.activity *= self.act_mul
        #neurons.activity.fill(0)

    def get_UI_Preview_Plots(self):
        x=np.arange(0,1,0.01)
        return [[x,self.activation_function(x)]]




def f_i(x, ci):
    return np.tanh(x * ci)


def f_i_derivative(x, ci):
    return (4 * ci) / np.power(np.exp(-ci * x) + np.exp(ci * x), 2)


def fi_2(x, ci, px):
    # px = 0.02
    fx = f_i(px, ci)
    fdx = f_i_derivative(px, ci)
    return fx + (x - px) * fdx

class Generate_Output_Inh(Behaviour):

    def set_variables(self, neurons):
        self.duration = self.get_init_attr('duration', 1.0)
        self.slope = self.get_init_attr('slope', 20.0)
        self.avg_act = 0
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)

    def activation_function(self, a):
        #return fi_2(a, self.slope, 0.019230769230769232)
        return np.tanh(a * self.slope)

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        #neurons.inh = np.tanh(self.avg_act * self.slope)
        neurons.output = neurons.get_neuron_vec('uniform') < self.activation_function(self.avg_act)#neurons.inh
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)

    def get_UI_Preview_Plots(self):
        x=np.arange(0,1,0.01)
        return [[x,self.activation_function(x)], [x, np.tanh(x * self.slope)]]



class Synapse_Operation_old(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1.0, neurons)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def new_iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            s.dst.activity += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class Synapse_Operation(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1.0, neurons)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def new_iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = np.sum(s.W[:, s.src.output], axis=1) * self.strength
            s.dst.activity += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class Learning_Inhibition(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)
        neurons.LI_threshold = self.get_init_attr('threshold', np.tanh(0.02*20), neurons)
        self.transmitter = self.get_init_attr('transmitter', 'GABA', neurons)
        self.input_tag = 'input_' + self.transmitter
        neurons.linh = 0.0

        #neuron.LTH = -self.threshold

    def new_iteration(self, neurons):
        o = np.abs(getattr(neurons, self.input_tag))
        #print(np.mean(o), neurons.LI_threshold)
        neurons.linh = np.clip(1 - (o - neurons.LI_threshold) * self.strength, 0.0, 1.0)

class Intrinsic_Plasticity(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 0.01, neurons)
        neurons.target_activity = self.get_init_attr('target_activity', 0.02)
        neurons.sensitivity = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.strength
        neurons.activity += neurons.sensitivity


class STDP_old(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.eta_stdp = self.get_init_attr('strength', 0.005)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            mul = self.eta_stdp * s.enabled
            dw = (s.dst.linh * s.dst.output)[:, None] * s.src.output_old[None, :] * mul
            s.W += dw
            #s.W.clip(0.0, None, out=s.W)



############################################
class STDP(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.eta_stdp = self.get_init_attr('strength', 0.005)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            src_len = np.sum(s.src.output_old)

            #active_synapses = s.W[np.ix_(s.dst.output, s.src.output_old)]

            weight_change = s.dst.linh[s.dst.output]*self.eta_stdp

            dw = np.tile(weight_change[:,None],(1, src_len))

            s.W[np.ix_(s.dst.output, s.src.output_old)] += dw

            #for speed increase
            s._dst_sum[s.dst.output] += np.sum(dw, axis=1)
            s._src_sum[s.src.output_old] += np.sum(dw, axis=0)

############################################




class Normalization(Behaviour):

    def set_variables(self, neurons):
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)
        neurons.weight_norm_factor = neurons.get_neuron_vec()+self.get_init_attr('norm_factor', 1.0, neurons)
        self.exec_every_x_step = self.get_init_attr('exec_every_x_step', 1)
        self.aff_eff = self.get_init_attr('syn_direction', 'afferent')

    def new_iteration(self, neurons):
        if (neurons.iteration-1) % self.exec_every_x_step == 0:
            if self.aff_eff=='afferent':
                normalize_synapse_attr('W', 'W', neurons.weight_norm_factor, neurons, self.syn_type)
            if self.aff_eff=='efferent':
                normalize_synapse_attr_efferent('W', 'W', neurons.weight_norm_factor, neurons, self.syn_type)


class Normalization2(Behaviour):

    def set_variables(self, neurons):
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)
        self.exec_every_x_step = self.get_init_attr('exec_every_x_step', 1)
        self.aff_eff = self.get_init_attr('syn_direction', 'afferent')

    def new_iteration(self, neurons):
        if (neurons.iteration-1) % self.exec_every_x_step == 0:
            if self.aff_eff=='afferent':
                self.norm_aff(neurons)
            #if self.aff_eff=='efferent':
            #    self.norm_eff(neurons)


    def norm_aff(self, neurons):
        neurons._temp_ws = neurons.get_neuron_vec()

        for s in neurons.afferent_synapses[self.syn_type]:
            #print('aff1', np.mean(s._dst_sum), np.mean(np.sum(s.W, axis=1)))
            neurons._temp_ws += s._dst_sum

        neurons._temp_ws[neurons._temp_ws == 0] = 1 #avoid division by zero error

        for s in neurons.afferent_synapses[self.syn_type]:
            s.W /= neurons._temp_ws[:, None]
            s._dst_sum.fill(1.0) # /= neurons._temp_ws

            #print('aff2', np.mean(s._dst_sum), np.mean(np.sum(s.W, axis=1)))


    def norm_eff(self, neurons):
        neurons._temp_ws = neurons.get_neuron_vec()

        for s in neurons.afferent_synapses[self.syn_type]:
            #print('eff1', np.mean(s._src_sum), np.mean(np.sum(s.W, axis=0)))
            neurons._temp_ws += s._src_sum

        neurons._temp_ws[neurons._temp_ws == 0] = 1 #avoid division by zero error

        for s in neurons.afferent_synapses[self.syn_type]:
            s.W /= neurons._temp_ws
            s._src_sum /= neurons._temp_ws

            #print('eff2', np.mean(s._src_sum), np.mean(np.sum(s.W, axis=0)))



class create_weights(Behaviour):

    def set_variables(self, synapses):
        distribution = self.get_init_attr('distribution', 'uniform(0.0,1.0)')#ones
        density = self.get_init_attr('density', 1.0)

        synapses.W = synapses.get_synapse_mat(distribution, density=density) * synapses.enabled

        #if self.get_init_attr('update_enabled', False):
        #    synapses.enabled *= synapses.W > 0

        #if self.get_init_attr('remove_autapses', False) and synapses.src == synapses.dst:
        #    diag = synapses.get_synapse_mat('ones')>0.0
        #    np.fill_diagonal(diag, False)
        #    synapses.enabled *= diag
        self.remove_autapses = self.get_init_attr('remove_autapses', False) and synapses.src == synapses.dst

        if self.get_init_attr('normalize', True):
            synapses.W /= np.sum(synapses.W, axis=1)[:, None]
            synapses.W *= self.get_init_attr('nomr_fac', 1.0)

        synapses._dst_sum = np.sum(synapses.W, axis=1)
        synapses._src_sum = np.sum(synapses.W, axis=0)

        #print(synapses._dst_sum)
        #print(synapses._src_sum)

    def new_iteration(self, synapses):
        #if type(synapses.enabled)!=bool or synapses.enabled!=True:
        #    synapses.W = synapses.W * synapses.enabled

        if self.remove_autapses:
            np.fill_diagonal(synapses.W, 0.0)




#####################################

class Refrac_New(Behaviour):

    def set_variables(self, neurons):
        neurons.exhaustion = neurons.get_neuron_vec()
        self.exh_add = self.get_init_attr('exh_add', 0.1, neurons)

    def new_iteration(self, neurons):
        neurons.exhaustion += neurons.output + (neurons.output - 1)
        neurons.exhaustion = np.clip(neurons.exhaustion, 0, None)
        neurons.activity -= neurons.exhaustion * self.exh_add




