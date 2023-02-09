import matplotlib.pyplot as plt
import numpy as np
from PymoNNto import *
from PymoNNto.NetworkBehaviour.Basics.Normalization import *

class Output_Input_Layer(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)
        neurons.linh=1.0

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.activity>0.0
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)


class Output_Excitatory(Behaviour):

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
        if self.act_mul==0.0:
            neurons.activity.fill(0.0)
        else:
            neurons.activity *= self.act_mul

    def get_UI_Preview_Plots(self):
        x=np.arange(0,1,0.01)
        return [[x,self.activation_function(x)]]


class Output_Inhibitory(Behaviour):

    def set_variables(self, neurons):
        self.duration = self.get_init_attr('duration', 1.0)
        self.slope = self.get_init_attr('slope', 20.0)
        self.avg_act = 0
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)

    def activation_function(self, a):
        return np.tanh(a * self.slope)

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        neurons.output = neurons.get_neuron_vec('uniform') < self.activation_function(self.avg_act)#neurons.inh
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)

    def get_UI_Preview_Plots(self):
        x=np.arange(0,1,0.01)
        return [[x,self.activation_function(x)], [x, np.tanh(x * self.slope)]]


class Synapse_Operation(Behaviour):#warning: only use for binary neuron output!

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

    def new_iteration(self, neurons):
        o = np.abs(getattr(neurons, self.input_tag))
        neurons.linh = np.clip(1 - (o - neurons.LI_threshold) * self.strength, 0.0, 1.0)


class Intrinsic_Plasticity(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 0.01, neurons)
        neurons.target_activity = self.get_init_attr('target_activity', 0.02)
        neurons.sensitivity = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.strength
        neurons.activity += neurons.sensitivity


class STDP(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.eta_stdp = self.get_init_attr('strength', 0.005)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            src_len = np.sum(s.src.output_old)
            weight_change = s.dst.linh[s.dst.output]*self.eta_stdp
            dw = np.tile(weight_change[:,None],(1, src_len))
            s.W[np.ix_(s.dst.output, s.src.output_old)] += dw


class Normalization(Behaviour):

    def set_variables(self, neurons):
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)
        self.exec_every_x_step = self.get_init_attr('exec_every_x_step', 1)
        self.afferent = 'afferent' in self.get_init_attr('direction', 'afferent')
        self.efferent = 'efferent' in self.get_init_attr('direction', 'afferent')

    def new_iteration(self, neurons):
        if (neurons.iteration-1) % self.exec_every_x_step == 0:
            if self.afferent:
                self.norm(neurons, afferent=True)
            if self.efferent:
                self.norm(neurons, afferent=False)

    def norm(self, neurons, afferent=True):
        neurons._temp_ws = neurons.get_neuron_vec()

        if afferent:
            axis = 1
            syn = neurons.afferent_synapses[self.syn_type]
        else:
            axis = 0
            syn = neurons.efferent_synapses[self.syn_type]

        for s in syn:
            neurons._temp_ws += np.sum(s.W, axis=axis)

        neurons._temp_ws[neurons._temp_ws == 0.0] = 1.0  # avoid division by zero error

        for s in syn:
            if afferent:
                s.W /= neurons._temp_ws[:, None]
            else:
                s.W /= neurons._temp_ws


class create_weights(Behaviour):

    def set_variables(self, synapses):
        distribution = self.get_init_attr('distribution', 'uniform(0.0,1.0)')#ones
        density = self.get_init_attr('density', 1.0)

        synapses.W = synapses.get_synapse_mat(distribution, density=density) * synapses.enabled

        self.remove_autapses = self.get_init_attr('remove_autapses', False) and synapses.src == synapses.dst

        if self.get_init_attr('normalize', True):
            synapses.W /= np.sum(synapses.W, axis=1)[:, None]
            synapses.W *= self.get_init_attr('nomr_fac', 1.0)


    def new_iteration(self, synapses):
        if self.remove_autapses:
            np.fill_diagonal(synapses.W, 0.0)