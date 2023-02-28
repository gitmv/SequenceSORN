import matplotlib.pyplot as plt
import numpy as np
from PymoNNto import *
from PymoNNto.NetworkBehaviour.Basics.Normalization import *

class Output_InputLayer(Behaviour):

    def set_variables(self, neurons):
        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)
        neurons.output_old = neurons.vector(bool)

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.voltage>0.0
        neurons._voltage = neurons.voltage.copy()  # for plotting
        neurons.voltage.fill(0)

class Output_Inhibitory(Behaviour):

    def set_variables(self, neurons):
        self.duration = self.parameter('duration', 2.0)
        self.slope = self.parameter('slope', 14.3)
        self.avg_act = 0
        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)

    def activation_function(self, a):
        return a*self.slope

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.voltage) / (self.duration + 1)
        neurons.output = neurons.vector('uniform') < self.activation_function(self.avg_act)#neurons.inh
        neurons._voltage = neurons.voltage.copy()  # for plotting
        neurons.voltage.fill(0)


class Output_Excitatory(Behaviour):

    def set_variables(self, neurons):
        self.exp = self.parameter('exp', 1.5)
        self.mul = self.parameter('mul', 2.0)
        self.act_mul = self.parameter('act_mul', 0.0)

        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)
        neurons.output_old = neurons.vector(bool)

    def activation_function(self, a):
        return np.power(np.clip(a*self.mul, 0.0, 1.0), self.exp)
        #return np.power(np.abs(a - 0.5) * 2, self.exp) * (a > 0.5)

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.vector("uniform") < self.activation_function(neurons.voltage)
        neurons._voltage = neurons.voltage.copy() #for plotting
        if self.act_mul==0.0:
            neurons.voltage.fill(0.0)
        else:
            neurons.voltage *= self.act_mul


class SynapseOperation(Behaviour):#warning: only use for binary neuron output!

    def set_variables(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.strength = self.parameter('strength', 1.0)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.vector())

    def new_iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.vector())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = np.sum(s.W[:, s.src.output], axis=1) * self.strength
            s.dst.voltage += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class LearningInhibition(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.parameter('strength', 1.0)
        neurons.LI_threshold = self.parameter('threshold', np.tanh(0.02*20))
        self.transmitter = self.parameter('transmitter', 'GABA')
        self.input_tag = 'input_' + self.transmitter
        neurons.linh = neurons.vector()

    def new_iteration(self, neurons):
        o = np.abs(getattr(neurons, self.input_tag))
        neurons.linh = np.clip(1 - (o - neurons.LI_threshold) * self.strength, 0.0, 1.0)


class IntrinsicPlasticity(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.parameter('strength', 0.01)
        neurons.target_activity = self.parameter('target_activity', 0.02)
        neurons.sensitivity = neurons.vector()+self.parameter('init_sensitivity', 0.0)

    def new_iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.strength
        neurons.voltage += neurons.sensitivity


class STDP(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.eta_stdp = self.parameter('strength', 0.005)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            src_len = np.sum(s.src.output_old)
            weight_change = s.dst.linh[s.dst.output]*self.eta_stdp
            dw = np.tile(weight_change[:,None],(1, src_len))
            s.W[np.ix_(s.dst.output, s.src.output_old)] += dw


class Normalization(Behaviour):

    def set_variables(self, neurons):
        self.syn_type = self.parameter('syn_type', 'GLU')
        self.exec_every_x_step = self.parameter('exec_every_x_step', 1)
        self.afferent = 'afferent' in self.parameter('direction', 'afferent')
        self.efferent = 'efferent' in self.parameter('direction', 'afferent')

    def new_iteration(self, neurons):
        if (neurons.iteration-1) % self.exec_every_x_step == 0:
            if self.afferent:
                self.norm(neurons, axis = 1)
            if self.efferent:
                self.norm(neurons, axis = 0)

    def norm(self, neurons, axis):
        neurons._temp_ws = neurons.vector()

        if axis == 1:
            syn = neurons.afferent_synapses[self.syn_type]
        else:
            syn = neurons.efferent_synapses[self.syn_type]

        for s in syn:
            neurons._temp_ws += np.sum(s.W, axis=axis)

        neurons._temp_ws[neurons._temp_ws == 0.0] = 1.0  # avoid division by zero error

        for s in syn:
            if axis == 1:
                s.W /= neurons._temp_ws[:, None]
            else:
                s.W /= neurons._temp_ws


class CreateWeights(Behaviour):

    def set_variables(self, synapses):
        distribution = self.parameter('distribution', 'uniform(0.0,1.0)')#ones
        density = self.parameter('density', 1.0)

        synapses.W = synapses.matrix(distribution, density=density) * synapses.enabled

        self.remove_autapses = self.parameter('remove_autapses', False) and synapses.src == synapses.dst

        if self.parameter('normalize', True):
            synapses.W /= np.sum(synapses.W, axis=1)[:, None]
            synapses.W *= self.parameter('nomr_fac', 1.0)


    def new_iteration(self, synapses):
        if self.remove_autapses:
            np.fill_diagonal(synapses.W, 0.0)