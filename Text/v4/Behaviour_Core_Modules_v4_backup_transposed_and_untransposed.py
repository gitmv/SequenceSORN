import matplotlib.pyplot as plt
import numpy as np
from PymoNNto import *
from PymoNNto.NetworkBehavior.Basics.Normalization import *

settings = {'def_dtype':np.float32, 'transposed_synapse_matrix_mode':True}

class Output_InputLayer(Behavior):

    def initialize(self, neurons):
        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)
        neurons.output_old = neurons.vector(bool)

    def iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.voltage>0.0
        neurons._voltage = neurons.voltage.copy()  # for plotting
        neurons.voltage.fill(0)

class Output_Inhibitory(Behavior):

    def initialize(self, neurons):
        self.duration = self.parameter('duration', 2.0)
        self.avg_inh = self.parameter('avg_inh', 0.28)
        self.avg_act = 0
        self.target_activity = self.parameter('target_activity', 0.02)
        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)

    def activation_function(self, a):
        return (a*self.avg_inh)/self.target_activity

    def iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.voltage) / (self.duration + 1)
        neurons.output = neurons.vector('uniform') < self.activation_function(self.avg_act)
        neurons._voltage = neurons.voltage.copy()  # for plotting
        neurons.voltage.fill(0)

class Output_Excitatory(Behavior):

    def initialize(self, neurons):
        neurons.voltage = neurons.vector()
        neurons.output = neurons.vector(bool)
        neurons.output_old = neurons.vector(bool)
        self.mul = self.parameter('mul', 2.127)
        self.exp = self.parameter('exp', 2.127)

    def activation_function(self, a):
        return np.power(np.clip(a * self.mul, 0.0, 1.0), self.exp)

    def iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.vector("uniform") < self.activation_function(neurons.voltage)
        neurons._voltage = neurons.voltage.copy() #for plotting
        neurons.voltage.fill(0.0)


class SynapseOperation(Behavior):#warning: only use for binary neuron output!

    def initialize(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.strength = self.parameter('strength', 1.0)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.vector())

    def iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.vector())
        for s in neurons.afferent_synapses[self.transmitter]:
            if neurons.network.transposed_synapse_matrix_mode:
                s.add = np.sum(s.W[s.src.output], axis=0) * self.strength
            else:
                s.add = np.sum(s.W[:, s.src.output], axis=1) * self.strength
            s.dst.voltage += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class LearningInhibition(Behavior):

    def initialize(self, neurons):
        self.strength = self.parameter('strength', 1.0)
        self.avg_inh = self.parameter('avg_inh', 0.28)
        self.min = self.parameter('min', 0.0)
        self.max = self.parameter('max', 1.0)
        self.input_tag = 'input_' + self.parameter('transmitter', 'GABA')
        neurons.li_stdp_mul = neurons.vector()

    def iteration(self, neurons):
        inhibition = np.abs(getattr(neurons, self.input_tag))
        neurons.li_stdp_mul = np.clip((1 - inhibition / self.avg_inh) * self.strength, self.min, self.max)


class IntrinsicPlasticity(Behavior):

    def initialize(self, neurons):
        self.strength = self.parameter('strength', 0.01)
        neurons.target_activity = self.parameter('target_activity', 0.02)
        neurons.sensitivity = neurons.vector()+self.parameter('init_sensitivity', 0.0)

    def iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.strength
        neurons.voltage += neurons.sensitivity


class STDP(Behavior):

    def initialize(self, neurons):
        self.transmitter = self.parameter('transmitter', None)
        self.eta_stdp = self.parameter('strength', 0.005)

    def iteration(self, neurons):
        if neurons.network.transposed_synapse_matrix_mode:

            for s in neurons.afferent_synapses[self.transmitter]:
                src_len = np.sum(s.src.output_old)
                weight_change = s.dst.li_stdp_mul[s.dst.output] * self.eta_stdp
                dw = np.tile(weight_change, (src_len, 1))

                mask = np.ix_(s.src.output_old, s.dst.output)
                s.W[mask] += dw

                if neurons.LearningInhibition.min < 0:
                    s.W[mask] = np.clip(s.W[mask],0.0, None)
        else:

            for s in neurons.afferent_synapses[self.transmitter]:
                src_len = np.sum(s.src.output_old)
                weight_change = s.dst.li_stdp_mul[s.dst.output]*self.eta_stdp
                dw = np.tile(weight_change[:,None],(1, src_len))
                
                mask=np.ix_(s.dst.output, s.src.output_old)
                s.W[mask] += dw

                if neurons.LearningInhibition.min<0:
                    s.W[mask] = np.clip(s.W[mask], 0.0, None)



class Normalization(Behavior):

    def initialize(self, neurons):
        self.syn_type = self.parameter('syn_type', 'GLU')
        self.exec_every_x_step = self.parameter('exec_every_x_step', 1)
        self.afferent = 'afferent' in self.parameter('direction', 'afferent')
        self.efferent = 'efferent' in self.parameter('direction', 'afferent')

    def iteration(self, neurons):
        if (neurons.iteration-1) % self.exec_every_x_step == 0:
            if neurons.network.transposed_synapse_matrix_mode:
                if self.afferent:
                    self.norm(neurons, axis = 0)
                if self.efferent:
                    self.norm(neurons, axis = 1)
            else:
                if self.afferent:
                    self.norm(neurons, axis = 1)
                if self.efferent:
                    self.norm(neurons, axis = 0)

    def norm(self, neurons, axis):
        neurons._temp_ws = neurons.vector()

        if neurons.network.transposed_synapse_matrix_mode:
            if axis == 1:
                syn = neurons.efferent_synapses[self.syn_type]
            else:
                syn = neurons.afferent_synapses[self.syn_type]
        else:
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


class CreateWeights(Behavior):

    def initialize(self, synapses):
        distribution = self.parameter('distribution', 'uniform(0.0,1.0)')#ones
        density = self.parameter('density', 1.0)

        synapses.W = synapses.matrix(distribution, density=density) * synapses.enabled

        self.remove_autapses = self.parameter('remove_autapses', False) and synapses.src == synapses.dst

        if self.parameter('normalize', True):
            for i in range(10):
                if synapses.network.transposed_synapse_matrix_mode:
                    synapses.W /= np.sum(synapses.W, axis=1)[:, None]
                    synapses.W /= np.sum(synapses.W, axis=0)
                else:
                    synapses.W /= np.sum(synapses.W, axis=0)
                    synapses.W /= np.sum(synapses.W, axis=1)[:, None]
            synapses.W *= self.parameter('nomr_fac', 1.0)


    def iteration(self, synapses):
        if self.remove_autapses:
            np.fill_diagonal(synapses.W, 0.0)

