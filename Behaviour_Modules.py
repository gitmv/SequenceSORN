from PymoNNto import *
from PymoNNto.NetworkBehaviour.Basics.Normalization import *

class Generate_Output(Behaviour):

    def set_variables(self, neurons):
        self.exp = self.get_init_attr('exp', 1.5, neurons)
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        chance = np.power(np.abs(neurons.activity - 0.5) * 2, self.exp) * (neurons.activity > 0.5)
        neurons.output = neurons.get_neuron_vec("uniform") < chance
        neurons.activity.fill(0)


class Generate_Output_Inh(Behaviour):

    def set_variables(self, neurons):
        self.duration = self.get_init_attr('duration', 1.0, neurons)
        self.slope = self.get_init_attr('slope', 20, neurons)
        self.avg_act = 0
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        neurons.inh = np.tanh(self.avg_act * self.slope)
        neurons.output = neurons.get_neuron_vec('uniform') < neurons.inh
        neurons.activity.fill(0)


class Synapse_Operation(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1, neurons)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def new_iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            s.dst.activity += s.add
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class Learning_Inhibition(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)
        self.threshold = self.get_init_attr('threshold', 0.02, neurons)
        self.transmitter = self.get_init_attr('transmitter', 'GABA', neurons)
        self.input_tag = 'input_' + self.transmitter

    def new_iteration(self, neurons):
        o = np.abs(getattr(neurons, self.input_tag)) - self.threshold #np.mean()
        neurons.linh = np.clip(1 - o * self.strength, 0.0, 1.0)


class Intrinsic_Plasticity(Behaviour):

    def set_variables(self, neurons):
        self.speed = self.get_init_attr('speed', 0.01, neurons)
        neurons.target_activity = self.get_init_attr('target_activity', None, neurons)
        neurons.sensitivity = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.speed
        neurons.activity += neurons.sensitivity


class STDP_simple(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('STDP')
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.eta_stdp = self.get_init_attr('eta_stdp', 0.005, neurons)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            mul = self.eta_stdp * neurons.linh * s.enabled
            dw = s.dst.output[:, None] * s.src.output_old[None, :] * mul
            s.W += dw


class Normalization(Behaviour):

    def set_variables(self, neurons):
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)
        neurons.weight_norm_factor = neurons.get_neuron_vec()+self.get_init_attr('norm_factor', 1.0, neurons)
        self.exec_every_x_step = self.get_init_attr('exec_every_x_step', 1)
        self.aff_eff = self.get_init_attr('syn_direction', 'afferent')

    def new_iteration(self, neurons):
        if neurons.iteration % self.exec_every_x_step == 0:
            if self.aff_eff=='afferent':
                normalize_synapse_attr('W', 'W', neurons.weight_norm_factor, neurons, self.syn_type)
            if self.aff_eff=='efferent':
                normalize_synapse_attr_efferent('W', 'W', neurons.weight_norm_factor, neurons, self.syn_type)


class create_weights(Behaviour):

    def set_variables(self, synapses):
        distribution = self.get_init_attr('distribution', 'uniform(1.0,1.0)')#ones
        density = self.get_init_attr('density', 1)

        synapses.W = synapses.get_synapse_mat(distribution, density=density) * synapses.enabled

        if self.get_init_attr('update_enabled', False):
            synapses.enabled *= synapses.W > 0

        if self.get_init_attr('normalize', True):
            synapses.W /= np.sum(synapses.W, axis=1)[:, None]

    def new_iteration(self, synapses):
        synapses.W = synapses.W * synapses.enabled