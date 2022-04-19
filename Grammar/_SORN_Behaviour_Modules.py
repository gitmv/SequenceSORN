from PymoNNto import *

class Init_Neurons(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.target_activity = self.get_init_attr('target_activity', None, neurons)

    def new_iteration(self, neurons):
        neurons.activity.fill(0)


class Collect_Synapse_Input(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.strength = self.get_init_attr('strength', 1, neurons)  # 1 or -1
        self.input_tag = 'input_' + self.transmitter
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())

    def new_iteration(self, neurons):
        setattr(neurons, self.input_tag, neurons.get_neuron_vec())
        for s in neurons.afferent_synapses[self.transmitter]:
            s.add = s.W.dot(s.src.output) * self.strength
            setattr(s.dst, self.input_tag, getattr(s.dst, self.input_tag) + s.add)


class Receptor_activity_change(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.input_tag = 'input_' + self.transmitter

    def new_iteration(self, neurons):
        neurons.activity += getattr(neurons, self.input_tag)


class Learning_Inhibition(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)
        self.threshold = self.get_init_attr('threshold', 0.02, neurons)

    def new_iteration(self, neurons):
        o = neurons.input_GABA - self.threshold

        neurons.linh = np.clip(o, 0, None)*self.strength
        #buffer = neurons.buffers['output']
        #buffer[1] = np.clip(buffer[1] - neurons.linh, 0.0, 1.0)


class IP(Behaviour):

    def set_variables(self, neurons):
        self.speed = self.get_init_attr('speed', 0.01, neurons)
        neurons.sensitivity = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.speed
        neurons.activity += neurons.sensitivity


class Generate_Output(Behaviour):

    def set_variables(self, neurons):
        self.exp = self.get_init_attr('exp', 1.5, neurons)

    def new_iteration(self, neurons):
        chance = np.power(np.abs(neurons.activity - 0.5) * 2, self.exp) * (x > 0.5)
        neurons.output = neurons.get_neuron_vec("uniform") < chance


class STDP_simple(Behaviour):

    def set_variables(self, neurons):
        neurons.eta_stdp = self.get_init_attr('eta_stdp', 0.005, neurons)
        neurons.prune_stdp = self.get_init_attr('prune_stdp', False, neurons)

        for s in neurons.afferent_synapses['GLU']:
            s.STDP_src_lag_buffer_old = np.zeros(s.src.size)#neurons.size bug?
            s.STDP_src_lag_buffer_new = np.zeros(s.src.size)#neurons.size bug?

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses['GLU']:
            grow = s.dst.output_new[:, None] * s.src.output[None, :]

            grow = np.clip(grow - neurons.linh, 0.0, 1.0)

            s.W += (neurons.eta_stdp * grow) * s.enabled #np.clip(, 0, None) not needed (only grow)


class Normalization(Behaviour):

    def set_variables(self, neurons):
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)

        neurons.require_synapses(self.syn_type, warning=False)#suppresses error when synapse group does not exist

        self.only_positive_synapses = self.get_init_attr('only_positive_synapses', True, neurons)

        self.behaviour_norm_factor = self.get_init_attr('behaviour_norm_factor', 1.0, neurons)
        neurons.weight_norm_factor = neurons.get_neuron_vec()+self.get_init_attr('neuron_norm_factor', 1.0, neurons)

        self.exec_every_x_step = self.get_init_attr('exec_every_x_step', 1)

    def new_iteration(self, neurons):

        if neurons.iteration % self.exec_every_x_step == 0:

            if self.only_positive_synapses:
                for s in neurons.afferent_synapses[self.syn_type]:
                    s.W[s.W < 0.0] = 0.0

            normalize_synapse_attr('W', 'W', neurons.weight_norm_factor*self.behaviour_norm_factor, neurons, self.syn_type)