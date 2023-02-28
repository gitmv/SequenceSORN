from PymoNNto import *

class Refrac_sub(Behaviour):

    def set_variables(self, neurons):
        neurons.exhaustion = neurons.get_neuron_vec()
        self.exh_add = self.get_init_attr('exh_add', 0.1, neurons)

    def new_iteration(self, neurons):
        neurons.exhaustion += neurons.output + (neurons.output - 1)
        neurons.exhaustion = np.clip(neurons.exhaustion, 0, None)
        neurons.activity -= neurons.exhaustion * self.exh_add

class Refrac_prob(Behaviour):

    def set_variables(self, neurons):
        neurons.refrac_spike_chance = neurons.get_neuron_vec()+1
        self.exhaustion_mul = self.get_init_attr('exhaustion_mul', 0.1)
        self.recovery_mul = self.get_init_attr('recovery_mul', 0.1)

    #c=c*0.8+(1-c)*0.05
    def new_iteration(self, neurons):
        neurons.refrac_spike_chance = neurons.refrac_spike_chance - neurons.output * neurons.refrac_spike_chance * self.exhaustion_mul + (1-neurons.refrac_spike_chance) * self.recovery_mul
        neurons.refrac_spike_chance = np.clip(neurons.refrac_spike_chance, 0, 1.0)
        neurons.activity *= neurons.get_neuron_vec("uniform") < neurons.refrac_spike_chance





