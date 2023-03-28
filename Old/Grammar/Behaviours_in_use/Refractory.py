from PymoNNto import *

class Refractory_D(Behavior):

    def initialize(self, neurons):
        neurons.refrac_ct = neurons.get_neuron_vec()
        self.steps = self.get_init_attr('steps', 5.0, neurons)

    def iteration(self, neurons):
        neurons.refrac_ct = np.clip(neurons.refrac_ct-1.0, 0.0, None)

        neurons.refrac_ct += neurons.output * self.steps

        neurons.activity *= (neurons.refrac_ct <= 1.0)


