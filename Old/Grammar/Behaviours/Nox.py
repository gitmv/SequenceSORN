from PymoNNto import *
from PymoNNto.NetworkBehavior.Basics.BasicHomeostasis import *

class NOX_Diffusion(Instant_Homeostasis):

    def partition_sum(self, neurons):
        neurons._temp_act_sum = neurons.get_neuron_vec()
        for sg, sg_rf in zip(self.subgroups, self.subgroups_rf):
            sg._temp_act_sum += np.mean(sg_rf.output)
        return neurons._temp_act_sum

    def initialize(self, neurons):
        super().initialize(neurons)

        self.subgroups = neurons.split_grid_into_sub_group_blocks(steps=[5, 5, 1])

        receptive_field = 5
        self.subgroups_rf = []
        for sg in self.subgroups:
            sg_rf = neurons.get_subgroup_receptive_field_mask(sg, [receptive_field, receptive_field, receptive_field])
            self.subgroups_rf.append(neurons.subGroup(sg_rf))

        neurons.nox = neurons.get_neuron_vec()
        self.adjustment_param = 'nox'

        self.measurement_param = self.get_init_attr('mp', 'self.partition_sum(n)', None)

        self.set_threshold(self.get_init_attr('th_nox', 0, neurons))
        self.adj_strength = -self.get_init_attr('strength', 0.002, neurons)



    def iteration(self, neurons):
        neurons.nox.fill(0)
        super().iteration(neurons)
        neurons.activity -= neurons.nox

