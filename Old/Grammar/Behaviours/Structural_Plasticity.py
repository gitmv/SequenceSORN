from PymoNNto import *

class SORN_weight_noise(Behavior):

    def initialize(self, neurons):
        self.add_tag('weight noise')
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)
        self.noise_fac = self.get_init_attr('max_noise', 0.0001, neurons)
        self.step_ct = self.get_init_attr('step_ct', 100, neurons)

    def iteration(self, neurons):
        if neurons.iteration % self.step_ct == 0:
            for s in neurons.afferent_synapses[self.syn_type]:
                noise = (np.random.random_sample(s.get_synapse_mat_dim())-0.5)*self.noise_fac
                s.W += noise



class SORN_structural_plasticity(Behavior):

    def initialize(self, neurons):
        self.add_tag('structural plasticity')
        self.syn_type = self.get_init_attr('syn_type', 'GLU', neurons)
        self.step_ct = self.get_init_attr('step_ct', 100, neurons)
        self.max_syns_per_step = self.get_init_attr('max_syns_per_step', 5, neurons)
        self.threshold = self.get_init_attr('threshold', 0.00001, neurons)

    def iteration(self, neurons):

        if neurons.iteration % self.step_ct == 0:

            #small_ct = 0
            #replaced_ct = 0

            for s in neurons.afferent_synapses[self.syn_type]:

                for d in range(s.dst.size):

                    small_syn_indices = np.where((s.W[d] <= self.threshold)*s.enabled[d])[0]
                    not_en = np.invert(s.enabled[d])
                    if s.src == s.dst:
                        not_en[d] = False#remove connection to itself
                    new_syn_opt_indices = np.where(not_en)[0]

                    replace_ct = np.min([len(small_syn_indices), len(new_syn_opt_indices), self.max_syns_per_step])

                    if replace_ct > 0:
                        replace_ct = np.random.randint(0, replace_ct)

                        remove_indices = np.random.choice(small_syn_indices, replace_ct, replace=False)
                        add_indices = np.random.choice(new_syn_opt_indices, replace_ct, replace=False)

                        s.enabled[d][remove_indices] = False
                        s.enabled[d][add_indices] = True

                        #small_ct += len(small_syn_indices)
                        #replaced_ct += len(remove_indices)

            #print(small_ct)
