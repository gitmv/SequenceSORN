from PymoNNto import *
import random

def one_hot_vec_to_neuron_mat(input_size, output_size, activation_size, input_weighting=None):

    result = np.zeros((output_size, input_size))

    available = set(range(output_size))

    for a in range(input_size):

        char_activiation_size = activation_size

        if input_weighting is not None:
            char_activiation_size = activation_size * input_weighting[a]

        temp = random.sample(available, int(char_activiation_size))
        result[temp, a] = 1
        available = set([n for n in available if n not in temp])
        assert len(available) > 0, 'Input too big for non-overlapping neurons'

    return result





class Text_Activator(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('text_activator')
        self.text_generator = neurons['text_generator', 0]

        input_density = self.get_init_attr('input_density', 1 / 60)
        if input_density < 1:
            activation_size = int(neurons.size * input_density)
        else:
            activation_size = int(input_density)
        neurons.mean_network_activity = activation_size/neurons.size #optional/ can be used by other (homeostatic) modules

        if self.get_init_attr('char_weighting', True):
            cw = self.text_generator.char_weighting
        else:
            cw = None

        neurons.Input_Weights = one_hot_vec_to_neuron_mat(len(self.text_generator.alphabet), neurons.size, activation_size, cw)
        neurons.Input_Mask = np.sum(neurons.Input_Weights, axis=1) > 0

        neurons.input_grammar = neurons.get_neuron_vec()

        self.strength = self.get_init_attr('strength', 1, neurons)

    def new_iteration(self, neurons):
        neurons.input_grammar = neurons.Input_Weights[:, neurons.current_char_index].copy()
        neurons.activity += neurons.input_grammar*self.strength







class Text_Activator_Simple(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('Text_Activator')
        self.text_generator = neurons['text_generator', 0]
        self.alphabet_length = len(self.text_generator.alphabet)
        neurons.one_hot_alphabet_act_vec = np.zeros(self.alphabet_length)

    def new_iteration(self, neurons):
        neurons.one_hot_alphabet_act_vec.fill(0)
        neurons.one_hot_alphabet_act_vec[neurons.current_char_index] = 1.0

        neurons.activity[0:self.alphabet_length] += neurons.one_hot_alphabet_act_vec

class input_synapse_operation(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)  # 1 or -1

        self.text_generator = neurons.network['text_generator', 0]

        input_density = self.get_init_attr('input_density', 1 / 60)
        if input_density < 1:
            activation_size = int(neurons.size * input_density)
        else:
            activation_size = int(input_density)
        neurons.mean_network_activity = activation_size/neurons.size #optional/ can be used by other (homeostatic) modules

        for s in neurons.afferent_synapses['Input_GLU']:
            s.W = one_hot_vec_to_neuron_mat(len(self.text_generator.alphabet), neurons.size, activation_size, self.text_generator.char_weighting)

            s.dst.Input_Weights = s.W

            neurons.Input_Mask = np.sum(s.W, axis=1) > 0

    def new_iteration(self, neurons):
        neurons.input_grammar = neurons.get_neuron_vec()
        for s in neurons.afferent_synapses['Input_GLU']:
            add = s.W.dot(s.src.output) * self.strength
            s.dst.activity = add
            s.dst.input_grammar+=add
