from PymoNNto import *
import random

def one_hot_vec_to_neuron_mat(input_size, output_size, activation_size, input_weighting=None):

    result = np.zeros((output_size, input_size))

    available = set(range(output_size))

    for a in range(input_size):

        char_activiation_size = activation_size

        if input_weighting is not None:
            char_activiation_size = int(activation_size * (input_weighting[a] / np.mean(input_weighting)))

        temp = random.sample(available, char_activiation_size)
        result[temp, a] = 1
        available = set([n for n in available if n not in temp])
        assert len(available) > 0, 'Input too big for non-overlapping neurons'

    return result


class Text_Activator(Behaviour):

    def set_variables(self, neurons):
        self.text_generator = neurons['text_generator', 0]

        input_density = self.get_init_attr('input_density', 1 / 60)
        if input_density < 1:
            activation_size = int(neurons.size * input_density)
        else:
            activation_size = int(input_density)
        neurons.mean_network_activity = activation_size/neurons.size #optional/ can be used by other (homeostatic) modules

        self.mat = one_hot_vec_to_neuron_mat(len(self.text_generator.alphabet), neurons.size, activation_size, self.text_generator.count_chars_in_blocks())
        neurons.Input_Mask = np.sum(self.mat, axis=1) > 0

        #neurons.input = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.activity += self.mat[:, neurons.current_char_index].copy()


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

class input_synapse_operations(Behaviour):

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
            s.W = one_hot_vec_to_neuron_mat(len(self.text_generator.alphabet), neurons.size, activation_size, self.text_generator.count_chars_in_blocks())
            #print(s.W.shape)
            #print(s.get_synapse_mat().shape)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses['Input_GLU']:
            s.slow_add = s.W.dot(s.src.output) * self.strength

            s.dst.activity += s.slow_add
            if self.strength > 0:
                s.dst.excitation += s.slow_add
            else:
                s.dst.inhibition += s.slow_add

#char_act = np.zeros(len(grammar_act.alphabet))
#for ng in Network_UI.network['prediction_source']:
#    recon = ng.Input_Weights.transpose().dot(ng.output)
#    char_act += recon
#char = grammar_act.index_to_char(np.argmax(char_act))


'''

dfsa

def max_source_act_text(network, steps):

    source = network['grammar_act', 0]
    alphabet = source.alphabet
    alphabet_length = len(alphabet)

    result_text = ''

    for i in range(steps):
        network.simulate_iteration()
        char_act = np.zeros(alphabet_length)

        for ng in network['prediction_source']:
            recon = ng.Input_Weights.transpose().dot(ng.output)
            char_act += recon#.numpy()

        char = source.index_to_char(np.argmax(char_act))
        result_text += char

    return result_text

def predict_text_max_source_act(network, steps_plastic, steps_recovery, steps_spont, display=True, stdp_off=True):
    network.simulate_iterations(steps_plastic, 100, measure_block_time=display)

    if stdp_off:
        network.deactivate_mechanisms('STDP')

    network['grammar_act', 0].behaviour_enabled = False

    network.simulate_iterations(steps_recovery, 100, measure_block_time=display)

    text = max_source_act_text(network, steps_spont)

    print(text)

    #print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')

    network['grammar_act', 0].behaviour_enabled = True

    if stdp_off:
        network.activate_mechanisms('STDP')

    return text

def get_max_text_score(network, steps_plastic, steps_recovery, steps_spont, display=True, stdp_off=True, return_key='total_score'):
    text = predict_text_max_source_act(network, steps_plastic, steps_recovery, steps_spont, display, stdp_off)
    scores = network['grammar_act', 0].get_text_score(text)
    return scores[return_key]


'''