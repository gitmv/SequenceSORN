from PymoNNto import *

class Text_Activator_IL(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('Text_Activator')
        self.text_generator = neurons['Text_Generator', 0]
        self.strength = self.get_init_attr('strength', 1, neurons)

    def new_iteration(self, neurons):
        neurons.input_grammar = (neurons.y == neurons.current_char_index)*self.strength
        neurons.activity += neurons.input_grammar
        #neurons.output = neurons.activity>0
        #print(neurons.input_grammar)



class Text_Reconstructor_IL(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('Text_Reconstructor')
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def new_iteration(self, neurons):
        if neurons['Text_Activator_IL', 0] is not None:

            neurons.rec_act = neurons.get_neuron_vec()
            for s in neurons.efferent_synapses['GLU']:
                s.src.rec_act += s.W.T.dot(s.dst.output)

            if np.sum(neurons.rec_act)==0:
                self.current_reconstruction_char_index = -1
                self.current_reconstruction_char = '#'
            else:
                index_act = np.sum(neurons.rec_act.reshape((neurons.height, neurons.width)), axis=1)
                self.current_reconstruction_char_index = np.argmax(index_act)
                self.current_reconstruction_char = neurons['Text_Activator_IL', 0].text_generator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char



class Out(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)
        neurons.linh=1.0

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.activity>0.0
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)