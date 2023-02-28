from PymoNNto import *

class TextReconstructor(Behaviour):

    def set_variables(self, neurons):
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def new_iteration(self, neurons):
        if neurons['TextActivator', 0] is not None:
            char_act = neurons.Input_Weights.transpose().dot(neurons.output)

            self.current_reconstruction_char_index = np.argmax(char_act)
            self.current_reconstruction_char = neurons['TextActivator', 0].TextGenerator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char


class TextReconstructor_Simple(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('TextReconstructor')
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def new_iteration(self, neurons):
        if len(neurons['TextActivator']) > 0:
            self.current_reconstruction_char_index = np.argmax(neurons.output)
            self.current_reconstruction_char = neurons['TextActivator', 0].TextGenerator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char

class Char_Cluster_Compensation(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 1)#1: full compensation 0: original activity

    def new_iteration(self, neurons):
        if len(neurons['TextGenerator']) > 0 and len(neurons['TextGenerator']) > 0:

            if not neurons['TextActivator', 0].behaviour_enabled:
                #neurons.activity *= neurons['TextGenerator', 0].char_weighting
                #neurons.activity += neurons.activity * neurons['TextGenerator', 0].char_weighting

                neurons.activity = neurons.activity*(1-self.strength) + neurons.activity * neurons['TextGenerator', 0].char_weighting*(self.strength)