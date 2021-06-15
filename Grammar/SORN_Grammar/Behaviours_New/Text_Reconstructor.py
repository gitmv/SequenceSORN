from PymoNNto import *

class Text_Reconstructor(Behaviour):

    def set_variables(self, neurons):
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def new_iteration(self, neurons):
        if neurons['Text_Activator', 0] is not None:
            char_act = neurons['Text_Activator', 0].mat.transpose().dot(neurons.output)

            self.current_reconstruction_char_index = np.argmax(char_act)
            self.current_reconstruction_char = neurons['Text_Activator', 0].text_generator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char


class Text_Reconstructor_Simple(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('Text_Reconstructor')
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''

    def new_iteration(self, neurons):
        if neurons['Text_Activator', 0] is not None:
            char_act = neurons.activity

            self.current_reconstruction_char_index = np.argmax(char_act)
            self.current_reconstruction_char = neurons['Text_Activator', 0].text_generator.index_to_char(self.current_reconstruction_char_index)

            self.reconstruction_history += self.current_reconstruction_char
