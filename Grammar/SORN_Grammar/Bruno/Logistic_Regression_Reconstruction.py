from PymoNNto import *
from sklearn import linear_model

class Classifier_Text_Reconstructor(Behaviour):

    def train(self):
        self.classifier = linear_model.LogisticRegression(solver='liblinear', multi_class='auto')
        self.readout_layer = self.classifier.fit(self.x_train, self.y_train)
        self.stop_recording()

    def predict_char_index(self, X):
        if self.readout_layer is not None:
            return int(self.readout_layer.predict(X.reshape(1, -1)))

    def start_recording(self):
        self.x_train = []
        self.y_train = []
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def set_variables(self, neurons):
        #self.add_tag('Text_Reconstructor')
        self.neurons = neurons
        self.current_reconstruction_char = ''
        self.current_reconstruction_char_index = ''
        self.reconstruction_history = ''
        self.classifier = None #if none: untrained
        self.readout_layer = None #if none: untrained
        self.recording = False
        self.strength = self.get_init_attr('strength', 1)
        self.Text_Activator = neurons.network['Text_Activator', 0]
        self.activate_predicted_char=True

    def get_current_char_index(self, neurons):
        for ng in neurons.network.NeuronGroups:
            if hasattr(ng, 'current_char_index'):
                return ng.current_char_index

    def new_iteration(self, neurons):
        if self.recording:
            self.x_train.append(neurons.output.copy())
            self.y_train.append(self.get_current_char_index(neurons))#(neurons.current_char_index)

        if self.Text_Activator is not None:
            if self.readout_layer is not None:
                self.current_reconstruction_char_index = self.predict_char_index(neurons.output)
                self.current_reconstruction_char = self.Text_Activator.text_generator.index_to_char(self.current_reconstruction_char_index)
                self.reconstruction_history += self.current_reconstruction_char

                if not self.Text_Activator.behaviour_enabled and self.activate_predicted_char:
                    neurons.input_grammar = neurons.Input_Weights[:, self.current_reconstruction_char_index].copy()
                    neurons.activity += neurons.input_grammar * self.strength
