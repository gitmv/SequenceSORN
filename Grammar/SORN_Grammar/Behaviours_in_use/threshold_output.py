from PymoNNto import *

class threshold_output(Behaviour):

    def set_variables(self, neurons):
        neurons.threshold = neurons.get_neuron_vec()+self.get_init_attr('threshold', 0.0, neurons)

    def new_iteration(self, neurons):
        neurons.output = (neurons.activity >= neurons.threshold)#.astype(def_dtype)

