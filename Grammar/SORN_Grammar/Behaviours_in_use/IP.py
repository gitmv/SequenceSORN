from PymoNNto import *

class IP(Behaviour):

    def set_variables(self, neurons):
        #neurons.target_activity = self.get_init_attr('target_activity', 0.02, neurons)
        neurons.sliding_average_activity = neurons.get_neuron_vec()+neurons.target_activity #initialize with target activity
        self.sliding_window = self.get_init_attr('sliding_window', 1000, neurons)
        self.speed = self.get_init_attr('speed', 0.01, neurons)
        neurons.sensitivity = neurons.get_neuron_vec()+self.get_init_attr('init_sens', 0, neurons)

    def new_iteration(self, neurons):
        #neurons.sliding_average_activity = neurons.output
        #neurons.sensitivity -= ((neurons.output > neurons.target_activity) - 0.5) * 2 * self.speed

        neurons.sliding_average_activity = (neurons.sliding_average_activity*self.sliding_window+neurons.output)/(1+self.sliding_window)

        #neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.speed

        #neurons.sensitivity -= ((neurons.sliding_average_activity > neurons.target_activity)-0.5)*2 * self.speed       *      neurons.target_activity#0.02
        neurons.sensitivity -= (neurons.sliding_average_activity - neurons.target_activity) * self.speed

        neurons.activity += neurons.sensitivity

class ip_new_apply(Behaviour):

    def new_iteration(self, neurons):
        neurons.activity += neurons.sensitivity