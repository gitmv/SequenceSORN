from PymoNNto import *


class SH_act(Behavior):

    def initialize(self, neurons):
        self.speed = self.get_init_attr('speed', 0.01, neurons)
        neurons.SH_target_activity = self.get_init_attr('SH_target_activity', None, neurons)

    def iteration(self, neurons):
        measure = neurons.input_GLU+neurons.input_grammar
        neurons.weight_norm_factor -= (measure - neurons.SH_target_activity) * self.speed


class ip_new_apply(Behavior):

    def iteration(self, neurons):
        neurons.activity += neurons.sensitivity



class IP(Behavior):

    def initialize(self, neurons):
        #neurons.target_activity = self.get_init_attr('target_activity', 0.02, neurons)
        neurons.sliding_average_activity = neurons.get_neuron_vec()+neurons.target_activity #initialize with target activity
        self.sliding_window = self.get_init_attr('sliding_window', 1000, neurons)
        self.speed = self.get_init_attr('speed', 0.01, neurons)
        neurons.sensitivity = neurons.get_neuron_vec()+self.get_init_attr('init_sens', 0, neurons)

    def iteration(self, neurons):
        #neurons.sliding_average_activity = neurons.output
        #neurons.sensitivity -= ((neurons.output > neurons.target_activity) - 0.5) * 2 * self.speed

        neurons.sliding_average_activity = (neurons.sliding_average_activity*self.sliding_window+neurons.output)/(1+self.sliding_window)

        #neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.speed

        #neurons.sensitivity -= ((neurons.sliding_average_activity > neurons.target_activity)-0.5)*2 * self.speed       *      neurons.target_activity#0.02
        neurons.sensitivity -= (neurons.sliding_average_activity - neurons.target_activity) * self.speed

        neurons.activity += neurons.sensitivity

class ip_new_apply(Behavior):

    def iteration(self, neurons):
        neurons.activity += neurons.sensitivity



########experimental

class IP2(Behavior):

    def initialize(self, neurons):
        #neurons.target_activity = self.get_init_attr('target_activity', 0.02, neurons)
        neurons.sliding_average_activity2 = neurons.get_neuron_vec()+neurons.target_activity #initialize with target activity
        self.sliding_window = self.get_init_attr('sliding_window', 1000, neurons)
        self.speed = self.get_init_attr('speed', 0.01, neurons)
        neurons.sensitivity_2 = neurons.get_neuron_vec()+self.get_init_attr('init_sens', 0, neurons)

    def iteration(self, neurons):
        #neurons.sliding_average_activity = neurons.output
        #neurons.sensitivity -= ((neurons.output > neurons.target_activity) - 0.5) * 2 * self.speed

        neurons.sliding_average_activity2 = (neurons.sliding_average_activity2*self.sliding_window+neurons.output)/(1+self.sliding_window)

        #neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.speed

        #neurons.sensitivity -= ((neurons.sliding_average_activity > neurons.target_activity)-0.5)*2 * self.speed       *      neurons.target_activity#0.02
        neurons.sensitivity2 -= (neurons.sliding_average_activity2 - neurons.target_activity) * self.speed

        neurons.activity += neurons.sensitivity2

class ip_new_apply2(Behavior):

    def iteration(self, neurons):
        neurons.activity += neurons.sensitivity2