from PymoNNto import *

class random_activity_simple(Behaviour):

    def set_variables(self, neurons):
        self.rate = self.get_init_attr('rate', 0.001, neurons)
        self.strength = self.get_init_attr('strength', 1.0, neurons)

    def new_iteration(self, neurons):
        neurons.activity += (neurons.get_neuron_vec('uniform(0.0,1.0)') < self.rate)*self.strength




#class activity_dependent_random_activity(Behaviour):

#    def set_variables(self, neurons):
#        self.target_activity = self.get_init_attr('target_activity', 0.02, neurons)
#        neurons.sliding_average_activity2 = neurons.get_neuron_vec()+neurons.target_activity #initialize with target activity
#        self.sliding_window = self.get_init_attr('sliding_window', 1000, neurons)

#        self.target_activity = self.get_init_attr('target_activity', 0.02, neurons)


#    def new_iteration(self, neurons):
#        neurons.sliding_average_activity2 = (neurons.sliding_average_activity2*self.sliding_window+neurons.output)/(1+self.sliding_window)

