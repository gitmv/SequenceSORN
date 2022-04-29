from PymoNNto import *


def inhibition_func(x, slope, strength, y=0):
    adj = (x - 0.02) * slope  # np.mean(neurons.target_activity)
    adj = adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1 + y
    # print(np.mean(neurons.activity), np.mean(self.avg_act), self.duration, self.strength, self.slope, np.mean(adj))
    # adj = adj#+0.05#+0.24
    return adj * strength

def inhibition_func2(x):
    return np.tanh(x*15)/6.6 *4.75



class inhibition_2_step_collect(Behaviour):

    def set_variables(self, neurons):
        self.duration = self.get_init_attr('duration', 10.0, neurons)
        self.slope = self.get_init_attr('slope', 20, neurons)
        self.avg_act = 0
        neurons.inh = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act*self.duration+np.mean(neurons.output))/(self.duration+1)
        neurons.inh = np.tanh(self.avg_act * self.slope)


class inhibition_2_step_apply(Behaviour):

    def new_iteration(self, neurons):
        neurons.activity -= neurons.inh

#strength='[4.75#S]', duration='[2#D]', slope='[29.4#E]'
class inhibition_test_long(Behaviour):

    def set_variables(self, neurons):
        #self.strength = self.get_init_attr('strength', 10.0, neurons)
        self.duration = self.get_init_attr('duration', 10.0, neurons)
        self.slope = self.get_init_attr('slope', 20, neurons)
        self.avg_act = 0

    def new_iteration(self, neurons):

        self.avg_act = (self.avg_act*self.duration+np.mean(neurons.output))/(self.duration+1)
        neurons.inh = np.tanh(self.avg_act * self.slope)

        #neurons.inh = np.tanh(np.mean(neurons.output)*self.slope)

        neurons.activity -= neurons.inh

        '''
        #inhibition_func2(self.avg_act) #inhibition_func(self.avg_act, self.slope, self.strength, 0.050686943101760265)
        adj = (self.avg_act - 0.02)*self.slope #np.mean(neurons.target_activity)
        adj = adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1

        #print(np.mean(neurons.activity), np.mean(self.avg_act), self.duration, self.strength, self.slope, np.mean(adj), np.mean(neurons.target_activity))

        #if adj>0:
        neurons.activity -= adj * self.strength

        neurons.inh = adj * self.strength

        neurons.input_GABA = -adj * self.strength
        '''

        #print(np.mean(neurons.output), np.mean(self.avg_act), np.mean(neurons.inh), self.duration, self.strength, self.slope)

        neurons.input_GABA = neurons.inh



def f4(x, slope):
    t = (x-0.02)*slope
    return t/np.sqrt(1+np.power(t, 2))*0.1




class inhibition_test(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 10.0, neurons)

    def new_iteration(self, neurons):
        adj = np.mean(neurons.output) - np.mean(neurons.target_activity)
        neurons.activity -= adj * self.strength


class linear_output(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 10.0, neurons)
        self.duration = self.get_init_attr('duration', 10.0, neurons)
        self.avg_act = 0

    def new_iteration(self, neurons):

        self.avg_act = (self.avg_act*self.duration+neurons.activity)/(self.duration+1)

        adj = self.avg_act - np.mean(neurons.target_activity)
        neurons.activity -= adj * self.strength
