from PymoNNto import *


#strength='[4.75#S]', duration='[2#D]', slope='[29.4#E]'
class inhibition_test_long(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 10.0, neurons)
        self.duration = self.get_init_attr('duration', 10.0, neurons)
        self.slope = self.get_init_attr('slope', 20, neurons)
        self.avg_act = 0

    def new_iteration(self, neurons):
        #print(np.mean(neurons.output))

        self.avg_act = (self.avg_act*self.duration+np.mean(neurons.output))/(self.duration+1)

        adj = (self.avg_act - np.mean(neurons.target_activity))*self.slope
        adj = adj/np.sqrt(1+np.power(adj, 2.0))*0.1

        print(np.mean(neurons.activity), np.mean(self.avg_act), self.duration, self.slope, np.mean(adj))

        #print(adj)

        #if adj>0:
        neurons.activity -= adj * self.strength

        #print(adj * self.strength)



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
