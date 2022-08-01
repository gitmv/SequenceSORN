##################################################Experimental


class Generate_Output_Analog(Generate_Output):

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = np.clip(self.activation_function(neurons.activity),0,1) #chance
        neurons._activity = neurons.activity.copy() #for plotting
        neurons.activity.fill(0)

class activity_amplifier(Behaviour):

    def set_variables(self, neurons):
        self.exp=self.get_init_attr('exp', 1)

    def new_iteration(self, neurons):
        neurons.activity = np.power(neurons.activity, self.exp)

class Generate_Output_Inh_Analog(Generate_Output_Inh):

    def new_iteration(self, neurons):
        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        neurons.output = np.clip(self.activation_function(self.avg_act),0,1)
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)

class Learning_Inhibition_test(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', 'GABA', neurons)
        self.input_tag = 'input_' + self.transmitter

        self.a = self.get_init_attr('a', 0.44)  # not needed
        self.b=self.get_init_attr('b', 0.0)#not needed
        self.c=self.get_init_attr('c', 10.0)
        self.d=self.get_init_attr('d', 1.0)


    def sig_f(self,x):
        return 1/(1+np.power(np.e, -self.c*(x-self.a)))*self.d+self.b

    def new_iteration(self, neurons):
        #neurons.linh=1
        neurons.linh = 1-self.sig_f(np.abs(getattr(neurons, self.input_tag)))