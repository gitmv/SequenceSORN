from PymoNNto import *



class Threshold_Output(Behavior):

    def initialize(self, neurons):
        neurons.threshold = neurons.get_neuron_vec()+self.get_init_attr('threshold', 0.0, neurons)

    def iteration(self, neurons):
        neurons.output = (neurons.activity >= neurons.threshold)#.astype(def_dtype)

class variable_slope_relu_exp(Behavior):

    def initialize(self, neurons):
        self.exp = self.get_init_attr('exp', 1.5, neurons)

    def f(self, x):
        return np.power(np.abs(x - 0.5) * 2, self.exp) * (x > 0.5)

    def iteration(self, neurons):
        chance = self.f(neurons.activity)
        neurons.output = neurons.get_neuron_vec("uniform") < chance

class ReLu_Output(Behavior):

    def relu(self, x):
        return np.clip((x - 0.5) * 2.0, 0.0, 1.0)

    def iteration(self, neurons):
        neurons.output = self.relu(neurons.activity)


class ReLu_Output_Prob(ReLu_Output):

    def iteration(self, neurons):
        chance = self.relu(neurons.activity)
        neurons.output = neurons.get_neuron_vec("uniform") < chance

class Mem_Noise_Output_Prob(Behavior):

    def iteration(self, neurons):
        chance = neurons.activity + neurons.get_neuron_vec("uniform")-0.5
        neurons.output = chance > 1.0

class Mem_Noise_Output_Prob_Triangular(Behavior):

    def initialize(self, neurons):
        self.tr_left = self.get_init_attr('tr_left', -0.7)

    def iteration(self, neurons):
        chance = neurons.activity+neurons.get_neuron_vec("triangular("+str(self.tr_left)+", -0.5, 0.0)")#-1.0 #-0.7
        neurons.output = chance > 0.5

class Mem_Noise_Output_Prob_Triangular_test(Behavior):

    def iteration(self, neurons):
        chance = neurons.activity+neurons.get_neuron_vec("triangular(-0.7, -0.5, 0.0)")#-1.0 #-0.7
        neurons.output = chance > 0.3

class Mem_Noise_Output_Prob_Normal(Behavior):

    def iteration(self, neurons):
        chance = neurons.activity+neurons.get_neuron_vec("normal(0.0,0.2)")
        neurons.output = chance > 0.5

class Power_Output(Behavior):

    def power(self, x):
        return np.clip(np.power(x, self.exp), 0.0, 1.0)

    def initialize(self, neurons):
        self.exp = self.get_init_attr('exp', 4.0, neurons)

    def iteration(self, neurons):
        neurons.output = self.power(neurons.activity)


class Power_Output_Prob(Power_Output):

    def iteration(self, neurons):
        chance = self.power(neurons.activity)
        neurons.output = neurons.get_neuron_vec("uniform") < chance


class ID_Output_no_clip(Behavior):

    def iteration(self, neurons):
        neurons.output = neurons.activity

class ID_Output_no_clip_prob(Behavior):

    def iteration(self, neurons):
        neurons.output = neurons.get_neuron_vec('uniform') < neurons.activity

class ID_Output(Behavior):

    def id(self, x):
        return np.clip(x, 0.0, 1.0)

    def iteration(self, neurons):
        neurons.output = self.id(neurons.activity)


class Sigmoid_Output(Behavior):

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.power(np.e, -(x - 0.5) * 15))

    def iteration(self, neurons):
        neurons.output = self.sigmoid(neurons.activity)


class ReLu_Step_Output(Behavior):

    def step(self, x):
        stairs = 4
        return np.clip(np.trunc((x-0.5)*2.0*stairs+1)/stairs, 0.0, 1.0)

    def iteration(self, neurons):
        neurons.output = self.step(neurons.activity)


class norm_output(Behavior):

    def initialize(self, neurons):
        self.factor = self.get_init_attr('factor', 1.0)

    def iteration(self, neurons):

        s=np.sum(neurons.output)
        if s>0:
            neurons.output = neurons.output/s*self.factor




from Old.Grammar.Behaviors_in_use.test import *


def get_a(neurons):
    return np.mean(neurons.afferent_synapses['GLU'][0].src.output)

def set_a(neurons, a):
    neurons.afferent_synapses['GLU'][0].src.activity -= a


#duration='[2#D]', slope='[29.4#E]'

#print(inhibition_func(0, 29.4, 1.0))

class inh_sigmoid_response(Behavior):


    def initialize(self, neurons):
        #self.strength = self.get_init_attr('strength', 10.0, neurons)
        self.duration = self.get_init_attr('duration', 1.0, neurons)
        self.slope = self.get_init_attr('slope', 20, neurons)
        self.avg_act = 0


    def iteration(self, neurons):

        self.avg_act = (self.avg_act * self.duration + neurons.activity) / (self.duration + 1)
        neurons.inh = np.tanh(self.avg_act * self.slope)

        #neurons.inh = np.tanh(neurons.activity*self.slope)

        neurons.output = neurons.get_neuron_vec('uniform') < neurons.inh


        '''
                #inhibition_func(self.avg_act, self.slope, self.strength, 0.050686943101760265)

        #neurons.output = neurons.inh
        
            #set_a(neurons, neurons.inh)
        #print(get_a(neurons), np.mean(neurons.activity))
        adj = (self.avg_act - 0.02) * self.slope #np.mean(neurons.target_activity)
        adj = adj / np.sqrt(1 + np.power(adj, 2.0)) * 0.1

        #print(np.mean(neurons.activity), np.mean(self.avg_act), self.duration, self.strength, self.slope, np.mean(adj))

        #adj = adj#+0.05#+0.24

        neurons.output = adj * self.strength * 0.1

        neurons.inh = adj * self.strength * 0.1

        #neurons.output = neurons.get_neuron_vec('uniform') < adj
        '''

        #print(np.mean(neurons.activity), np.mean(self.avg_act), np.mean(neurons.inh), self.duration, self.strength, self.slope)















#x=np.arange(0.0,1.0,0.01)
'''
import matplotlib.pyplot as plt

a = np.random.normal(0.0, 0.15, size=100000)
b = np.random.triangular(-0.5, 0.0, 0.5, size=100000)

bins = np.histogram(np.hstack((a, b)), bins=100)[1] #get the bin edges

plt.hist(b, bins)
plt.hist(a, bins)


plt.show()


x = []
y = []
for a in np.arange(0, 1, 0.01):
    #chance = a + np.random.uniform(size=10000)-0.5
    #spikes = np.sum(chance > 1.0)
    chance = a + np.random.triangular(-0.5, -0.5, 0.0, size=10000)
    spikes = np.sum(chance > 0.5)
    #chance = a + np.random.normal(0.0, 0.15, size=10000)
    #spikes = np.sum(chance > 0.5)
    x.append(a)
    y.append(spikes/10000)

plt.plot(x, y)
#plt.show()


def f(x,e):
    return np.power((x - 0.5) * 2, e) * (x > 0.5)

for e in [1,2,3,4]:
    plt.plot(np.arange(0, 1, 0.01), [f(x, e) for x in np.arange(0, 1, 0.01)])
plt.show()
'''

#plt.plot(x, relu_output().relu(x))
#plt.show()
#plt.plot(x, sigmoid_output().sigmoid(x))
#plt.show()
#plt.plot(x, x)
#plt.show()
#plt.plot(x, x>0.5)
#plt.show()
#plt.plot(x, power_output().power(x))
#plt.show()
#plt.plot(x, relu_step_output().step(x))
#plt.show()
