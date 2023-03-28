from PymoNNto import *

#def smooth(y, box_pts):
#    box = np.ones(box_pts)/box_pts
#    y_smooth = np.convolve(y, box, mode='same')
#    return y_smooth

def normal_f(x, mean = 0, std = 1):
    variance = np.square(std)
    return np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))


class isi_reaction_module(Behavior):

    def initialize(self, neurons):
        self.strength = self.get_init_attr('strength', 1, neurons)

        self.isi_history = np.zeros((neurons.size, 100))
        self.last_spike_iteration = neurons.get_neuron_vec()

        self.isi_inhibition = np.zeros((neurons.size, 100))

        neurons.input_isi_inh = neurons.get_neuron_vec()

        target_exp = self.get_init_attr('target_exp', -0.5)
        self.target_distribution = np.array([np.power(x+1, target_exp) for x in range(100)])
        self.target_distribution=self.target_distribution/np.sum(self.target_distribution)

        self.smooth = self.get_init_attr('smooth', 5)
        self.smooth_x = np.array(range(self.smooth*2+1))-self.smooth
        self.smooth_y = s = normal_f(self.smooth_x, mean=0.0, std=3.0/float(self.smooth))
        #(self.smooth+1)-np.abs(self.smooth_x)
        self.smooth_y = self.smooth_y/np.sum(self.smooth_y)/10

        #import matplotlib.pyplot as plt
        #plt.plot(self.smooth_x, self.smooth_y)
        #plt.show()

    def iteration(self, neurons):

        isi = neurons.iteration-self.last_spike_iteration[neurons.output]
        isi = np.clip(isi, 0+self.smooth, 99-self.smooth).astype(int)

        for i,sx in enumerate(self.smooth_x):
            self.isi_history[neurons.output, isi+sx] += self.smooth_y[i]

        s = np.sum(self.isi_history, axis=1)
        self.isi_history = self.isi_history / (s[:, None] + (s[:, None] == 0))

        #self.isi_history *= 0.999

        self.isi_inhibition = np.roll(self.isi_inhibition, -1, axis=1)
        self.isi_inhibition[:, -1] = 0.0

        new_inhibtion = self.isi_history[neurons.output] - self.target_distribution
        new_inhibtion[0:3] = 0.0 #refractory...
        self.isi_inhibition[neurons.output] += new_inhibtion * self.strength

        neurons.input_isi_inh = -self.isi_inhibition[:, 0]
        neurons.activity += neurons.input_isi_inh

        self.last_spike_iteration[neurons.output] = neurons.iteration

        #print(self.isi_inhibition[0])

        #print(self.isi_history[0])

