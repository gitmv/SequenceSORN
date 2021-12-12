from PymoNNto import *
from Grammar.SORN_Grammar.Behaviours_in_use import *

class frequency_activator(Behaviour):

    def expand(self, neurons, vec):
        result = []
        for freq_i in vec:
            result += [freq_i for _ in range(neurons.width)]
        return np.array(result)

    def set_variables(self, neurons):
        self.freq = self.get_init_attr('freq', 0.01)

        if self.get_init_attr('cluster_sizes', False):
            self.f = self.expand(neurons, self.freq)
            self.mask = True
        else:
            cw = self.freq / np.mean(self.freq)
            s = cw / np.max(cw)

            self.mask = neurons.get_neuron_vec('uniform') < self.expand(neurons, s)

            #import matplotlib.pyplot as plt
            #plt.matshow(self.mask.reshape(neurons.height, neurons.width))
            #plt.show()

            su = np.sum(self.mask.reshape(neurons.height, neurons.width), axis=1)

            if min(su)==0:
                print('warning representation has zero neurons')

            f = self.freq / su

            self.f = self.expand(neurons, f)

    def new_iteration(self, neurons):
        neurons.output = (neurons.get_neuron_vec('uniform') < self.f).astype(def_dtype) * self.mask


SORN = Network(tag='SORN')

group_size = 100
freq = np.array([0.173, 0.043, 0.01923])#[0.09, 0.06, 0.03]

input = NeuronGroup(net=SORN, tag='frequency_input', size=NeuronDimension(width=group_size, height=len(freq)), behaviour={
    1: frequency_activator(freq=freq),
    41: Buffer_Variables()
})

exc_neurons = NeuronGroup(net=SORN, tag='exc_neurons', size=get_squared_dim(1000), behaviour={
    #init
    1: Init_Neurons(target_activity='lognormal_rm(0.02,0.3)'),

    #input
    18: Synapse_Operation(transmitter='GLU', strength=25.0),

    #stability
    21: IP(sliding_window=0, speed='[0.007#IP]'),
    22: Refractory_D(steps=4.0),

    #output
    30: ReLu_Output_Prob(),

    #learning
    41: Buffer_Variables(),
    #41.5: Learning_Inhibition(transmitter='GABA', strength=-2),
    #41.5: Learning_Inhibition_mean(strength='-[200#LIM]'),
    42: STDP_C(transmitter='GLU', eta_stdp=0.015, STDP_F={-1: 1}),#0.0015
    45: Normalization(syn_type='GLU')
})

SynapseGroup(net=SORN, src=input, dst=exc_neurons, tag='GLU', behaviour={
    3: create_weights(distribution='lognormal(1.0,0.6)', density='1.0')
})

sm = StorageManager(SORN.tags[0], random_nr=True, print_msg=True)

SORN.initialize(info=True, storage_manager=sm)


ui = True
#User interface
if __name__ == '__main__' and ui:
    input.color = red
    exc_neurons.color = blue
    show_UI(SORN, sm, 2)


def get_w_sum_binay_score(w, fc):
    cs = int(w.shape[1] / fc)
    result = np.zeros(fc)
    for i in range(w.shape[0]):
        ws = np.zeros(fc)
        for j in range(fc):
            ws[j] += np.sum(w[i, j*cs:(j+1)*cs-1])
        result[np.argmax(ws)] += 1
    return result/w.shape[0]

def get_w_max_score(w, fc):
    cs = int(w.shape[1] / fc)
    result = np.zeros(fc)
    am = np.argmax(w, axis=1)
    for j in range(fc):
        result[j] = np.sum((am >= j*cs)*(am < (j+1)*cs))
    return result/w.shape[0]

def get_w_sum_analog_score(w, fc):
    cs = int(w.shape[1] / fc)
    result = np.zeros(fc)
    for i in range(w.shape[0]):
        for j in range(fc):
            result[j] += np.sum(w[i, j*cs:(j+1)*cs])
    return result/w.shape[0]


M_single = []
M_group = []
M_sum = []

x_step = 100
x = []
for i in range(1000):
    print(i)
    SORN.simulate_iterations(x_step, 100, False)
    M_single.append(get_w_max_score(SORN['GLU', 0].W, len(freq)))
    M_group.append(get_w_sum_binay_score(SORN['GLU', 0].W, len(freq)))
    M_sum.append(get_w_sum_analog_score(SORN['GLU', 0].W, len(freq)))
    x.append(i*x_step)

for s in [M_sum, M_group, M_single]:
    s = np.array(s)
    plt.stackplot(x, s[:, 0], s[:, 1], s[:, 2])

    w = x[-1]/100
    y = 0
    for i in range(3):
        h = freq[i]/np.sum(freq)
        plt.bar([x[-1]+w], [h], [w], bottom=[y])
        y += h

    #freq = freq*freq*freq
    #y = 0
    #for i in range(3):
    #    h = freq[i]/np.sum(freq)
    #    plt.bar([-w], [h], [w], bottom=[y])
    #    y += h

    #plt.axis('off')
    #plt.savefig(sm.absolute_path+'out.png', bbox_inches='tight', pad_inches=0)

    #plt.plot(s)
    plt.show()


plt.matshow(SORN['GLU', 0].W[:200, :])
plt.show()

'''
SORN.simulate_iterations(100000, 100)

s1 = get_w_sum_binay_score(SORN['GLU', 0].W, 3)
s2 = get_w_max_score(SORN['GLU', 0].W, 3)
s3 = get_w_sum_analog_score(SORN['GLU', 0].W, 3)

print(s1)
print(s2)
print(s3)

import matplotlib.pyplot as plt

X = np.arange(3)
plt.bar(X + 0.00, [s1[0], s2[0], s3[0]], color='b', width=0.25, label='f1')
plt.bar(X + 0.25, [s1[1], s2[1], s3[1]], color='g', width=0.25, label='f2')
plt.bar(X + 0.50, [s1[2], s2[2], s3[2]], color='r', width=0.25, label='f3')

plt.legend()

plt.title("Clusters")
plt.xlabel("clusters (2 measurements)")
plt.ylabel("number of neurons")

plt.show()

'''