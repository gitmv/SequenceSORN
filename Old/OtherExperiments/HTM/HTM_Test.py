
from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
from UI.Tabs import *

class Init_Synapses(Behavior):

    def initialize(self, synapses):
        mode = self.get_init_attr('mode', 'all_to_all')

        if mode == 'all_to_all':
            synapses.W = synapses.get_synapse_mat()

        if mode == 'column_wise':
            src = synapses.src
            dst = synapses.dst
            synapses.W = np.random.rand(dst.width * dst.height, src.width * src.height).astype(np.float64)



class Init_HTM(Behavior):

    def initialize(self, neurons):
        neurons.predictive = neurons.get_neuron_vec().astype(bool)
        neurons.active = neurons.get_neuron_vec().astype(bool)
        neurons.active_old = neurons.get_neuron_vec().astype(bool)

    def iteration(self, neurons):
        neurons.active_old = neurons.active.copy()
        neurons.active.fill(0)

class HTM_dynamics(Behavior):

    def initialize(self, neurons):
        self.K = 10

    def iteration(self, neurons):

        neurons.input = neurons.get_neuron_vec()
        for syn in neurons.afferent_synapses['proximal']:
            column_act = np.dot(syn.src.active_old[0:syn.src.width*syn.src.height], syn.W)
            neurons.input += np.tile(column_act, neurons.depth)

        neurons.predicitve_input = neurons.get_neuron_vec()
        for syn in neurons.afferent_synapses['dendral']:
            neurons.predicitve_input += np.dot(syn.src.active_old, syn.W)

        #for syn in neurons.afferent_synapses['apical']:
        #    neurons.input += np.dot(syn.src.active * syn.W)

        neurons.input = np.reshape(neurons.input, (neurons.depth, neurons.height, neurons.width))
        column_activation = np.sum(neurons.input, axis=0)
        column_activation = column_activation.flatten()

        #KWTA
        ind = np.argpartition(column_activation, -self.K)[-self.K:]
        column_activation.fill(0)
        column_activation[ind] = 1

        neurons.active += np.tile(column_activation.astype(bool), neurons.depth)



class HTM_Learning(Behavior):

    def initialize(self, neurons):
        self.eta_stdp = self.get_init_attr('strength', 0.005)

    def iteration(self, neurons):
        for s in neurons.afferent_synapses['proximal']:
            dw = s.dst.active[:, None] * s.src.active_old[None, :]
            s.W += dw

        # s.W.clip(0.0, None, out=s.W)#TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        normalize_synapse_attr('W', 'W', 1, neurons, 'proximal')


class rnd_noise_input(Behavior):

    def iteration(self, neurons):
        neurons.active += neurons.get_neuron_vec('uniform')<0.1




htm_net = Network(tag='htm_net')

NeuronGroup(tag='htm_neurons', net=htm_net, size=NeuronDimension(width=10, height=10, depth=10), behavior={
    1: Init_HTM(),
    2: HTM_dynamics()
})

SynapseGroup(tag='proximal', net=htm_net, src='htm_neurons', dst='htm_neurons', behavior={
    1: Init_Synapses(mode='column_wise')
})

SynapseGroup(tag='apical', net=htm_net, src='htm_neurons', dst='htm_neurons', behavior={
    1: Init_Synapses(mode='all_to_all')
})

SynapseGroup(tag='dendral', net=htm_net, src='htm_neurons', dst='htm_neurons', behavior={
    1: Init_Synapses(mode='all_to_all')
})


NeuronGroup(tag='input_neurons', net=htm_net, size=NeuronDimension(width=10, height=10, depth=1), behavior={
    1: Init_HTM(),#only used for n.active
    2: rnd_noise_input()
})

SynapseGroup(tag='proximal', net=htm_net, src='input_neurons', dst='htm_neurons', behavior={
    1: Init_Synapses(mode='column_wise')
})

htm_net.initialize()

Network_UI(htm_net, modules=get_default_UI_modules(['active']), label=htm_net.tags[0], storage_manager=None, group_display_count=len(htm_net.NeuronGroups), reduced_layout=False).show()

htm_net.simulate_iterations(1000, 100, True)
