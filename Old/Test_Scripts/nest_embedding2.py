from PymoNNto import *
import nest

class Nest_embedding(Behavior):

    def initialize(self, neurons):
        # define resolution
        nest.SetKernelStatus({'resolution':1.0})

        # define neuron model
        self.G = nest.Create(
            "iaf_psc_delta", 1, params={
                'E_L': 0.0,
                't_ref': 1.0,
                'V_th': 1.0,
                'V_reset': 0.}
            )

        # set variables
        nest.SetStatus(self.G, 'V_m', 1.0)
        nest.SetStatus(self.G, 'tau_m', 100.0)

    def iteration(self, n):
        nest.Simulate(1.0)
        n.v = nest.GetStatus(self.G, 'V_m')

net = Network()

My_Neurons=NeuronGroup(1, net=net, behavior={
    1: Nest_embedding()
})

net.initialize()

for i in range(1000):
    net.simulate_iteration()
    print(My_Neurons.v)
