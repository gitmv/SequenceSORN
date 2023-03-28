from PymoNNto import *

class STDP(Behavior):

    def initialize(self, neurons):
        neurons.eta_stdp = self.get_init_attr('eta_stdp', 0.00015, neurons)
        self.transmitter = self.get_init_attr('transmitter', 'GLU', neurons)

        neurons.output_old = neurons.output.copy()

    def iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:

            #s.W[s.dst.output.astype(bool), s.src.output_old.astype(bool)] += np.clip(s.W[s.dst.output.astype(bool), s.src.output_old.astype(bool)]+neurons.eta_stdp, 0.0, 1.0)

            pre_post = s.dst.output[:, None] * s.src.output_old[None, :]
            #simu = s.dst.output[:, None] * s.src.output[None, :]
            #post_pre = s.dst.output_old[:, None] * s.src.output[None, :]

            dw = neurons.eta_stdp * (pre_post.astype(def_dtype))# - post_pre +simu

            s.W = np.clip(s.W + dw, 0.0, 1.0)

            #print(np.min(s.W), np.max(s.W))

            neurons.output_old = neurons.output.copy()
