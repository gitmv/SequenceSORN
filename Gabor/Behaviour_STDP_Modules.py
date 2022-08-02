from PymoNNto import *

class Complex_STDP_Buffer(Behaviour):

    def set_variables(self, neurons):
        buffer_length = self.get_init_attr('length', neurons._STDP_buffer_length)
        neurons.buffer = np.zeros((buffer_length, neurons.size))
        neurons.linh_buffer = np.zeros((buffer_length, neurons.size))

    def new_iteration(self, neurons):
        neurons.buffer_roll(neurons.buffer, neurons.output.copy())
        neurons.buffer_roll(neurons.linh_buffer, neurons.linh.copy())

class Max_Syn_Size(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        self.max = self.get_init_attr('max', 0.001, neurons)

    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:
            s.W[s.W>self.max] = self.max


class Complex_STDP(Behaviour):

    def set_variables(self, neurons):
        self.transmitter = self.get_init_attr('transmitter', None, neurons)
        eta_stdp = self.get_init_attr('strength', 0.0002)#, strength=0.0015
        self.LTP_function = np.array(self.get_init_attr('LTP', [+0.0, +1.0, +1.0])) * eta_stdp #[+0.0, +0.0, +0.0, +0.0, +0.1, +0.2, +0.6, +1.0, +1.0, +1.0, +0.8, +0.7, +0.5, +0.4, +0.2]
        self.LTD_function = np.array(self.get_init_attr('LTD', [-0.4, -0.6, -0.4])) * eta_stdp #* 0.8 #[-0.1, -0.3, -0.4, -0.5, -0.6, -0.6, -0.7, -0.6, -0.6, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]

        buffer_length = len(self.LTP_function+self.LTD_function)
        for s in neurons.afferent_synapses[self.transmitter]:
            s.src._STDP_buffer_length = buffer_length
            s.dst._STDP_buffer_length = buffer_length

        self.STDP_f_center = self.get_init_attr('STDP_f_center', int(np.round(len(self.LTP_function)/2))-1)
        print(self.STDP_f_center, np.sum(self.LTP_function+self.LTD_function))

        if self.get_init_attr('plot', False):
            plt.bar(np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]),self.LTP_function+self.LTD_function,width=0.3)
            plt.bar(np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])-0.3,self.LTD_function,width=0.3)
            plt.bar(np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])+0.3,self.LTP_function,width=0.3)
            plt.show()

            plt.plot(self.LTP_function + self.LTD_function)
            plt.plot(self.LTP_function*0.5 + self.LTD_function)
            plt.plot(self.LTP_function*0.0 + self.LTD_function)
            plt.show()

            plt.bar(np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]),self.LTP_function+self.LTD_function,width=0.3)
            plt.bar(np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])-0.3,self.LTP_function*0.5+self.LTD_function,width=0.3)
            plt.bar(np.array([-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7])+0.3,self.LTP_function*0.0+self.LTD_function,width=0.3)
            plt.show()


    def new_iteration(self, neurons):
        for s in neurons.afferent_synapses[self.transmitter]:

            pre_buffer = s.src.buffer
            post_act = s.dst.buffer[self.STDP_f_center]

            STDP = self.LTD_function[:,None]+self.LTP_function[:,None]*s.dst.linh_buffer[self.STDP_f_center]

            pre = np.sum(pre_buffer*STDP, axis=0)

            dw = post_act[:,None] * pre[None, :]

            s.W += dw * s.enabled

            s.W.clip(0.0, None, out=s.W)