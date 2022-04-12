from PymoNNto import *

##########################################################################
#Generate output with k winner takes all algorithm
##########################################################################
class SORN_generate_output_K_WTA(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('K_WTA')

        neurons.output = neurons.get_neuron_vec()

        self.K = self.get_init_attr('K', 10, neurons)

        if self.K < 1:
            self.K = int(neurons.size * self.K)


    def new_iteration(self, neurons):
        ind = np.argpartition(neurons.activity, -self.K)[-self.K:]
        neurons.output.fill(0)
        neurons.output[ind] = 1


class K_WTA_output_local(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('K_WTA_partitioned')

        self.filter_temporal_output = self.get_init_attr('filter_temporal_output', False, neurons)

        self.K = self.get_init_attr('K', 0.1, neurons)#only accepts values between 0 and 1

        partition_size = self.get_init_attr('partition_size', 7, neurons)
        self.partitioned_ng=neurons.partition_size(partition_size)

    def new_iteration(self, neurons):

        for ng in self.partitioned_ng:#
            K = ng.size * self.K
            #for non integer K
            K_floor = int(np.floor(K))
            if np.random.rand() < (K-K_floor):
                K = K_floor+1
            else:
                K = K_floor

            ng.output *= False

            if K>0:
                act = ng.activity.copy()

                if self.filter_temporal_output:
                    act = ng.activity*ng.output#-(s.dst.output*-10000)

                ind = np.argpartition(act, -K)[-K:]
                act_mat = np.zeros(ng.size).astype(bool)
                act_mat[ind] = True

                ng.output += act_mat