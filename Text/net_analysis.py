import matplotlib.pyplot as plt
import numpy as np
from PymoNNto import *
from PymoNNto.Exploration.HelperFunctions.Save_Load import *
from PymoNNto.Exploration.AnalysisModules import *

from Behavior_Core_Modules_v5 import *
from Behavior_Text_Modules_v5 import *
from Helper import *
from PymoNNto.Exploration.AnalysisModules.Weight_Classifier_Base import *
from matplotlib.patches import Circle # for simplified usage, import this patch
from collections import OrderedDict

class MyWeight_Classifier(Classifier_base):

    def get_data_matrix(self, neurons):
        syn_tag = self.parameter('syn_tag', 'EE')

        matrix=[]
        for syn in neurons.afferent_synapses[syn_tag]:
            matrix.append(syn.ignore_transpose_mode(syn.W))

        cm = np.concatenate(matrix, axis=1)
        return cm.T

        #
        #return get_partitioned_synapse_matrix(neurons, syn_tag, 'W').T

class NeuronCluster:

    def __init__(self, class_id, mask, label=''):
        self.class_id = class_id
        self.mask = mask
        self.label = label
        self.size = np.sum(self.mask)

'''

net = load_network('ff_fb_multi_layer_test')#v5_test

for ng in net.NeuronGroups:

    if len(ng.afferent_synapses['GLU'])>0:
        wc1 = MyWeight_Classifier(ng, syn_tag='GLU')
        ng.classes = wc1.exec(sensitivity=25)#3 #15

        key = list(wc1.get_results().keys())[0]
        ng.correlation_matrix = wc1.corrMatrices[key]

        ng.idx = np.argsort(ng.classes)

        #print(np.unique(ng.classes))




        #clusters = []
        #for c in np.unique(classes):
        #    clusters.append(NeuronCluster(c, classes==c))
        #    print(np.sum(classes==c))

        #ng.clusters = clusters


#save_network(net, 'v5_test_c')

save_network(net, 'ff_fb_multi_layer_test_c2')

'''


#net = load_network('v5_test_c')
net = load_network('ff_multi_layer_test_c')
#net = load_network('ff_fb_multi_layer_test_c')
#net = load_network('ff_fb_multi_layer_test_c2')


ng=net.NeuronGroups.pop(-1)
net.NeuronGroups.insert(0,ng)

for ng in net.NeuronGroups:
    ng.clusters = []

net.inp_neurons.clusters = []
for i, c in enumerate(net.inp_neurons.TextGenerator.alphabet):
    mask = net.inp_neurons.y == i
    net.inp_neurons.clusters.append(NeuronCluster(i, mask, c))


for ng in net.NeuronGroups:
    if len(ng.afferent_synapses['GLU']) > 0:
        #plt.matshow(np.array(ng.correlation_matrix)[ng.idx, :][:, ng.idx])
        #plt.show()

        for c_id in np.unique(ng.classes):
            mask=ng.classes==c_id
            sm = np.sum(mask)
            if sm>0:#5
                ng.clusters.append(NeuronCluster(c_id, mask))


    ng.clusters = sorted(ng.clusters, key=lambda x: -x.size)
    ng.clusters = ng.clusters[0:50]

'''

for nc, ng in enumerate(net.NeuronGroups):
    print(nc)
    for cl in ng.clusters:
        connections = OrderedDict()
        if cl.label=='':
            cl.label='#####'

            #get current state
            for n in net.NeuronGroups:
                n._rc_buffer_last = n.vector()
                n._rc_buffer_current = n.vector()
            ng._rc_buffer_last = cl.mask

            for s in net.SynapseGroups:

                if 'GLU' in s.tags and s.dst!=s.src:
                    if net.transposed_synapse_matrix_mode:
                        s.src._rc_buffer_current += s.W.dot(s.dst._rc_buffer_last)
                    else:
                        s.src._rc_buffer_current += s.W.T.dot(s.dst._rc_buffer_last)

            #get cluster activation
            for n in net.NeuronGroups:
                for cl2 in n.clusters:
                    strength=np.sum(n._rc_buffer_current*cl2.mask)
                    if strength>0.0:
                        connections[cl2] = strength

        cl.connections = OrderedDict(sorted(connections.items(), key=lambda x: -x[1]))
        cl.label = list(cl.label)
        #print(cl.connections)


#for _ in range(3):
if True:
    for ng in net.NeuronGroups:
        for cl in ng.clusters:

            for conn_cl, conn_cl_strength in cl.connections.items():

                #if conn_cl.label!='#####' and cl.label=='#####':
                #    cl.label='#'+conn_cl.label

                for i in range(len(conn_cl.label)):
                    if conn_cl.label[i]!='#' and cl.label[i+1]=='#':
                        cl.label[i + 1] = conn_cl.label[i]

for ng in net.NeuronGroups:
    for cl in ng.clusters:
        cl.label.reverse()
        for i in range(len(cl.label)):
            if cl.label[0]=='#':
                cl.label.pop(0)
            else:
                break

        cl.label=''.join(cl.label).replace(' ', '_')

'''


alphabet=net.inp_neurons.TextGenerator.alphabet

for nc, ng in enumerate(net.NeuronGroups):
    print(nc)
    for cl in ng.clusters:
        if cl.label=='':

            #get current state
            for n in net.NeuronGroups:
                n._rc_buffer_last = n.vector()
                n._rc_buffer_current = n.vector()
            ng._rc_buffer_last = cl.mask


            result = ''
            for step in range(4):
                for s in net.SynapseGroups:

                    if 'GLU' in s.tags and s.dst!=s.src:
                        if net.transposed_synapse_matrix_mode:
                            s.src._rc_buffer_current += s.W.dot(s.dst._rc_buffer_last)
                        else:
                            s.src._rc_buffer_current += s.W.T.dot(s.dst._rc_buffer_last)

                for n in net.NeuronGroups:
                    n._rc_buffer_last = n._rc_buffer_current.copy()
                    n._rc_buffer_current *= 0


                index_act = np.sum(net.inp_neurons._rc_buffer_last.reshape((net.inp_neurons.height, net.inp_neurons.width)), axis=1)
                if np.sum(index_act)==0:
                    result = '#' + result
                else:
                    char_id = np.argmax(index_act)
                    result = alphabet[char_id] + result

            cl.label = result.replace(' ', '_')

                    #collection
                    #for n in neurons.network.NeuronGroups:
                    #    n._rc_buffer_last = n._rc_buffer_current.copy()

                    #index_act = np.sum(neurons._rc_buffer_last.reshape((neurons.height, neurons.width)), axis=1)#get "activations" for different characters

                    #self.recon_act_buffer[-(step+1)] += index_act













x = []
y = []
labels = []
sizes = []

xp=0
yp=0
for ng in net.NeuronGroups:

    yp = 0
    for cl in ng.clusters:
        x.append(xp)
        y.append(yp)
        labels.append(cl.label)
        sizes.append(np.sqrt(cl.size*30))#np.sqrt(cl.size/2)
        yp += 10#cl.size
    xp += 100


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

plt.scatter(x, y, alpha = 0.5,s = sizes)

#ax.set_xlim(-10, 110)
#ax.set_ylim(-10, 710)

for i in range(len(x)):
    #ax.add_artist(Circle(xy=(x[i], y[i]), radius=sizes[i]))
    ax.text(x[i]+10, y[i], labels[i], size='medium', color='black')#, horizontalalignment='center' , weight='semibold'

plt.show()





