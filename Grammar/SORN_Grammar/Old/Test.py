from PymoNNto import *

from PymoNNto.NetworkBehaviour.Logic.SORN.SORN_experimental import *
from PymoNNto.NetworkBehaviour.Logic.SORN.SORN_WTA import *
from PymoNNto.NetworkBehaviour.Input.Text.TextActivator import *

from Grammar.Common.Grammar_Helper import *

from Grammar.SORN_Grammar.Behaviours.Structural_Plasticity import *
from PymoNNto.NetworkBehaviour.Input.Images.Lines import *

from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *

from PymoNNto.NetworkBehaviour.Input.Images.Image_Patterns import *

class init(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('init_neuron_vars')

        neurons.activity = neurons.get_neuron_vec()
        neurons.excitation = neurons.get_neuron_vec()
        neurons.inhibition = neurons.get_neuron_vec()
        neurons.input_act = neurons.get_neuron_vec()

        neurons.output = neurons.get_neuron_vec()

        neurons.timescale = self.get_init_attr('timescale', 1)

    def new_iteration(self, neurons):
        #print(np.sum(neurons.activity))
        neurons.output = neurons.activity.copy()#(neurons.activity>0)*1.0
        neurons.activity *= 0.0#9
        #if first_cycle_step(neurons):
        #    neurons.activity.fill(0)# *= 0
        #    neurons.excitation.fill(0)# *= 0
        #    neurons.inhibition.fill(0)# *= 0
        #    neurons.input_act.fill(0)# *= 0

        #    neurons.output_old=neurons.output.copy()

class Input_Behaviour(Behaviour):

    def set_variables(self, neurons):
        for synapse in neurons.afferent_synapses['GLU']:
            synapse.W = synapse.get_synapse_mat('uniform',density=0.1)

    def new_iteration(self, neurons):
        return
        #for synapse in neurons.afferent_synapses['GLU']:
        #    neurons.activity += synapse.W.dot(synapse.src.output)/synapse.src.size

        #neurons.voltage += neurons.get_neuron_vec('uniform',density=0.01)

#source = Line_Patterns(tag='image_act', group_possibility=1.0, grid_width=30, grid_height=30, center_x=list(range(40)), center_y=30 / 2, degree=90, line_length=60)

source = TNAP_Image_Patches(tag='image_act', image_path='C:/Users/Nutzer/Programmieren/Python_Modular_Neural_Network_Toolbox/Images/pexels-photo-275484.jpeg', grid_width=30, grid_height=30, dimensions=['on_center_white', 'off_center_white'], patch_norm=True)#'red', 'green', 'blue', 'gray', '255-red', '255-green', '255-blue', '255-gray',, 'rgbw', '255-rgbw''off_center_white',


SORN = Network()

e_ng = NeuronGroup(net=SORN, tag='PC_{},prediction_source'.format(1), size= NeuronDimension(width=30, height=30, depth=2), behaviour={
    1: init(),
    2: Input_Behaviour(),
    9: SORN_external_input(write_to='activity', strength=1.0, pattern_groups=[source]),
})

SynapseGroup(net=SORN, src=e_ng, dst=e_ng, tag='GLU,syn')

SORN.initialize(info=False)

# print(e_ng['GLU'])
# print(SORN.SynapseGroups)

###################################################################################################################

my_modules = get_default_UI_modules(['output']) + get_my_default_UI_modules()

Network_UI(SORN, modules=my_modules, label='test', storage_manager=None, group_display_count=1, reduced_layout=False).show()