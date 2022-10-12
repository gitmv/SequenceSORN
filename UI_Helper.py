from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
from PymoNNto.Exploration.AnalysisModules import *

from PymoNNto.Exploration.Network_UI.Advanced_Tabs import *
from Plots.iterative_map_plot import *

class activity_to_volts(Behaviour):

    def set_variables(self, neurons):
        neurons.mV = neurons.get_neuron_vec()

    def new_iteration(self, neurons):

        #0.5=average
        #>-50: threshold  (0.5)
        #-60: rest   (0.4)
        #-70: reset  (0.3)

        neurons.mV = 80.0/(1+np.power(np.e,(-neurons._activity+0.6)*10.0))-70.0

        #neurons.mV = (neurons._activity - 0.4) * 100.0 * -50.0
        #neurons.mV = _activity* 100.0 - 40  - 50.0

def show_UI(net, sm, qa=['STDP', 'Text_Activator'], additional_modules=None):

    # create ui tab dict
    my_modules = get_modules_dict(
        get_default_UI_modules(['output'], quick_access_tags=qa),#['STDP', 'Text_Activator', 'Input', 'ff', 'fb']
        get_my_default_UI_modules(),
        #Reaction_Analysis_Tab(),
        additional_modules
    )

    my_modules['sc1'] = iterative_map_tab("sensitivity", "_activity", title='IM s-a')
    my_modules['imp1'] = iterative_map_tab("_activity", "_activity", title='IM a-a')
    my_modules['imp2'] = iterative_map_tab("output", "output", title='IM o-o')

    #modify some tabs
    my_modules[multi_group_plot_tab].__init__(['output|target_activity|0.0|target_activity*2', '_activity', 'sensitivity', 'input_GABA*(-1)|LI_threshold', 'linh'])
    my_modules[single_group_plot_tab].__init__(['output', '_activity', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons')

    #create classification AnalysisModules to classify characters and input-non-input neuron classification

    neurons = net.exc_neurons

    if hasattr(neurons, 'Input_Weights'):
        char_classes = np.sum((neurons.Input_Weights>0) * np.arange(1,neurons.Input_Weights.shape[1]+1,1), axis=1).transpose()#neurons.Input_Weights.shape[1]
        Static_Classification(parent=neurons, name='char', classes=char_classes)

    if hasattr(neurons, 'Input_Mask'):
        Static_Classification(parent=neurons, name='input class', classes=neurons.Input_Mask)

    #net.exc_neurons.add_behaviour(100, activity_to_volts())
    #net.add_behaviours_to_object({100:}, net.exc_neurons)
    #my_modules['mV'] = multi_group_plot_tab(['mV', 'output', '_activity'])

    # launch ui
    Network_UI(net, modules=my_modules, title=net.tags[0], storage_manager=sm, group_display_count=len(net.NeuronGroups), reduced_layout=False).show()



