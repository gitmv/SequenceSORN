from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
from PymoNNto.Exploration.AnalysisModules import *

from PymoNNto.Exploration.Network_UI.Advanced_Tabs import *
from Plots.iterative_map_plot import *

#class activity_to_volts(Behavior):

#    def initialize(self, neurons):
#        neurons.mV = neurons.get_neuron_vec()

#    def iteration(self, neurons):

        #0.5=average
        #>-50: threshold  (0.5)
        #-60: rest   (0.4)
        #-70: reset  (0.3)

#        neurons.mV = 80.0/(1+np.power(np.e,(-neurons._activity+0.6)*10.0))-70.0

        #neurons.mV = (neurons._activity - 0.4) * 100.0 * -50.0
        #neurons.mV = _activity* 100.0 - 40  - 50.0

def show_UI(net, sm, qa=['STDP', 'Normalization', 'TextActivator'], additional_modules=None):

    if net['EE',0] is not None:
        Weight_Classifier_Pre(net.exc_neurons1, syn_tag='EE')
    if net['ES', 0] is not None:
        Weight_Classifier_Pre(net.exc_neurons1, syn_tag='ES')

    # create ui tab dict
    my_modules = get_modules_dict(
        get_default_UI_modules(['output'], quick_access_tags=qa),#['STDP', 'TextActivator', 'Input', 'ff', 'fb']
        get_my_default_UI_modules(),
        #Reaction_Analysis_Tab(),
        additional_modules
    )

    my_modules['sc1'] = iterative_map_tab("sensitivity", "_voltage", title='IM s-a')
    my_modules['imp1'] = iterative_map_tab("_activity", "_voltage", title='IM a-a')
    my_modules['imp2'] = iterative_map_tab("output", "output", title='IM o-o')

    my_modules['imp3'] = iterative_map_tab("input_GLU", "input_GLU", title='IM iglu-iglu')
    my_modules['imp4'] = iterative_map_tab("input_GABA", "input_GABA", title='IM iga-iga')
    my_modules['imp5'] = iterative_map_tab("input_GLU+input_GABA", "input_GLU+input_GABA", title='IM igluga-igluga')
    my_modules['imp6'] = iterative_map_tab("input_GLU+input_GABA+sensitivity", "input_GLU+input_GABA+sensitivity", title='IM iglugas-iglugas')

    #modify some tabs
    my_modules[multi_group_plot_tab].__init__(['output|target_activity|0.0|target_activity*2', '_voltage', 'sensitivity', 'input_GABA*(-1)|LI_threshold', 'linh'])
    my_modules[single_group_plot_tab].__init__(['output', '_voltage', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons1')

    #create classification AnalysisModules to classify characters and input-non-input neuron classification

    neurons = net.exc_neurons1
    if hasattr(net, 'inp_neurons'):
        neurons = net.inp_neurons
        neurons.Input_Weights = np.zeros((neurons.size, len(net['TextGenerator',0].alphabet)))
        neurons.Input_Weights[np.arange(neurons.size).astype(int), neurons.y.astype(int)] = 1

    if hasattr(neurons, 'Input_Weights'):
        char_classes = np.sum((neurons.Input_Weights>0) * np.arange(1,neurons.Input_Weights.shape[1]+1,1), axis=1).transpose()#neurons.Input_Weights.shape[1]
        Static_Classification(parent=neurons, name='char', classes=char_classes)

    if hasattr(neurons, 'Input_Mask'):
        Static_Classification(parent=neurons, name='input class', classes=neurons.Input_Mask)

    #net.exc_neurons1.add_behavior(100, activity_to_volts())
    #net.add_behaviors_to_object({100:}, net.exc_neurons1)
    #my_modules['mV'] = multi_group_plot_tab(['mV', 'output', '_voltage'])

    # launch ui
    Network_UI(net, modules=my_modules, title=net.tags[0], storage_manager=sm, group_display_count=len(net.NeuronGroups), reduced_layout=False).show()



