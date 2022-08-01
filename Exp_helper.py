from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
#from recurrent_unsupervised_stdp_learning.UI.Analysis_Modules import *

blue = (0.0, 0.0, 255.0, 255.0)
red = (255.0, 0.0, 0.0, 255.0)
yellow = (255.0, 150.0, 0.0, 255.0)

black = (0.0, 0.0, 0.0, 255.0)

#h = 0.1
#b = 0.7179041071061821 + h*100 * -0.04398732037836847
#c = 25.220023435454422 + h*100 * -2.0929686503913683

#class Static_Classification(AnalysisModule):

#    def initialize(self, neurons):
#        self.add_tag('classifier')
#        self.name = self.get_init_attr('name', super()._get_base_name_())
#        self.classes = self.get_init_attr('classes', None)
#        self.save_result(self.name, self.classes)

#    def _get_base_name_(self):
#        return self.name



def show_UI_2(net, sm):

    # all modules in dict
    my_modules = get_modules_dict(
        get_default_UI_modules(['output'], quick_access_tags=['STDP', 'Input', 'ff', 'fb']),
        get_my_default_UI_modules(),
        # isi_module_tab(),
        #Reaction_Analysis_Tab(),
        cluster_bar_tab(),
        similarity_matrix_tab(),
        sidebar_patch_reconstructor_module()
    )

    neurons = net['exc_neurons', 0]

    #char_classes = np.sum((neurons.Input_Weights>0) * np.arange(1,neurons.Input_Weights.shape[1]+1,1), axis=1)#neurons.Input_Weights.shape[1]
    #neurons.add_analysis_module(Static_Classification(name='char', classes=char_classes.transpose()))
    if hasattr(neurons, 'Input_Mask'):
        neurons.add_analysis_module(Static_Classification(name='input class', classes=neurons.Input_Mask))

    net['exc_neurons', 0].b = 0.0
    net['exc_neurons', 0].t = net['exc_neurons', 0].target_activity*2# 0.04

    # modify some modules
    # my_modules[UI_sidebar_activity_module].__init__(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})#
    my_modules[multi_group_plot_tab].__init__(['output|target_activity|b|t', '_activity', 'sensitivity', 'linh', 'input_GABA'])  #buffers["output"][1]|linh 'input_isi_inh' , 'total_dw*1000' , 'sliding_average_activity|target_activity'
    my_modules[single_group_plot_tab].__init__(['output', '_activity', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity', 'weight_norm_factor'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons')

    # launch ui
    Network_UI(net, modules=my_modules, label=net.tags[0], storage_manager=sm, group_display_count=len(net.NeuronGroups), reduced_layout=False).show()

def save_imgs(it, net):
    neurons=net['exc_neurons', 0]
    array = np.zeros([neurons.height, neurons.width, 3], dtype=np.uint8)

    base_img=np.array(neurons['Neuron_Classification_Colorizer', 0].get_color_list(neurons.Input_Mask*1, format='[r,g,b]'))

    array[:, :, 0] += uint8(np.clip(base_img[:,0].reshape(neurons.height, neurons.width)+neurons.output.reshape(neurons.height, neurons.width)*255,0,255))
    array[:, :, 1] += uint8(np.clip(base_img[:,1].reshape(neurons.height, neurons.width)+neurons.output.reshape(neurons.height, neurons.width)*255,0,255))
    array[:, :, 2] += uint8(np.clip(base_img[:,2].reshape(neurons.height, neurons.width)+neurons.output.reshape(neurons.height, neurons.width)*255,0,255))

    im = Image.fromarray(array, 'RGB')
    im = im.resize((neurons.width*10,neurons.height*10), Image.NEAREST)
    im.save(net.sm.absolute_path + "img"+ str(neurons.iteration) +".png")
    #im.show()




def measure_stability_score(net):
    neurons = net['exc_neurons', 0]
    net.recording_off()
    net.simulate_iterations(100000,100)
    net.recording_on()
    steps=10000
    net.simulate_iterations(steps, 100)

    score = (steps/100)-np.sum(np.power(neurons['np.mean(n.output)', 0, 'np'] - neurons.target_activity, 2))

    return score