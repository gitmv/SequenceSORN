import numpy as np
import random
#np.random.seed(1)
#random.seed(1)

from Grammar.SORN_Grammar.Behaviours_in_use.SORN_experimental import *
from Grammar.SORN_Grammar.Behaviours_in_use.SORN_WTA import *
from Grammar.SORN_Grammar.Behaviours_in_use.Text_Generation import *
from Grammar.SORN_Grammar.Behaviours_in_use.Text_Activator import *
from Grammar.SORN_Grammar.Behaviours_in_use.Text_Reconstructor import *
from Grammar.SORN_Grammar.Behaviours_in_use.ip_new import *
from Grammar.SORN_Grammar.Behaviours_in_use.Synapse_Operation import *
from Grammar.SORN_Grammar.Behaviours_in_use.init_neuron import *
from Grammar.SORN_Grammar.Behaviours_in_use.Refractory import *
from Grammar.SORN_Grammar.Behaviours_in_use.threshold_output import *
from Grammar.SORN_Grammar.Behaviours_in_use.isi_reaction_module import *
from Grammar.SORN_Grammar.Behaviours_in_use.activation_function_output import *

from Grammar.SORN_Grammar.Behaviours_in_use.random_activity import *

from Grammar.SORN_Grammar.UI_tabs.isi_module_tab import *

from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *

blue = (0.0, 0.0, 255.0, 255.0)
red = (255.0, 0.0, 0.0, 255.0)
yellow = (255.0, 150.0, 0.0, 255.0)

def show_UI(SORN, sm, n_groups):

    #for ng in SORN.NeuronGroups:


    my_modules = get_default_UI_modules(['output'])+get_my_default_UI_modules()+[isi_module_tab()]
    my_modules[0] = UI_sidebar_activity_module(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})#
    my_modules[1] = multi_group_plot_tab(['output|target_activity', 'activity', 'sensitivity', 'sliding_average_activity|target_activity'])#'input_isi_inh'
    my_modules[8] = single_group_plot_tab({'activity': (0, 0, 0), 'input_GLU': (0, 0, 255), 'input_GABA': (255, 0, 0), 'input_grammar': (255, 0, 255),'sensitivity': (0, 255, 0), 'input_isi_inh': (255, 255, 0)})
    Network_UI(SORN, modules=my_modules, label=SORN.tags[0], storage_manager=sm, group_display_count=n_groups, reduced_layout=False).show()

    # , 'nox', 'refractory_counter', 'refrac_ct', 'input_GLU', 'input_GABA', 'nox' 'input_GABA', 'exhaustion_value'
    # , 'nox', 'refractory_counter' #, 'exhaustion_value', 'weight_norm_factor'