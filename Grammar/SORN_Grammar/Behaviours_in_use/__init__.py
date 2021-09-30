import numpy as np
import random
#np.random.seed(1)
#random.seed(1)

from Grammar.SORN_Grammar.Behaviours_in_use.Init_Neurons import *
from Grammar.SORN_Grammar.Behaviours_in_use.Init_Synapses import *
from Grammar.SORN_Grammar.Behaviours_in_use.IP import *
from Grammar.SORN_Grammar.Behaviours_in_use.ISI_Inhibition_Module import *
from Grammar.SORN_Grammar.Behaviours_in_use.KWTA import *
from Grammar.SORN_Grammar.Behaviours_in_use.Learning import *
from Grammar.SORN_Grammar.Behaviours_in_use.Output_Generation import *
from Grammar.SORN_Grammar.Behaviours_in_use.Refractory import *
from Grammar.SORN_Grammar.Behaviours_in_use.Synapse_Operation import *
from Grammar.SORN_Grammar.Behaviours_in_use.Text_Activator import *
from Grammar.SORN_Grammar.Behaviours_in_use.Text_Generatior import *
from Grammar.SORN_Grammar.Behaviours_in_use.Text_Reconstructor import *

from Grammar.SORN_Grammar.UI_tabs.isi_module_tab import *
from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *

blue = (0.0, 0.0, 255.0, 255.0)
red = (255.0, 0.0, 0.0, 255.0)
yellow = (255.0, 150.0, 0.0, 255.0)

def show_UI(SORN, sm, n_groups):
    my_modules = get_default_UI_modules(['output'])+get_my_default_UI_modules()+[isi_module_tab()]
    my_modules[0] = UI_sidebar_activity_module(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})#
    my_modules[1] = multi_group_plot_tab(['output|target_activity', 'activity', 'sensitivity', 'buffers["output"][1]|linh'])#'input_isi_inh' , 'total_dw*1000' , 'sliding_average_activity|target_activity'
    my_modules[8] = single_group_plot_tab(['output', 'activity', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    Network_UI(SORN, modules=my_modules, label=SORN.tags[0], storage_manager=sm, group_display_count=n_groups, reduced_layout=False).show()
