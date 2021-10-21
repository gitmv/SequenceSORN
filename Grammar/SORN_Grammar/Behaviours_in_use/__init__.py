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
from Grammar.SORN_Grammar.UI_tabs.Reaction_Analysis_Tab import *
from Grammar.SORN_Grammar.UI_tabs.similarity_matrix_tab import *

from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *

from Grammar.SORN_Grammar.Behaviours_in_use.LineActivator import *
from Grammar.SORN_Grammar.Behaviours_in_use.MNISTActivator import *

from Grammar.SORN_Grammar.Behaviours_in_use.Out_Normalization import *

from Grammar.SORN_Grammar.Analysis_Modules import *

blue = (0.0, 0.0, 255.0, 255.0)
red = (255.0, 0.0, 0.0, 255.0)
yellow = (255.0, 150.0, 0.0, 255.0)

def show_UI(SORN, sm, n_groups):
    #all modules in dict
    my_modules = get_modules_dict(
        similarity_matrix_tab(),
        get_default_UI_modules(['output']),
        get_my_default_UI_modules(),
        isi_module_tab(),
        Reaction_Analysis_Tab()
    )

    #modify some modules
    #my_modules[UI_sidebar_activity_module].__init__(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})#
    my_modules[multi_group_plot_tab].__init__(['output|target_activity', 'activity', 'sensitivity', 'buffers["output"][1]|linh'])#'input_isi_inh' , 'total_dw*1000' , 'sliding_average_activity|target_activity'
    my_modules[single_group_plot_tab].__init__(['output', 'activity', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons')

    #launch ui
    Network_UI(SORN, modules=my_modules, label=SORN.tags[0], storage_manager=sm, group_display_count=n_groups, reduced_layout=False).show()



def load_state(SORN, subfolder):
    folder = get_data_folder(create_when_not_found=True) + '/' + subfolder + '/'

    set_partitioned_synapse_matrix(SORN['exc_neurons', 0], 'EE', 'W', np.load(folder + '_EEw.npy'))
    set_partitioned_synapse_matrix(SORN['exc_neurons', 0], 'EE', 'enabled', np.load(folder + '_EEe.npy'))
    # network['EE',0].W = np.load(folder+'_EEw.npy')
    # network['EE',0].enabled = np.load(folder+'_EEe.npy')
    # SORN['IE',0].W = np.load(folder+'_IEw.npy')
    # SORN['IE',0].enabled = np.load(folder+'_IEe.npy')
    # SORN['EI',0].W = np.load(folder+'_EIw.npy')
    # SORN['EI',0].enabled = np.load(folder+'_EIe.npy')
    SORN['exc_neurons', 0].target_activity = np.load(folder + '_ENt.npy')
    SORN['exc_neurons', 0].Input_Weights = np.load(folder + '_ENw.npy')
    SORN['exc_neurons', 0].Input_Mask = np.load(folder + '_ENm.npy')
    SORN['exc_neurons', 0].sensitivity = np.load(folder + '_ENs.npy')

def plot_corellation_matrix(network):
    w = get_partitioned_synapse_matrix(network['exc_neurons', 0], 'GLU', 'W', True)
    return plot_annotated_corellation_matrix_w(network, w)

def plot_corellation_matrix_T(network):
    w = get_partitioned_synapse_matrix(network['exc_neurons', 0], 'GLU', 'W', True).T
    return plot_annotated_corellation_matrix_w(network, w)

def get_correlation_matrix_and_swap(m):
    import scipy.cluster.hierarchy as sch

    def cluster_corr(corr_array, inplace=False):
        pairwise_distances = sch.distance.pdist(corr_array)
        linkage = sch.linkage(pairwise_distances, method='complete')
        cluster_distance_threshold = pairwise_distances.max() / 2
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
        idx = np.argsort(idx_to_cluster_array)

        if not inplace:
            corr_array = corr_array.copy()

        if isinstance(corr_array, pd.DataFrame):
            return corr_array.iloc[idx, :].T.iloc[idx, :], idx

        return corr_array[idx, :][:, idx], idx

    import pandas as pd

    df = pd.DataFrame(m)
    corrMatrix = df.corr()

    return cluster_corr(corrMatrix)

def plot_annotated_corellation_matrix_w(network, w):

    mat, idx = get_correlation_matrix_and_swap(w)

    steps=1
    print('')

    lidx=len(idx)

    labels = []
    for i, id in enumerate(idx):
        if i % steps == 0:
            print('\rrecon: {}/{}'.format(i, lidx), end='')
            res = compute_temporal_reconstruction(network, network['exc_neurons', 0], id, recon_group_tag='exc_neurons')
            res = np.array(res)
            res = res - np.min(res)
            text = generate_text_from_recon_mat(res, network['Text_Generator', 0])
            labels.append(text)

    xaxis = np.arange(0, len(idx), steps)
    # plt.xticks(xaxis, labels[idx])
    plt.yticks(xaxis, labels)

    plt.imshow(mat)
    plt.show()