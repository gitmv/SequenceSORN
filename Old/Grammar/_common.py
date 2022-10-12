# np.random.seed(1)
# random.seed(1)
from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *

from Old.Grammar.Behaviours_in_use import *

blue = (0.0, 0.0, 255.0, 255.0)
red = (255.0, 0.0, 0.0, 255.0)
yellow = (255.0, 150.0, 0.0, 255.0)


def train_and_generate_text(SORN, plastic_steps, recovery_steps=None, text_gen_steps=5000, sm=None, pretrained=False):
    SORN.simulate_iterations(plastic_steps, 100)

    # deactivate STDP and Input
    SORN.deactivate_behaviours('STDP')
    SORN.deactivate_behaviours('Normalization')
    SORN.deactivate_behaviours('Text_Activator')

    # recovery phase
    if recovery_steps is not None:
        SORN.simulate_iterations(recovery_steps, 100)

    # text generation
    tr = SORN['Text_Reconstructor', 0]
    tr.reconstruction_history = ''
    SORN.simulate_iterations(text_gen_steps, 100)
    print(tr.reconstruction_history)

    # scoring
    score = SORN['Text_Generator', 0].get_text_score(tr.reconstruction_history)
    set_score(score, info={'text': tr.reconstruction_history, 'simulated_iterations': SORN.iteration})


def show_UI(SORN, sm):
    add_all_analysis_modules(SORN['exc_neurons', 0])

    # all modules in dict
    my_modules = get_modules_dict(
        get_default_UI_modules(['output']),
        get_my_default_UI_modules(),
        # isi_module_tab(),
        Reaction_Analysis_Tab(),
        cluster_bar_tab(),
        similarity_matrix_tab()
    )

    SORN['exc_neurons', 0].b = 0.0
    SORN['exc_neurons', 0].t = 0.04

    SORN['exc_neurons', 0].add_analysis_module(Static_Classification(name='input class', classes=SORN['exc_neurons', 0].Input_Mask))

    # modify some modules
    # my_modules[UI_sidebar_activity_module].__init__(1, add_color_dict={'output': (255, 255, 255), 'Input_Mask': (-100, -100, -100)})#
    my_modules[multi_group_plot_tab].__init__(['output|target_activity|b|t', 'activity', 'sensitivity', 'inh'])  #buffers["output"][1]|linh 'input_isi_inh' , 'total_dw*1000' , 'sliding_average_activity|target_activity'
    my_modules[single_group_plot_tab].__init__(['output', 'activity', 'input_GLU', 'input_GABA', 'input_grammar', 'sensitivity', 'weight_norm_factor'], net_lines=[0.02], neuron_lines=[0, 0.5, 1.0])
    my_modules[reconstruction_tab].__init__(recon_groups_tag='exc_neurons')

    # launch ui
    Network_UI(SORN, modules=my_modules, label=SORN.tags[0], storage_manager=sm, group_display_count=len(SORN.NeuronGroups), reduced_layout=False).show()


def new_grammar(n_sentences):
    sentences = [' fox eats meat.', ' boy eats bread.', ' man drinks coffee.', ' man drives car.',
                 ' plant loves rain.', ' parrots can fly.', 'the fish swims']
    return [sentences[i] for i in range(n_sentences)]

#sentences = [' fox eats meat.', ' boy drinks juice.', 'the fish swims', ' plant loves rain.', ' penguin likes ice.', ' parrots can fly.', ' man drives car.']
def get_default_grammar(n_sentences):
    sentences = [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' plant loves rain.', ' parrots can fly.', 'the fish swims']
    return [sentences[i] for i in range(n_sentences)]


def get_bruno_grammar(n_sentences):
    # 64
    subjects = [' man ',
                ' woman ',
                ' girl ',
                ' boy ',
                ' child ',
                ' cat ',
                ' dog ',
                ' fox ']

    verbs = ['eats ',
             'drinks ']

    objects_eat = ['meat.',
                   'bread.',
                   'fish.',
                   'vegetables.']

    objects_drink = ['milk.',
                     'water.',
                     'juice.',
                     'tea.']

    all_sentences = []
    for s in subjects:
        for o in objects_eat:
            all_sentences.append(s + verbs[0] + o)

    for s in subjects:
        for o in objects_drink:
            all_sentences.append(s + verbs[1] + o)

    # 58
    removed_sentences = [' woman drinks milk.',
                         ' fox drinks tea.',
                         ' cat eats vegetables.',
                         ' girl eats meat.',
                         ' child eats fish.',
                         ' boy drinks juice.',
                         ' man drinks water.',
                         ' dog eats bread.',
                         ' woman eats meat.',
                         ' fox eats bread.',
                         ' cat drinks tea.',
                         ' girl drinks juice.',
                         ' child drinks water.',
                         ' boy eats fish.',
                         ' man eats vegetables.',
                         ' dog drinks milk.',
                         ' woman eats vegetables.',
                         ' fox eats fish.',
                         ' cat drinks juice.',
                         ' girl drinks tea.',
                         ' child drinks milk.',
                         ' boy eats bread.',
                         ' man eats meat.',
                         ' dog drinks water.',
                         ' woman drinks water.',
                         ' fox drinks juice.',
                         ' cat eats bread.',
                         ' girl eats fish.',
                         ' child eats vegetables.',
                         ' boy drinks tea.',
                         ' man drinks milk.',
                         ' dog eats meat.',
                         ' woman drinks tea.',
                         ' fox drinks milk.',
                         ' cat eats meat.',
                         ' girl eats vegetables.',
                         ' child eats bread.',
                         ' boy drinks water.',
                         ' man drinks juice.',
                         ' dog eats fish.',
                         ' woman eats fish.',
                         ' fox eats meat.',
                         ' cat drinks milk.',
                         ' girl drinks water.',
                         ' child drinks juice.',
                         ' boy eats vegetables.',
                         ' man eats bread.',
                         ' dog drinks tea.',
                         ' woman drinks juice.',
                         ' fox drinks water.',
                         ' cat eats fish.',
                         ' girl eats bread.',
                         ' child eats meat.',
                         ' boy drinks milk.',
                         ' man drinks tea.',
                         ' dog eats vegetables.']

    n_remove = len(all_sentences)-n_sentences

    for r in range(n_remove):
        if removed_sentences[r] in all_sentences:
            all_sentences.remove(removed_sentences[r])
        else:
            print('not found', removed_sentences[r])

    return all_sentences

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

    SORN.deactivate_behaviours('STDP')
    SORN.deactivate_behaviours('Normalization')
    SORN.deactivate_behaviours('Text_Activator')


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

    steps = 1
    print('')

    lidx = len(idx)

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