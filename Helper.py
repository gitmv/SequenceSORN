from PymoNNto.Exploration.Evolution.Interface_Functions import *
from PymoNNto import *

from PymoNNto.Exploration.AnalysisModules import *

def ex_file_name():
    return os.path.basename(sys.argv[0]).replace('.py','')

def n_unique_chars(grammar):
    return len(set(''.join(grammar).replace('#','')))

def n_chars(grammar):
    return len(''.join(grammar)) #1.0 / ...

def get_random_sentences(n_sentences):
    sentences = [' fox eats meat.', ' boy drinks juice.', ' penguin likes ice.', ' man drives car.', ' the fish swims.', ' plant loves rain.', ' parrots can fly.']
    return sentences[0:n_sentences]

def get_char_sequence(n_chars):
    sequence = '. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[](){}<>'
    return [sequence[0:n_chars]]

def get_long_text():
    return [' fox eats meat. boy drinks juice. penguin likes ice.']


def train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=None):
    net.deactivate_behaviors('TextReconstructor')

    net.simulate_iterations(input_steps, 100)

    # deactivate Input
    net.deactivate_behaviors('STDP')
    net.deactivate_behaviors('Normalization')
    net.deactivate_behaviors('TextActivator')

    #ensure synapses are normalized
    for neurons in net.NeuronGroups:
        for b in neurons['Normalization']:
            b.exec_every_x_step=1
            b.iteration(neurons)

    net.simulate_iterations(recovery_steps, 100)

    # text generation
    net.exc_neurons1.add_behavior(90, Recorder(variables=['np.mean(n.output)']))
    net.activate_behaviors('TextReconstructor')
    tr = net['TextReconstructor', 0]
    tr.clear_history()
    net.simulate_iterations(free_steps, 100)

    # scoring
    txt_score = net['TextGenerator', 0].get_text_score(tr.reconstruction_history)
    mean = net.exc_neurons1['np.mean(n.output)', 0, 'np']
    osc_score = np.mean(net.exc_neurons1.target_activity-np.abs(mean - net.exc_neurons1.target_activity))/net.exc_neurons1.target_activity
    if osc_score<0:
        osc_score=0.00001
    dist_score, classes = get_class_score(net)

    set_score(txt_score * osc_score * dist_score, info={
        'text': tr.reconstruction_history,
        'osc_score': osc_score,
        'txt_score': txt_score,
        'dist_score': dist_score,
        'classes': str(classes),
        'simulated_iterations': net.iteration
    }, sm=sm)

class Weight_Classifier_PreT(Classifier_base):

    def get_data_matrix(self, neurons):
        syn_tag = self.parameter('syn_tag', 'EE')
        return neurons.afferent_synapses[syn_tag][0].W #get_partitioned_synapse_matrix(neurons, syn_tag, 'W')

def get_class_score(net):
    if net['ES', 0] is None:
        return 1, []

    wcp = Weight_Classifier_PreT(net.exc_neurons1, syn_tag='ES')
    tg = net['TextGenerator', 0]

    classification = wcp(sensitivity=3.0)
    classes = []
    for i in range(len(tg.alphabet)):
        classes.append(np.sum(np.equal(classification, i + 1)))
    classes = -np.sort(-np.array(classes))

    cw = -np.sort(-tg.char_weighting)
    cw = net.exc_neurons1.size / len(tg.alphabet) * cw
    score = 1.0 - np.sum(np.abs(classes - cw)) / net.exc_neurons1.size / 2.0

    #print(np.array2string(classes, separator=", "), net.iteration, score)  # repr: with commas

    return score, classes

def train_ES_and_get_distribution_score(net, input_steps, sm=None, deactivateEE=True):
    if deactivateEE:
        net.deactivate_behaviors('STDP_EE')
        net.deactivate_behaviors('Norm_EE')

    net.simulate_iterations(input_steps, 100)

    score, classes = get_class_score(net)

    set_score(score, info={'classes': str(classes)})

    #print(classes)
    #print(cw)
    #print(score)

    # cw = -np.sort(-tg.char_weighting)*net.exc_neurons1.size*net.exc_neurons1.target_activity

    #print(net.exc_neurons1.target_activity)
    #print(np.sum(classes))
    #print(np.sum(cw))

    #classes = -np.sort(-np.array(classes))
    #cw = -np.sort(-tg.char_weighting)#*net.exc_neurons1.size*net.exc_neurons1.target_activity
    #classes = classes/np.sum(classes)*len(tg.alphabet)
    #score = 1.0 - np.sum(np.abs(classes-cw))/len(tg.alphabet)/2.0
    #set_score(score, info={'classes': str(classes)})

    #import matplotlib.pyplot as plt
    #plt.barh(np.arange(len(cw)), cw, height=0.8)
    #plt.barh(np.arange(len(classes)), classes, height=0.5)
    #plt.show()





def plot_output_trace(data, plastic_steps, recovery_steps, target_activity, w=500, g=200):
    w2=int(w/2)

    plt.hlines(target_activity,-g, (w+g)*3+w+g, colors='k')

    plt.vlines((w+g)+w2, -0.1, 0.1, colors='k')
    plt.vlines((w+g)*2+w2, -0.1, 0.1, colors='k')

    plt.plot(np.arange(0, w), data[0:w])
    plt.plot(np.arange(0, w)+(w+g), data[plastic_steps-w2:plastic_steps+w2])
    plt.plot(np.arange(0, w)+(w+g)*2, data[plastic_steps+recovery_steps-w2:plastic_steps+recovery_steps+w2])
    plt.plot(np.arange(0, w)+(w+g)*3, data[-w:])
    plt.show()

def save_trace(it, net):
    neurons = net['exc_neurons1', 0]
    a = net['np.mean(n.output)',0][-500:]

    max_a = neurons.target_activity * 2.0#np.maximum(np.max(a), neurons.target_activity * 2.0)
    min_a = 0.0#np.min(a)

    pps = 1  # pixels_per_step

    w = (len(a) - 1) * pps
    h = 100 * 5

    array = np.zeros([h, w, 3], dtype=np.uint8) + 255
    im = Image.fromarray(array, 'RGB')
    draw = ImageDraw.Draw(im)

    def f(y):
        return h - (y - min_a) / (max_a - min_a) * h

    temp = f(a[0])
    for i, b in enumerate(a[1:]):
        pos = f(b)
        draw.line((i * pps, temp, (i + 1) * pps, pos), fill=(0, 0, 0))
        temp = pos

    #draw.line((0, f(0), w, f(0)), fill=(255, 0, 0))
    draw.line((0, f(neurons.target_activity), w, f(neurons.target_activity)), fill=(0, 0, 255))
    #draw.line((0, f(neurons.target_activity * 2.0), w, f(neurons.target_activity * 2.0)), fill=(0, 255, 0))

    im.save(net.sm.absolute_path + "act"+ str(neurons.iteration) +".png")


def generate_response_images(net, input_steps, recovery_steps, free_steps):
    neurons = net['exc_neurons1', 0]
    neurons.add_analysis_module(Neuron_Classification_Colorizer())

    neurons.add_behavior(100, Recorder(variables=['np.mean(n.output)']))
    #net.add_behaviors_to_object({100:Recorder(variables=['np.mean(n.output)'])}, neurons)

    net.simulate_iterations(input_steps, 501, batch_progress_update_func=save_trace)

    #net.deactivate_behaviors('STDP')
    #net.deactivate_behaviors('Normalization')
    net.deactivate_behaviors('Input')

    net.simulate_iterations(recovery_steps, 201, batch_progress_update_func=save_trace)

    net.simulate_iterations(free_steps, 501, batch_progress_update_func=save_trace)