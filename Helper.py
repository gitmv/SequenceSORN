from PymoNNto.Exploration.Evolution.Interface_Functions import *
from PymoNNto import *

from PymoNNto.Exploration.AnalysisModules import *

def train_and_generate_text(net, input_steps, recovery_steps, free_steps, sm=None, pretrained=False):

    net.simulate_iterations(input_steps, 100)

    # deactivate Input
    net.deactivate_behaviours('STDP')
    net.deactivate_behaviours('Normalization')
    net.deactivate_behaviours('Text_Activator')

    net.simulate_iterations(recovery_steps, 100)

    # text generation
    net.exc_neurons.add_behaviour(90, Recorder(variables=['np.mean(n.output)']))

    tr = net['Text_Reconstructor', 0]
    tr.reconstruction_history = ''
    net.simulate_iterations(free_steps, 100)
    print(tr.reconstruction_history)

    # scoring
    txt_score = net['Text_Generator', 0].get_text_score(tr.reconstruction_history)

    mean = net.exc_neurons['np.mean(n.output)', 0, 'np']
    osc_score = np.mean(net.exc_neurons.target_activity-np.abs(mean - net.exc_neurons.target_activity))/net.exc_neurons.target_activity

    if osc_score<0:
        osc_score=0.00001

    dist_score, classes = get_class_score(net)

    set_score(txt_score * osc_score * dist_score, info={
        'osc_score': osc_score,
        'txt_score': txt_score,
        'dist_score': dist_score,
        'classes': str(classes),
        'text': tr.reconstruction_history,
        'simulated_iterations': net.iteration
    }, sm=sm)

def get_class_score(net):
    if net['ES', 0] is None:
        return 1, []

    wcp = Weight_Classifier_Pre(net.exc_neurons, syn_tag='ES')
    tg = net['Text_Generator', 0]

    classification = wcp(sensitivity=3.0)
    classes = []
    for i in range(len(tg.alphabet)):
        classes.append(np.sum(np.equal(classification, i + 1)))
    classes = -np.sort(-np.array(classes))

    cw = -np.sort(-tg.char_weighting)
    cw = net.exc_neurons.size / len(tg.alphabet) * cw
    score = 1.0 - np.sum(np.abs(classes - cw)) / net.exc_neurons.size / 2.0

    print(np.array2string(classes, separator=", "), net.iteration, score)  # repr: with commas

    return score, classes

def train_ES_and_get_distribution_score(net, input_steps, sm=None, deactivateEE=True):
    if deactivateEE:
        net.deactivate_behaviours('STDP_EE')
        net.deactivate_behaviours('Norm_EE')

    net.simulate_iterations(input_steps, 100)

    score, classes = get_class_score(net)

    set_score(score, info={'classes': str(classes)})

    #print(classes)
    #print(cw)
    #print(score)

    # cw = -np.sort(-tg.char_weighting)*net.exc_neurons.size*net.exc_neurons.target_activity

    #print(net.exc_neurons.target_activity)
    #print(np.sum(classes))
    #print(np.sum(cw))

    #classes = -np.sort(-np.array(classes))
    #cw = -np.sort(-tg.char_weighting)#*net.exc_neurons.size*net.exc_neurons.target_activity
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
    neurons = net['exc_neurons', 0]
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
    neurons = net['exc_neurons', 0]
    neurons.add_analysis_module(Neuron_Classification_Colorizer())

    neurons.add_behaviour(100, Recorder(variables=['np.mean(n.output)']))
    #net.add_behaviours_to_object({100:Recorder(variables=['np.mean(n.output)'])}, neurons)

    net.simulate_iterations(input_steps, 501, batch_progress_update_func=save_trace)

    #net.deactivate_behaviours('STDP')
    #net.deactivate_behaviours('Normalization')
    net.deactivate_behaviours('Input')

    net.simulate_iterations(recovery_steps, 201, batch_progress_update_func=save_trace)

    net.simulate_iterations(free_steps, 501, batch_progress_update_func=save_trace)