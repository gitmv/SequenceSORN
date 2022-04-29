from PymoNNto.Exploration.Evolution.Interface_Functions import *

def train_and_generate_text(net, plastic_steps, recovery_steps=None, text_gen_steps=5000, sm=None, pretrained=False):
    net.simulate_iterations(plastic_steps, 100)

    # deactivate STDP and Input
    net.deactivate_mechanisms('STDP')
    net.deactivate_mechanisms('Normalization')
    net.deactivate_mechanisms('Text_Activator')

    # recovery phase
    if recovery_steps is not None:
        net.simulate_iterations(recovery_steps, 100)

    # text generation
    tr = net['Text_Reconstructor', 0]
    tr.reconstruction_history = ''
    net.simulate_iterations(text_gen_steps, 100)
    print(tr.reconstruction_history)

    # scoring
    score = net['Text_Generator', 0].get_text_score(tr.reconstruction_history)
    set_score(score, sm, info={'text': tr.reconstruction_history, 'simulated_iterations': net.iteration})