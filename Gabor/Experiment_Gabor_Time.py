from Behaviour_Core_Modules import *
from Behaviour_Image_Patch_Modules import *
from Behaviour_STDP_Modules import *
from UI_Helper import *
from Gabor.sidebar_patch_reconstructor_module import *
#from Old.Grammar.Behaviours_in_use.Behaviour_Bar_Activator import *


ui = True
# neuron_count = 2000#1500#2400
# plastic_steps = 30000
# recovery_steps = 10000
# text_gen_steps = 5000



target_activity = 0.1#0.0275#0.075#0.11#0.02#0.00625#0.01#0.025#5#0.05
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)



net = Network(tag='Grammar Learning Network', behaviour={
    1: Image_Patch_Generator(strength=1, img_path='../../Images/Lenna_(test_image).png', patch_w=patch_w, patch_h=patch_h, patch_min=20, patch_speed=0.5),#0.1
    100: Image_Patch_Reconstructor()
})

NeuronGroup(net=net, tag='exc_neurons', size=NeuronDimension(width=patch_w*w_multiply, height=patch_h, depth=neurons_per_pixel), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    10: Image_Patch_Activator(strength=1, patch_name='on_off_center_white'),

    12: Synapse_Operation(transmitter='GLU', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-1.0),#

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007), #0.02
    #31: Refractory_D(steps=4.0),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    #41: STDP(transmitter='GLU', strength=0.0015),#0.0015
    41: Complex_STDP(transmitter='GLU', strength=0.0002,#0.0002,
                     LTP=np.array([+0.0, +0.0, +0.0, +0.0, +0.1, +0.2, +0.6, +1.0, +1.0, +1.0, +0.8, +0.7, +0.5, +0.4, +0.2]),
                     LTD=np.array([-0.1, -0.3, -0.4, -0.5, -0.6, -0.6, -0.7, -0.6, -0.6, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1])*0.0,#*1.5
                     #LTP=np.array([+0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.0, +0.5, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0]),
                     #LTD=np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]),#*0.6,
                     plot=False),

    #42: Max_Syn_Size(transmitter='GLU', max=0.05),

    1: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    2: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent),#[0.614#EXP]
    51: Complex_STDP_Buffer()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(net['exc_neurons',0].size/10), color=red, behaviour={
    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),
    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'
})


SynapseGroup(net=net, tag='EE,GLU', src='exc_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons', dst='inh_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})



sm = StorageManager(net.tags[0], random_nr=True)
net.initialize(info=True, storage_manager=sm)

#CWP(net['exc_neurons', 0])

#net.exc_neurons.target_activity *= net.exc_neurons.get_neuron_vec('ones')
#net.exc_neurons.target_activity[net.exc_neurons.Input_Mask] = 0.05#0.1
#net.exc_neurons.target_activity[np.invert(net.exc_neurons.Input_Mask)] = 0.005#0.05

net.exc_neurons.sensitivity+=0.5

#from PymoNNto.Exploration.Network_UI.TabBase import *
#from PymoNNto.Exploration.Visualization.Reconstruct_Analyze_Label.Reconstruct_Analyze_Label import *

#User interface
if __name__ == '__main__' and ui:
    show_UI(net, sm, qa=['Input', 'Complex_STDP'], additional_modules=[sidebar_patch_reconstructor_module()])
else:
    print('TODO implement')
    #train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)#remove
