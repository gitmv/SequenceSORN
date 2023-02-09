from Text.New2.Behaviour_Core_Modules import *
from UI_Helper import *
from Old.Grammar.Behaviours_in_use.MNISTActivator import *
from Gabor.sidebar_patch_reconstructor_module import *
#from Old.Grammar.Behaviours_in_use.Behaviour_Bar_Activator import *


ui = True
# neuron_count = 2000#1500#2400
# plastic_steps = 30000
# recovery_steps = 10000
# text_gen_steps = 5000

patch_w = 7#7#10
patch_h = 7#7#10
neurons_per_pixel = 5
w_multiply = 4

target_activity = 0.0275#0.075#0.11#0.02#0.00625#0.01#0.025#5#0.05
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)


class Refractory_D(Behaviour):

    def set_variables(self, neurons):
        neurons.refrac_ct = neurons.get_neuron_vec()
        self.steps = self.get_init_attr('steps', 5.0, neurons)

    def new_iteration(self, neurons):
        neurons.refrac_ct = np.clip(neurons.refrac_ct-1.0, 0.0, None)

        neurons.refrac_ct += neurons.output * self.steps

        neurons.activity *= (neurons.refrac_ct <= 1.0)


class Image_Patch_Generator(Behaviour):

    def set_variables(self, network):
        self.add_tag('Input')

        image_path = self.get_init_attr('img_path', '')
        print(image_path)

        network.patch_w = self.get_init_attr('patch_w', 1)
        network.patch_h = self.get_init_attr('patch_h', 1)

        self.patch_min = self.get_init_attr('patch_min', 10)

        pil_image = Image.open(image_path)
        pil_image_gray = pil_image.convert('L')
        #pil_image_gray.show()
        self.image_rgb = np.array(pil_image).astype(np.float64)
        self.white = np.array(pil_image_gray).astype(np.float64)
        self.on_center_white, self.off_center_white = get_LOG_On_Off(self.white)

        self.px = 100.0
        self.py = 100.0

        self.phi = 0

        #plt.matshow(white)
        #plt.show()
        #plt.matshow(on_center_white)
        #plt.show()
        #plt.matshow(off_center_white)
        #plt.show()

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y


    def get_random_frame_position(self, network):
        self.px = np.random.randint(self.white.shape[1] - network.patch_w)
        self.py = np.random.randint(self.white.shape[0] - network.patch_h)
        return int(self.px),int(self.py)

    def get_moving_one_direction_frame_position(self, network):
        self.px += 1#1np.random.randint(self.white.shape[1] - network.patch_w)

        if self.px > self.white.shape[1] - network.patch_w:
            self.px = 0
            self.py = np.random.randint(self.white.shape[0] - network.patch_h)
        #y = np.random.randint(self.white.shape[0] - network.patch_h)
        return int(self.px),int(self.py)

    def get_moving_frame_position(self, network):
        #self.phi += (np.random.rand()-0.5)#/20.0 #

        x, y = self.pol2cart(2, self.phi)#np.deg2rad(np.random.rand()*360)

        self.px += x
        self.py += y

        if self.px < 0: self.px = self.white.shape[1] - network.patch_w
        if self.py < 0: self.py = self.white.shape[0] - network.patch_h

        if self.px > self.white.shape[1] - network.patch_w: self.px = 0
        if self.py > self.white.shape[0] - network.patch_h: self.py = 0

        #self.px = np.clip(self.px+x,0,self.white.shape[1] - network.patch_w)
        #self.py = np.clip(self.py+y,0,self.white.shape[0] - network.patch_h)
        x = int(self.px)
        y = int(self.py)

        return x,y

    def new_iteration(self, network):

        m = -1

        while m < self.patch_min:#10:#20

            #x, y = self.get_random_frame_position(network)
            #x, y = self.get_moving_frame_position(network)
            x,y = self.get_moving_one_direction_frame_position(network)

            network.on_center_white = self.on_center_white[y:y + network.patch_h, x:x + network.patch_w]
            network.off_center_white = self.off_center_white[y:y + network.patch_h, x:x + network.patch_w]

            #neurons.activity += pattern.flatten()*self.strength

            m = np.maximum(np.max(network.on_center_white), np.max(network.off_center_white))
            s = np.sum(network.on_center_white) + np.sum(network.off_center_white)

        network.on_center_white = network.on_center_white.astype('float64') / s / 255.0 * 2000.0#4000.0
        network.off_center_white = network.off_center_white.astype('float64') / s / 255.0 * 2000.0#4000.0

        #network.on_center_white = network.on_center_white.astype('float64') / 255.0 * 10.0
        #network.off_center_white = network.off_center_white.astype('float64') / 255.0 * 10.0

        network.on_off_center_white = np.hstack([network.on_center_white, network.off_center_white])

        self.input_image = np.zeros([network.patch_h, network.patch_w, 3])

        self.input_image[:, :, 0] = network.on_center_white
        self.input_image[:, :, 1] = network.off_center_white

        #plt.imshow(image)
        #plt.show()

        #if m>10:
        #plt.imshow(image)
        #plt.show()



class Image_Patch_Activator(Behaviour):

    def set_variables(self, neurons):
        self.add_tag('Input')

        self.patch_name = self.get_init_attr('patch_name', '')

        self.x = self.get_init_attr('x', 0)
        self.y = self.get_init_attr('y', 0)
        self.strength = self.get_init_attr('strength', 1.0)

        self.npp = neurons.depth#neurons per pixel#self.get_init_attr('neurons_per_pixel', 1)

        neurons.Input_Mask = neurons.shape.get_area_mask(0,patch_w*2,0,patch_h,0,neurons_per_pixel)

        neurons.linh = 1.0

    def new_iteration(self, neurons):
        patch = getattr(neurons.network, self.patch_name) #network.on_center_white #network.off_center_white
        neurons.activity[neurons.Input_Mask] = np.vstack([patch.flatten() for _ in range(self.npp)]).flatten()


class Image_Patch_Reconstructor(Behaviour):

    def reconstruct_image(self, data):
        shaped_data=data.reshape((neurons_per_pixel, patch_h, patch_w*2))
        image = np.zeros([patch_h, patch_w, 3])
        image[:, :, 0] = np.mean(shaped_data[:,:,0:patch_w], axis=0)
        image[:, :, 1] = np.mean(shaped_data[:,:,patch_w:patch_w*2], axis=0)
        return image

    def new_iteration(self, network):
        neurons = network['exc_neurons', 0]

        data = neurons.output[neurons.Input_Mask]
        self.reconstruction_image = self.reconstruct_image(data)




net = Network(tag='Grammar Learning Network', behaviour={
    1: Image_Patch_Generator(strength=1, img_path='../../Images/Lenna_(test_image).png', patch_w=patch_w, patch_h=patch_h),
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
    41: STDP(transmitter='GLU', strength=0.0015),#0.0015
    1: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    2: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent),#[0.614#EXP]

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
    show_UI(net, sm, qa=['Input', 'ff', 'fb'], additional_modules=[sidebar_patch_reconstructor_module()])
else:
    print('TODO implement')
    #train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)#remove
