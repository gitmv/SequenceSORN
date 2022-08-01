from Behaviour_Modules import *
from recurrent_unsupervised_stdp_learning.Exp_helper import *
from Old.Grammar.Behaviours_in_use.MNISTActivator import *
from Old.Grammar.Behaviours_in_use.Behaviour_Bar_Activator import *

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

        while m < 20:

            #x, y = self.get_random_frame_position(network)
            #x, y = self.get_moving_frame_position(network)
            x,y = self.get_moving_one_direction_frame_position(network)

            network.on_center_white = self.on_center_white[y:y + network.patch_h, x:x + network.patch_w]
            network.off_center_white = self.off_center_white[y:y + network.patch_h, x:x + network.patch_w]

            #neurons.activity += pattern.flatten()*self.strength

            m = np.maximum(np.max(network.on_center_white), np.max(network.off_center_white))
            s = np.sum(network.on_center_white) + np.sum(network.off_center_white)

        network.on_center_white = network.on_center_white.astype('float64') / s / 255.0 * 4000.0
        network.off_center_white = network.off_center_white.astype('float64') / s / 255.0 * 4000.0

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

        neurons.linh = 1.0

        #w = neurons.network.patch_w
        #h = neurons.network.patch_h

        #neurons.Input_Mask = (neurons.x < w/2)*(neurons.x >= -w/2)*\
        #                     (neurons.y < h*self.npp/2)*\
        #                     (neurons.z >= -h*self.npp/2)

    def new_iteration(self, neurons):
        patch = getattr(neurons.network, self.patch_name) #network.on_center_white #network.off_center_white

        #neurons.activity += patch.flatten()

        neurons.activity = np.vstack([patch.flatten() for _ in range(self.npp)]).flatten()

        #neurons.activity[neurons.Input_Mask] = patch.flatten()

class Image_Patch_Reconstructor(Behaviour):

    #def set_variables(self, network):
    #    self.r_group = self.get_init_attr('r_group', None)
    #    self.g_group = self.get_init_attr('g_group', None)
    #    self.b_group = self.get_init_attr('b_group', None)

    def new_iteration(self, network):


        self.reconstruction_image = np.zeros([network.patch_h, network.patch_w, 3])

        data = np.reshape(network['inp_neurons_on', 0].output,(5, network.patch_h, network.patch_w))
        self.reconstruction_image[:, :, 0] = np.mean(data, axis=0)#[0,:,:]

        data = np.reshape(network['inp_neurons_off', 0].output,(5, network.patch_h, network.patch_w))
        self.reconstruction_image[:, :, 1] = np.mean(data, axis=0)

        #image[:, :, 1] = g/m*255

        #if m>10:
        #plt.imshow(image)
        #plt.show()

        #network[self.r_group,0].output
        #network[self.r_group,0]
        #network[self.r_group,0]

#imp=Image_Patch(strength=1, img_path='../../Images/Lenna_(test_image).png', x=0,y=0,patch_w=10, patch_h=10)#pexels-photo-275484.jpeg
#imp.set_variables(None)
#for i in range(100):
#    imp.new_iteration(None)


from PymoNNto.Exploration.Network_UI.TabBase import *
from PymoNNto.Exploration.Visualization.Reconstruct_Analyze_Label.Reconstruct_Analyze_Label import *




ui = True
neuron_count = 2000#1500#2400
plastic_steps = 30000
recovery_steps = 10000
text_gen_steps = 5000

patch_w = 5
patch_h = 5
neurons_per_pixel = 5

net = Network(tag='Grammar Learning Network', behaviour={
    1: Image_Patch_Generator(strength=1, img_path='../../Images/Lenna_(test_image).png', patch_w=patch_w, patch_h=patch_h),

    100: Image_Patch_Reconstructor()
})


target_activity = 0.01#0.01#0.025#5#0.05
exc_output_exponent = 0.01 / target_activity + 0.22
inh_output_slope = 0.4 / target_activity + 3.6
LI_threshold = np.tanh(inh_output_slope * target_activity)


inp_target_activity = 0.3#0.01#0.025#5#0.05
inp_exc_output_exponent = 0.01 / target_activity + 0.22
inp_inh_output_slope = 0.4 / target_activity + 3.6
inp_LI_threshold = np.tanh(inh_output_slope * target_activity)


class Out(Behaviour):

    def set_variables(self, neurons):
        neurons.activity = neurons.get_neuron_vec()
        neurons.output = neurons.get_neuron_vec().astype(bool)
        neurons.output_old = neurons.get_neuron_vec().astype(bool)
        neurons.linh=1.0

    def new_iteration(self, neurons):
        neurons.output_old = neurons.output.copy()
        neurons.output = neurons.activity.copy()>neurons.get_neuron_vec('uniform')
        neurons._activity = neurons.activity.copy()  # for plotting
        neurons.activity.fill(0)


class IP_Mul(Behaviour):

    def set_variables(self, neurons):
        self.strength = self.get_init_attr('strength', 0.01, neurons)
        neurons.target_activity = self.get_init_attr('target_activity', 0.02)
        neurons.sensitivity = neurons.get_neuron_vec()

    def new_iteration(self, neurons):
        neurons.sensitivity -= (neurons.output - neurons.target_activity) * self.strength
        neurons.activity *= neurons.sensitivity

NeuronGroup(net=net, tag='inp_neurons_on', size=NeuronDimension(width=patch_w, height=patch_h, depth=neurons_per_pixel), color=black, behaviour={
    10: Image_Patch_Activator(strength=1, patch_name='on_center_white'),
    12: Synapse_Operation(tag='fb', transmitter='GLU', strength=1.0),
    20: Synapse_Operation(transmitter='GABA', strength=-0.1),
    30: IP_Mul(target_activity=inp_target_activity, strength=0.007), #0.02
    41: STDP(transmitter='GLU', strength=0.0015),
    1: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    #50: Generate_Output(exp=exc_output_exponent),#[0.614#EXP]
    50: Generate_Output(exp=inp_exc_output_exponent)
})

NeuronGroup(net=net, tag='inp_neurons_off', size=NeuronDimension(width=patch_w, height=patch_h, depth=neurons_per_pixel), color=black, behaviour={
    10: Image_Patch_Activator(strength=1, patch_name='off_center_white'),
    12: Synapse_Operation(tag='fb', transmitter='GLU', strength=1.0),
    20: Synapse_Operation(transmitter='GABA', strength=-0.1),
    30: IP_Mul(target_activity=inp_target_activity, strength=0.007), #0.02
    41: STDP(transmitter='GLU', strength=0.0015),
    1: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    #50: Generate_Output(exp=exc_output_exponent),#[0.614#EXP]
    50: Generate_Output(exp=inp_exc_output_exponent)
})

NeuronGroup(net=net, tag='inh_neurons_on', size=get_squared_dim(patch_w*patch_h*neurons_per_pixel/10), color=red, behaviour={
    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=0.05),
    # output
    70: Generate_Output_Inh(slope=inp_inh_output_slope, duration=2), #'[20.0#S]'
})

NeuronGroup(net=net, tag='inh_neurons_off', size=get_squared_dim(patch_w*patch_h*neurons_per_pixel/10), color=red, behaviour={
    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=0.05),
    # output
    70: Generate_Output_Inh(slope=inp_inh_output_slope, duration=2), #'[20.0#S]'
})

SynapseGroup(net=net, tag='IE,GLUI', src='inp_neurons_on', dst='inh_neurons_on', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons_on', dst='inp_neurons_on', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='IE,GLUI', src='inp_neurons_off', dst='inh_neurons_off', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons_off', dst='inp_neurons_off', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

NeuronGroup(net=net, tag='exc_neurons', size=get_squared_dim(neuron_count), color=blue, behaviour={#60 30#NeuronDimension(width=10, height=10, depth=1)

    12: Synapse_Operation(tag='ff', transmitter='GLUff', strength=1.0),
    13: Synapse_Operation(transmitter='GLUrc', strength=1.0),

    # inhibitory input
    20: Synapse_Operation(transmitter='GABA', strength=-0.1),

    # stability
    30: Intrinsic_Plasticity(target_activity=target_activity, strength=0.007), #0.02
    31: Refractory_D(steps=4.0),

    # learning
    40: Learning_Inhibition(transmitter='GABA', strength=31, threshold=LI_threshold), #0.377 #0.38=np.tanh(0.02 * 20) , threshold=0.38 #np.tanh(get_gene('S',20.0)*get_gene('TA',0.03))
    41: STDP(transmitter='GLU', strength=0.0015),#0.0015
    1: Normalization(syn_direction='afferent', syn_type='GLU', exec_every_x_step=10),
    2: Normalization(syn_direction='efferent', syn_type='GLU', exec_every_x_step=10),

    #44: Normalization(syn_direction='afferent', syn_type='ES', exec_every_x_step=10),

    # output
    50: Generate_Output(exp=exc_output_exponent),#[0.614#EXP]

    # reconstruction
    #80: Text_Reconstructor()

})

NeuronGroup(net=net, tag='inh_neurons', size=get_squared_dim(neuron_count/10), color=red, behaviour={
    # excitatory input
    60: Synapse_Operation(transmitter='GLUI', strength=1.0),
    # output
    70: Generate_Output_Inh(slope=inh_output_slope, duration=2), #'[20.0#S]'
})

SynapseGroup(net=net, tag='EIn,GLU,GLUff', src='inp_neurons_on', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='EIf,GLU,GLUff', src='inp_neurons_off', dst='exc_neurons', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})


SynapseGroup(net=net, tag='GLU,EIn', src='exc_neurons', dst='inp_neurons_on', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})

SynapseGroup(net=net, tag='GLU,EIf', src='exc_neurons', dst='inp_neurons_off', behaviour={
    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
})


#SynapseGroup(net=net, tag='SE,GLU', src='exc_neurons', dst='inp_neurons', behaviour={
#    1: create_weights(distribution='uniform(0.0,1.0)', density=1.0)
#})

SynapseGroup(net=net, tag='EE,GLU,GLUrc', src='exc_neurons', dst='exc_neurons', behaviour={
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

CWP(net['exc_neurons', 0])

net['exc_neurons', 0].sensitivity+=0.4

#User interface
if __name__ == '__main__' and ui:
    show_UI(net, sm, qa=['Input', 'ff', 'fb'], additional_modules=[sidebar_patch_reconstructor_module()])
else:
    print('TODO implement')
    #train_and_generate_text(net, plastic_steps, recovery_steps, text_gen_steps, sm=sm)#remove
