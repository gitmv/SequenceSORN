from PymoNNto import *


class Refractory_D(Behavior):

    def initialize(self, neurons):
        neurons.refrac_ct = neurons.get_neuron_vec()
        self.steps = self.get_init_attr('steps', 5.0, neurons)

    def iteration(self, neurons):
        neurons.refrac_ct = np.clip(neurons.refrac_ct-1.0, 0.0, None)

        neurons.refrac_ct += neurons.output * self.steps

        neurons.activity *= (neurons.refrac_ct <= 1.0)


class Image_Patch_Generator(Behavior):

    def initialize(self, network):
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

    def iteration(self, network):

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



class Image_Patch_Activator(Behavior):

    def initialize(self, neurons):
        self.add_tag('Input')

        self.patch_name = self.get_init_attr('patch_name', '')

        self.x = self.get_init_attr('x', 0)
        self.y = self.get_init_attr('y', 0)
        self.strength = self.get_init_attr('strength', 1.0)

        self.npp = neurons.depth#neurons per pixel#self.get_init_attr('neurons_per_pixel', 1)

        neurons.Input_Mask = neurons.shape.get_area_mask(0,patch_w*2,0,patch_h,0,neurons_per_pixel)

        neurons.linh = 1.0

    def iteration(self, neurons):
        patch = getattr(neurons.network, self.patch_name) #network.on_center_white #network.off_center_white
        neurons.activity[neurons.Input_Mask] = np.vstack([patch.flatten() for _ in range(self.npp)]).flatten()


class Image_Patch_Reconstructor(Behavior):

    def reconstruct_image(self, data):
        shaped_data=data.reshape((neurons_per_pixel, patch_h, patch_w*2))
        image = np.zeros([patch_h, patch_w, 3])
        image[:, :, 0] = np.mean(shaped_data[:,:,0:patch_w], axis=0)
        image[:, :, 1] = np.mean(shaped_data[:,:,patch_w:patch_w*2], axis=0)
        return image

    def iteration(self, network):
        neurons = network['exc_neurons', 0]

        data = neurons.output[neurons.Input_Mask]
        self.reconstruction_image = self.reconstruct_image(data)