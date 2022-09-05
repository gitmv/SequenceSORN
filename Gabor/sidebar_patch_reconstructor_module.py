from PymoNNto.Exploration.Network_UI import *
from PymoNNto.Exploration.Network_UI.Sequence_Activation_Tabs import *
#from UI.Tabs import *
#from UI.Analysis_Modules import *



class sidebar_patch_reconstructor_module(TabBase):

    def initialize(self, Network_UI):

        if Network_UI.network['Image_Patch_Generator', 0] is not None and Network_UI.network['Image_Patch_Reconstructor', 0] is not None:
            # def image_activator_on_off(event):
            #    Network_UI.network['image_act', 0].behaviour_enabled = self.input_select_box.currentText() != 'None'

            image_canvas = pg.GraphicsLayoutWidget()
            image_canvas.ci.setToolTip('...')
            image_canvas.setBackground((255, 255, 255))
            plot = image_canvas.addPlot(row=0, col=0)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            self.image = pg.ImageItem(np.random.rand(291, 291, 3))
            plot.addItem(self.image)

            input_image_canvas = pg.GraphicsLayoutWidget()
            input_image_canvas.ci.setToolTip('...')
            input_image_canvas.setBackground((255, 255, 255))
            plot = input_image_canvas.addPlot(row=0, col=0)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            self.input_image = pg.ImageItem(np.random.rand(291, 291, 3))
            plot.addItem(self.input_image)

            reconstruction_image_canvas = pg.GraphicsLayoutWidget()
            reconstruction_image_canvas.ci.setToolTip('...')
            reconstruction_image_canvas.setBackground((255, 255, 255))
            plot = reconstruction_image_canvas.addPlot(row=0, col=0)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            self.reconstruction_image = pg.ImageItem(np.random.rand(291, 291, 3))
            plot.addItem(self.reconstruction_image)

            osic = pg.GraphicsLayoutWidget()
            osic.ci.setToolTip('...')
            osic.setBackground((255, 255, 255))
            plot = osic.addPlot(row=0, col=0)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            self.osri = pg.ImageItem(np.random.rand(291, 291, 3))
            plot.addItem(self.osri)

            clicked_reconstruction_image_canvas = pg.GraphicsLayoutWidget()
            clicked_reconstruction_image_canvas.ci.setToolTip('...')
            clicked_reconstruction_image_canvas.setBackground((255, 255, 255))
            plot = clicked_reconstruction_image_canvas.addPlot(row=0, col=0)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            self.clicked_reconstruction_image = pg.ImageItem(np.random.rand(291, 291, 3))
            plot.addItem(self.clicked_reconstruction_image)


            Network_UI.Add_Sidebar_Element([image_canvas,input_image_canvas, osic, reconstruction_image_canvas, clicked_reconstruction_image_canvas])

            #self.input_image = Network_UI.Add_Image_Item(False, True, title='', stretch=1)
            #self.reconstruction_image = Network_UI.Add_Image_Item(False, True, title='', stretch=1)
            #self.clicked_reconstruction_image = Network_UI.Add_Image_Item(False, True, title='', stretch=1)



    def update(self, Network_UI):

        if Network_UI.network['Image_Patch_Generator', 0] is not None and Network_UI.network['Image_Patch_Reconstructor', 0] is not None:

            if not Network_UI.update_without_state_change:
                input_image = Network_UI.network['Image_Patch_Generator', 0].input_image
                osreconstruction_image = Network_UI.network['Image_Patch_Reconstructor', 0].one_step_reconstruction_image
                reconstruction_image = Network_UI.network['Image_Patch_Reconstructor', 0].reconstruction_image


                self.input_image.setImage(np.rot90(input_image, k=3), levels=(0, 1))
                self.osri.setImage(np.rot90(osreconstruction_image, k=3), levels=(0, 1))
                self.reconstruction_image.setImage(np.rot90(reconstruction_image, k=3), levels=(0, 1))


            #if Network_UI.get
            #c_recon = np.zeros([5, 5, 3])
            #data = np.reshape(Network_UI.network['inp_neurons_on', 0].efferent_synapses['GLU'][0].W[Network_UI.selected_neuron_id()], (5, 5, 5))#(10, 5, 5)
            #c_recon[:, :, 0] = np.mean(data, axis=0)  # [0,:,:]

            #data = np.reshape(Network_UI.network['inp_neurons_off', 0].efferent_synapses['GLU'][0].W[Network_UI.selected_neuron_id()], (5, 5, 5))
            #c_recon[:, :, 1] = np.mean(data, axis=0)



            W = Network_UI.network['EE', 0].W[Network_UI.selected_neuron_id()]
            img_area=W[Network_UI.network['exc_neurons', 0].Input_Mask]
            c_recon=Network_UI.network['Image_Patch_Reconstructor', 0].reconstruct_image(img_area)

            c_recon /= np.max(c_recon)

            self.clicked_reconstruction_image.setImage(np.rot90(c_recon, k=3), levels=(0, 1))

            gen=Network_UI.network['Image_Patch_Generator', 0]
            img=gen.white.copy()#gen.on_center_white.copy()*20.0
            x,y=int(gen.py),int(gen.px)
            img[x-10:x+Network_UI.network.patch_w+10,y-10:y+Network_UI.network.patch_h+10] = 255
            self.image.setImage(np.rot90(img, k=3), levels=(0, 255))