from PymoNNto import *

mnist_folder = '../../../EMNIST'
import numpy as np

#from mnist.loader import MNIST  # pip install python-mnist

def draw_mnist(mnist_img, width, height, centerx, centery):
    h = mnist_img.shape[0]
    w = mnist_img.shape[1]
    result = np.zeros((height, width))
    x = centerx-int(w/2)
    y = centery-int(h/2)
    result[y:y+h, x:x+w] += mnist_img
    return result

class MNIST_Patterns(Behaviour):

    def get_images(self, mnist_pictures, indx):
        return np.array(mnist_pictures[indx]).reshape(28, 28)

    def set_variables(self, neurons):

        mndata = MNIST(mnist_folder)
        mndata.select_emnist('balanced')
        mnist_pictures, mnist_labels = mndata.load_testing()

        self.strength=self.get_init_attr('strength', 1.0)

        center_x = self.get_init_attr('center_x', 1)
        center_y = self.get_init_attr('center_y', 1)

        self.class_ids = self.get_init_attr('class_ids', np.array(list(range(26)))+10)#all chars upper case
        patterns_per_class = self.get_init_attr('pattern_count', 1)

        self.chars = {}
        for c in self.class_ids:
            self.chars[c] = []
            ct = 0
            i = 0
            while ct < patterns_per_class:
                if mnist_labels[i] == c:
                    mnist_img = self.get_images(mnist_pictures, i)
                    self.chars[c].append(draw_mnist(mnist_img, neurons.width, neurons.height, center_x, center_y))
                    ct += 1
                i += 1



    def new_iteration(self, neurons):
        char_indx = np.random.choice(self.class_ids)
        pattern_indx = np.random.choice(np.arange(len(self.chars[char_indx])))
        pattern = self.chars[char_indx][pattern_indx]
        neurons.activity += pattern.flatten()*self.strength
        #plt.matshow(pattern)
        #plt.show()



def get_LOG_On_Off(data, sigma=1):
    import scipy.ndimage as ndimage
    LOG = ndimage.gaussian_laplace(data.astype(np.int32), sigma=sigma)
    on_center = (LOG > 0) * LOG
    off_center = (LOG < 0) * LOG * -1
    return on_center, off_center



class MNIST_Patterns_On_Off(MNIST_Patterns):

    def get_images(self, mnist_pictures, indx):
        data = np.array(mnist_pictures[indx]).reshape(28, 28)

        on, off = get_LOG_On_Off(data, 1)

        res = np.concatenate([on, off], axis=1)

        #plt.matshow(res)
        #plt.show()

        return res