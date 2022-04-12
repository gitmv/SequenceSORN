from PymoNNto import *


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def getLinePicture2(deg, center_x, center_y, length, width, height):
    im = Image.new('L', (width, height), (0))
    draw = ImageDraw.Draw(im)
    rot_point = rotatearoundpoint((length, 0), deg, (0, 0))
    #x, y = pol2cart(length, np.deg2rad(deg))
    #rot_point=(x,y)
    draw.line((center_x - np.floor(rot_point[0]), center_y - np.floor(rot_point[1]), center_x + np.floor(rot_point[0]), center_y + np.floor(rot_point[1])), fill=255)
    return picture_to_array(im)

class Line_Patterns(Behaviour):

    def set_variables(self, neurons):

        self.strength=self.get_init_attr('strength', 1.0)

        center_x = self.get_init_attr('center_x', 1)#self.kwargs.get('center_x', 1)
        center_y = self.get_init_attr('center_y', 1)#self.kwargs.get('center_y', 1)
        degree = self.get_init_attr('degree', 1)#self.kwargs.get('degree', 1)
        line_length = self.get_init_attr('line_length', 1)#self.kwargs.get('line_length', 1)

        pattern_count=0
        if type(center_x) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(center_x))
        if type(center_y) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(center_y))
        if type(degree) in [list, np.ndarray]: pattern_count = np.maximum(pattern_count, len(degree))
        if pattern_count == 0: pattern_count=1

        if not type(center_x) in [list, np.ndarray]: center_x=np.ones(pattern_count)*center_x
        if not type(center_y) in [list, np.ndarray]: center_y=np.ones(pattern_count)*center_y
        if not type(degree) in [list, np.ndarray]: degree=np.ones(pattern_count)*degree

        self.patterns = []
        self.labels = []
        for i in range(pattern_count):
            self.patterns.append(np.array([getLinePicture2(degree[i], center_x[i], center_y[i], line_length/2, neurons.width, neurons.height)]))
            self.labels.append('deg{}cy{}cx{}'.format(degree[i], center_x[i], center_y[i]))

        self.pattern_count = pattern_count

    def new_iteration(self, neurons):
        pattern_indx = np.random.choice(np.arange(self.pattern_count))
        pattern = self.patterns[pattern_indx][0]
        neurons.activity += pattern.flatten()*self.strength



#[0,45,90,135,180,225]
#[1,2,3,4,5,6,7,8,9]
#[0,45,90,135,180,225,270,315,360]
#net,ng,behaviour = behaviour_test_environment(Line_Patterns(center_x=5, center_y=[1,2,3,4,5,6,7,8,9], degree=0, line_length=9), size=get_squared_dim(121))

#net.initialize()
#net.simulate_iteration()

#import matplotlib.pyplot as plt
#print(ng.width, ng.height)
#plt.imshow(ng.activity.reshape(ng.height, ng.width))
#plt.show()
