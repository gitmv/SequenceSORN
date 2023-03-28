from PymoNNto.NetworkBehavior.Structure.Structure import *
import random

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

class Ray_Line():
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

class Box:

    def __init__(self, x, y, w, h, color=(0, 0, 0, 255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color

    def collision(self, box):
            return (self.x < box.x + box.w and
                    self.x + self.w > box.x and
                    self.y < box.y + box.h and
                    self.y + self.h > box.y)

    def get_edge_lines(self):
        return [Ray_Line(self.x, self.y, 0, self.h),
                Ray_Line(self.x, self.y, self.w, 0),
                Ray_Line(self.x + self.w, self.y, 0, self.h),
                Ray_Line(self.x, self.y + self.h, self.w, 0)]

    def get_ray_line_dist(self, ray, line):

        b_div = -ray.dy*line.dx+ray.dx*line.dy
        if b_div != 0:
            b = (ray.dy*(line.x-ray.x)-ray.dx*(line.y-ray.y))/b_div
            if b >= 0 and b <= 1:
                if ray.dx != 0:
                    dist = (line.x-ray.x+b*line.dx) / ray.dx
                    if dist>=0:
                        return dist
                elif ray.dy !=0:
                    dist = (line.y-ray.y+b*line.dy) / ray.dy
                    if dist >= 0:
                        return dist

        return np.inf


    def ray_collision_distance(self, ray):
        return np.min([self.get_ray_line_dist(ray, line) for line in self.get_edge_lines()])


#b = Box(1, 1, 10, 10)
#r = Ray_Line(5, 5, 1, 2)
#print(b.ray_collision_distance(r))

class Maze_vision_behavior(Behavior):

    def initialize(self, neurons):
        self.add_tag('maze_vision_behavior')
        self.maze = self.get_init_attr('maze', None)

    def iteration(self, neurons):

        image_data = []

        for ray in self.maze.rays:
            ray.dist = 1000.0
            ray.collision_box = None
            for box in self.maze.boxes:
                collision = box.ray_collision_distance(ray)
                if collision < ray.dist and collision > 0:
                    ray.dist = collision
                    ray.collision_box = box

            if ray.collision_box is not None:
                image_data.append(ray.collision_box.color[0:3])
            else:
                image_data.append([0, 0, 0])

        #print(image_data)
        #neurons.activity += np.array(image_data).transpose().flatten()



class Maze_sense_behavior(Behavior):

    def initialize(self, neurons):
        self.add_tag('maze_sense_behavior')
        self.maze = self.get_init_attr('maze', None)

    def iteration(self, neurons):
        nsm=self.maze.network_size_mul

        xmin = (self.maze.player.x)*nsm #-(self.maze.maze_w-1)/2
        ymin = (self.maze.player.y)*nsm #-(self.maze.maze_h-1)/2

        xmax = (self.maze.player.x+1)*nsm #-(self.maze.maze_w-1)/2
        ymax = (self.maze.player.y+1)*nsm #-(self.maze.maze_h-1)/2

        mask = (neurons.x >= xmin)*(neurons.x <= xmax)*(neurons.y >= ymin)*(neurons.y <= ymax)
        #print(x,y, np.sum(mask))
        #print(x,y,np.sum(mask))
        #mask=np.random.rand(len(mask))>0.4
        neurons.activity[mask] += 1


class Maze_action_behavior(Behavior):

    def initialize(self, neurons):
        self.add_tag('maze_act_behavior')
        self.maze = self.get_init_attr('maze', None)
        self.right= 0
        self.left = 0
        self.bottom=0
        self.top =  0

    def iteration(self, neurons):

        block_size = int(len(neurons.output)/4)

        #self.right+= np.sum(neurons.output[block_size * 0:block_size * 1])
        #self.left += np.sum(neurons.output[block_size * 1:block_size * 2])
        #self.bottom+=np.sum(neurons.output[block_size * 2:block_size * 3])
        #self.top +=  np.sum(neurons.output[block_size * 3:block_size * 4])

        self.right+= np.sum(neurons.output[(neurons.x < 0) * (neurons.y < 0)])
        self.left += np.sum(neurons.output[(neurons.x > 0) * (neurons.y < 0)])
        self.bottom+=np.sum(neurons.output[(neurons.x < 0) * (neurons.y > 0)])
        self.top +=  np.sum(neurons.output[(neurons.x > 0) * (neurons.y > 0)])

        #import matplotlib.pyplot as plt
        #plt.scatter(neurons.x, neurons.y)
        #mask = (neurons.x < 0) * (neurons.y < 0)
        #plt.scatter(neurons.x[mask], neurons.y[mask])
        #plt.show()


        max_act_sum = np.max([self.right, self.left, self.bottom, self.top])

        coll = False

        if neurons.iteration % self.maze.reaction_time == 0:
            if max_act_sum > 1:

                if self.right == max_act_sum:
                    coll = coll or self.maze.move_player(+1, 0)
                elif self.left == max_act_sum:
                    coll = coll or self.maze.move_player(-1, 0)
                elif self.bottom == max_act_sum:
                    coll = coll or self.maze.move_player(0, +1)
                elif self.top == max_act_sum:
                    coll = coll or self.maze.move_player(0, -1)

            self.right = 0
            self.left = 0
            self.bottom = 0
            self.top = 0

        self.maze.collision = coll



class Maze_reward_behavior(Behavior):

    def initialize(self, neurons):
        self.add_tag('Maze_reward_behavior')
        self.maze = self.get_init_attr('maze', None)

    def iteration(self, neurons):

        if self.maze.player.collision(self.maze.goal):
            neurons.activity += 1
            self.maze.reset_player()



class Maze_punishment_behavior(Behavior):

    def initialize(self, neurons):
        self.add_tag('Maze_punishment_behavior')
        self.maze = self.get_init_attr('maze', None)

    def iteration(self, neurons):

        if self.maze.collision:
            neurons.activity += 1




def get_rnd_color():
    color = np.array([np.random.rand(), np.random.rand(), np.random.rand(), 1.0])
    color = color > 0.5
    color = color * 100.0
    color[3] = 255.0
    #color[2] = color[0]
    #color[1] = color[0]
    return color

class Maze:

    def add_Box(self, x, y, w, h, c, split=True):
        if split:
            for w_i in range(w):
                for h_i in range(h):
                    self.boxes.append(Box(x+w_i, y+h_i, 1, 1, get_rnd_color()))
        else:
            self.boxes.append(Box(x, y, w, h, c))

    def reset_player(self):
        self.player.x = 1
        self.player.y = 1
        self.player.last_x = self.player.x
        self.player.last_y = self.player.y


    def __init__(self, level='default', same_color=True):

        self.collision_ray_color=not same_color

        self.boxes = []

        self.reaction_time = 1

        self.ray_count = 32
        self.rays = []
        self.collision = False

        self.network_size_mul = 3

        if level == 'default':
            self.player = Box(1, 1, 1, 1, (0, 0, 255, 255))
            self.player.last_x = self.player.x
            self.player.last_y = self.player.y
            self.goal = Box(8, 1, 1, 1, (255, 0, 0, 255))
            self.maze_w = 10
            self.maze_h = 10

            #for x in range(self.maze_w):
            #    for y in range(self.maze_h):

            self.add_Box(1, 0, 8, 1, get_rnd_color())
            self.add_Box(9, 0, 1, 10, get_rnd_color())
            self.add_Box(0, 0, 1, 10, get_rnd_color())
            self.add_Box(1, 9, 8, 1, get_rnd_color())

            self.add_Box(2, 5, 1, 1, get_rnd_color())
            self.add_Box(5, 1, 3, 1, get_rnd_color())
            self.add_Box(3, 4, 1, 2, get_rnd_color())
            self.add_Box(5, 2, 1, 5, get_rnd_color())
            self.add_Box(7, 5, 1, 1, get_rnd_color())

            if same_color:
                for box in self.boxes:
                    box.color = (100, 100, 100, 255)

            for r_i in range(self.ray_count):
                r = float(r_i) * 2.0 * np.pi / float(self.ray_count) + 0.00001
                x, y = pol2cart(r, 1.0)
                ray = Ray_Line(self.player.x + self.player.w / 2, self.player.y + self.player.h / 2, x, y)
                self.rays.append(ray)



    '''
    def create_activation_Matrix(self, neurons, option_count, neurons_per_option):

        result = np.zeros((neurons.size, option_count))

        for opt in range(option_count):

            mask =

            result[mask, opt] = 1.0



        if hasattr(neurons, 'input_density'):
            density = neurons.input_density
        else:
            density = self.kwargs.get('input_density', 1/60)#int(output_size / 60)

        output_size = neurons.size

        #output_size = self.kwargs.get('output_size', 600)
        # self.kwargs.get('activation_size', int(output_size / 60))

        if density<=1:
            self.activation_size = int(output_size * density)
        else:
            self.activation_size = int(density)

        neurons.Input_Weights = np.zeros((output_size, self.get_alphabet_length()))
        available = set(range(output_size))

        frequency_adjustment = self.kwargs.get('frequency_adjustment', False)
        char_count = self.get_char_input_statistics_list()
        mean_char_count = np.mean(char_count)
        sum_char_count = np.sum(char_count)

        for a in range(self.get_alphabet_length()):

            char_activiation_size=self.activation_size

            if frequency_adjustment:
                char_activiation_size = int(self.activation_size*(char_count[a]/mean_char_count))

                avg_char_act = char_count[a]/sum_char_count
                avg_cluster_red = char_activiation_size/28
                avg_cluster_act = avg_cluster_red * 0.02
                print(self.index_to_char(a), ': ', char_activiation_size, char_count[a], np.round(avg_char_act,decimals=4), np.round(avg_cluster_red,decimals=4), np.round(avg_cluster_act,decimals=4))

            temp = random.sample(available, char_activiation_size)
            neurons.Input_Weights[temp, a] = 1
            available = set([n for n in available if n not in temp])
            assert len(available) > 0, 'Input alphabet too big for non-overlapping neurons'

        neurons.Input_Mask = (np.sum(neurons.Input_Weights, axis=1) > 0)
        neurons.mean_network_input_activity=self.activation_size/output_size
        neurons.add_tag('text_input_group')
    '''





    def random_move(self, repeat_when_collide=True):#TODO:
        #random.choice([0.0, 1.0, -1.0])
        self.move_player(random.choice([0.0, 1.0, -1.0]), random.choice([0.0, 1.0, -1.0]))



    def move_player(self, x, y):

        self.player.last_x = self.player.x
        self.player.last_y = self.player.y

        self.player.x += x
        self.player.y += y

        collision = False
        for box in self.boxes:
            if box.collision(self.player):
                collision = True

        if collision:
            self.player.x = self.player.last_x
            self.player.y = self.player.last_y

        self.player.x = np.clip(self.player.x, 0, self.maze_w - 1)
        self.player.y = np.clip(self.player.y, 0, self.maze_h - 1)

        for ray in self.rays:
            ray.x = self.player.x+self.player.w/2
            ray.y = self.player.y+self.player.h/2

        return collision



    def get_reward_neuron_behavior(self):
        return Maze_reward_behavior(maze=self)

    def get_reward_neuron_dimension(self):
        return NeuronDimension(width=4, height=4, depth=1)


    def get_punishment_neuron_behavior(self):
        return Maze_punishment_behavior(maze=self)

    def get_punishment_neuron_dimension(self):
        return NeuronDimension(width=4, height=4, depth=1)



    def get_vision_neuron_behavior(self):
        return Maze_vision_behavior(maze=self)

    def get_vision_neuron_dimension(self):
        return NeuronDimension(width=self.ray_count, height=3, depth=1)




    def get_location_neuron_behavior(self):
        return Maze_sense_behavior(maze=self)

    def get_location_neuron_dimension(self):
        return NeuronDimension(width=self.maze_w*self.network_size_mul, height=self.maze_h*self.network_size_mul, depth=1, centered=False)

    def get_inhibitory_location_neuron_dimension(self):
        return NeuronDimension(width=int(self.maze_w/2), height=int(self.maze_h/2), depth=1)


    def get_action_neuron_behavior(self):
        return Maze_action_behavior(maze=self)

    def get_action_neuron_dimension(self):
        return NeuronDimension(width=8, height=8, depth=1)


    def get_sensing_neuron_behavior(self):
        return

    def get_sensing_neuron_dimension(self):
        return


    def iteration(self, neurons):
        return


