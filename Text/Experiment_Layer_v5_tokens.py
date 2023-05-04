from PymoNNto import *
from Behavior_Core_Modules_v5 import *
from Behavior_Text_Modules_v5 import *
from Helper import *

ui = False
n_exc_neurons = 2400
n_inh_neuros = n_exc_neurons/10


#3.2. Behaviour
#Behaviour modules allow to define custom dynamics of neurons and synapses. Each Behaviour module typically consists of two functions: set_variables is called when the Network is initialized and new_iteration is called every time step. Both functions receive an additional attribute, in code block 2 it is named neurons, which points to the group the behaviour belongs to, in this case a NeuronGroup. This attribute allows to use parent group specific functions and to define and modify its variables. In this example, we initialize the NeuronGroup variable voltage with zero values via the get_neuron_vec function. At every timestep, we add random membrane noise to these voltages with get_neuron_vec(“uniform,”…). Further, we define a local variable threshold, defining the voltage above which the neuron will create a spike before being reset, as well as the variable leak_factor for the voltage reduction at each iteration. Here, it is not relevant whether variables are stored in the neuron- or the behaviour-object. Though, in more complex simulations it can be advantageous to store variables only used by the behaviour in the Behaviour object and other variables in the parent object.

txt = ["""
from PymoNNto import *
My_Network = Network()
My_Neurons = NeuronGroup(net=My_Network, tag='my_neurons', size=100)
SynapseGroup(net=My_Network, src=My_Neurons, dst=My_Neurons, tag='GLUTAMATE')
My_Network.initialize()
My_Network.simulate_iterations(1000)

class Basic_Behaviour(Behaviour):

 def set_variables(self, neurons):
  neurons.voltage = neurons.get_neuron_vec()
  self.threshold = 0.5
  self.leak_factor = self.get_init_attr('leak_factor', 0.9, neurons)

 def new_iteration(self, neurons):
  neurons.spike = neurons.voltage > self.threshold
  neurons.voltage[neurons.spike] = 0.0 #reset

  neurons.voltage *= self.leak_factor #voltage decay
  neurons.voltage += neurons.vector('uniform',density=0.01)
"""]



'''
txt = ["""First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.
"""]


txt=[
'The Python Modular Neural Network Toolbox (PymoNNto) provides a versatile and adaptable Python-based framework to develop and investigate brain-inspired neural networks. ',
'In contrast to other commonly used simulators such as Brian2 and NEST, PymoNNto imposes only minimal restrictions for implementation and execution. ',
'The basic structure of PymoNNto consists of one network class with several neuron- and synapse-groups. ',
'The behaviour of each group can be flexibly defined by exchangeable modules. ',
'The implementation of these modules is up to the user and only limited by Python itself. ' ,
'Behaviours can be implemented in Python, Numpy, Tensorflow, and other libraries to perform computations on CPUs and GPUs. ',
'PymoNNto comes with convenient high level behaviour modules, allowing differential equation-based implementations similar to Brian2, and an adaptable modular Graphical User Interface for real-time observation and modification of the simulated network and its parameters.',

]


'Simulating neural networks has become an indispensable part of brain research, allowing neuroscientists to efficiently develop, explore, and evaluate hypotheses. ',
'Working with such models is facilitated by various simulation environments, which typically provide high level classes and functions for convenient model generation, simulation, and analysis. ',

'Each simulation environment has particular strengths and limitations. Neural network models can be formulated at different levels of detail/abstraction. ',
'Reflecting the various scales of investigation, several simulation environments exist, each with its own focus area. ',
'While for example, Neuron excels at simulating neurons with a high degree of biological detail, NEST is optimized to simulate large networks of rather simplified spiking neurons on distributed computing clusters. ',
'Another simulator, Brian/Brian2 prioritizes concise model definition over scaling to large computing environments. ',

'Typically, the convenience provided by a particular neural network simulation toolbox comes at the price of reduced flexibility. ',
'This can cause problems when researchers need to leave the “comfort zone” of a particular simulator. ',
'For example, when aiming to explore a novel plasticity rule, investigators may be confronted with a difficult choice: ',
'They either have to work their way around the constraints of the simulator or write their own simulation environment from scratch. ',
'While implementing a workaround may turn out to be arduous and complicated, writing a simulation environment from scratch is time consuming, error prone, hampering reproducibility, and sacrificing useful features of mature simulation environments. ',

'The scientific community has become increasingly aware of this dilemma. ',
'Several developments aim to increase the flexibility of existing simulators. ',
'For example, NEST has been extended with its own modeling language to allow for custom model definition without having to write C++ modules. ',
'Brian2 simulations, limited to a single core, can be accelerated by executing them on GPUs via automated code translation to GeNN. ',
'However, in all cases, specific simulator-inherent restrictions remain. ',

'An alternative strategy to achieve both flexibility and reproducibility is to detach model definition from its execution. ',
'Simulator-independent model description interfaces, such as PyNN or general model description languages, such as NeuroML, allow to first specify a model using a fixed set of vocabulary and syntax. ',
'In a second step, model definition is automatically translated to a selected simulation environment. In either approach flexibility remains bounded: ',
'The ability to express new mechanisms is limited by a finite number of language elements and the restrictions of the available simulation environments. ',

'To address the dilemma between flexibility and convenience with a novel approach, we designed PymoNNto as a modular low level Python framework with minimal restrictions, while at the same time providing several high level modules for convenient analysis and interaction (see Figure 1 for an overview of PymoNNtos key features and core structure). ',
'Its lightweight structure comes with a number of advantages: ',
'(1) Dynamics of neurons and synapses can be freely designed by custom behaviour modules. ',
'(2) The content of behaviour modules is only limited by the expressive power of Python. ',
'(3) These modules can be optimized for speed, for example via Tensorflow or Cython, and can even wrap around and combine established simulators, facilitating multi-scale approaches. ',
'Without sacrificing flexibility, PymoNNto allows for efficient implementation and analysis via a multitude of features, such as a powerful and extendable graphical user interface, a storage manager, and several pre-implemented neuronal/synaptic mechanisms and network models (compare Table 1). '
]

'''

Generator = TokenTextGenerator(text_blocks=[''.join(txt)])#txt

print(Generator.n_unique_tokens)
print(Generator.n_tokens)

target_act = 1/Generator.n_tokens

net = Network(tag=ex_file_name(), settings=settings)

NeuronGroup(net=net, tag='inp_neurons', size=Grid(width=10, height=Generator.n_unique_tokens, depth=1, centered=False), color=green, behavior={
    # text input
    10: Generator,

    # group output
    50: Output_TextActivator(),

    # text reconstruction
    80: TokenTextReconstructor()
})

NeuronGroup(net=net, tag='exc_neurons1', size=getGrid(n_exc_neurons), color=blue, behavior={
    # weight normalization
    3: Normalization(direction='afferent and efferent', syn_type='DISTAL', exec_every_x_step=200),#watch out when using higher STDP speeds!
    3.1: Normalization(direction='afferent', syn_type='SOMA', exec_every_x_step=200),

    # excitatory and inhibitory input
    12: SynapseOperation(transmitter='GLU', strength=1.0),
    20: SynapseOperation(transmitter='GABA', strength=-1.0),

    # stability
    30: IntrinsicPlasticity(target_activity=target_act, strength=0.008735764741458582, init_sensitivity=0),

    # learning
    40: LearningInhibition(transmitter='GABA', strength=6.450234496564654, avg_inh=0.3427857658747104, min=-0.15), #(optional)(higher sccore/risky/higher spread)
    41: STDP(transmitter='GLU', strength=0.0030597477411211885),

    # group output
    51: Output_Excitatory(exp=0.7378726012049153, mul=2.353594052973287),
})

NeuronGroup(net=net, tag='inh_neurons1', size=getGrid(n_inh_neuros), color=red, behavior={
    # excitatory input
    60: SynapseOperation(transmitter='GLUI', strength=1.0),

    # group output
    70: Output_Inhibitory(avg_inh=0.3427857658747104, target_activity=target_act, duration=2),
})

SynapseGroup(net=net, tag='ES,GLU,SOMA', src='inp_neurons', dst='exc_neurons1', behavior={
    1: CreateWeights(nomr_fac=10)
})

SynapseGroup(net=net, tag='EE,GLU,DISTAL', src='exc_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights(normalize=False)
})

SynapseGroup(net=net, tag='IE,GLUI', src='exc_neurons1', dst='inh_neurons1', behavior={
    1: CreateWeights()
})

SynapseGroup(net=net, tag='EI,GABA', src='inh_neurons1', dst='exc_neurons1', behavior={
    1: CreateWeights()
})

sm = StorageManager(net.tag, random_nr=True)
sm.backup_execued_file()

net.initialize(info=True, storage_manager=sm)

#User interface
if __name__ == '__main__' and ui and not is_evolution_mode():
    from UI_Helper import *
    show_UI(net, sm)
else:
    train_and_generate_text(net, input_steps=260000, recovery_steps=10000, free_steps=5000, sm=sm)