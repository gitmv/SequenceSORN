from PymoNNto import *
from PymoNNto.Exploration.Evolution.Devices.Evolution_Device import *

genome = {}
genome['evo_name'] = 'plot_generation'
genome['gen'] = 1

for id, value in enumerate(np.arange(1,3,0.1)):

    genome['gene_name'] = value
    genome['id'] = id

    execute_local_file('', genome)

