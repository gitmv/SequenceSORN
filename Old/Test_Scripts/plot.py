import sys

py_file = open('SORN_WTA.py', "r")
execution_string = py_file.read()

def get_gene_id(gene):
    id = ''
    for key, value in gene.items():
        id += '#'+key+'@'+str(value)
    return id+'#'

def execute_local_file( genome):
    for arg in sys.argv:  # remove old
        if 'genome=' in arg:
            sys.argv.remove(arg)
    sys.argv.append('genome=' + get_gene_id(genome))  # add new

    exec(execution_string)
    py_file.close()


for run in range(3):
    for i, density in enumerate([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        execute_local_file({'evo_name': 'synapse_d_WTA_rf', 'gen': run, 'id': i, 'sd': density})
