from PymoNNto import *

class MyModule(AnalysisModule):

    def execute(self, a, b, c=5):
        return a+b+c



neurons=NeuronGroup()

neurons.add_analysis_module(MyModule(tag='category'))

