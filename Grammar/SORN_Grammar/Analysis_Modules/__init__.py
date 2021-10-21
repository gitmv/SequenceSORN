from Grammar.SORN_Grammar.Analysis_Modules.Classifier_Weights_Pre import *
from Grammar.SORN_Grammar.Analysis_Modules.Classifier_Weights_Post import *
from Grammar.SORN_Grammar.Analysis_Modules.Classifier_Backpropagation import *
from Grammar.SORN_Grammar.Analysis_Modules.Classifier_Activity_Response import *
from Grammar.SORN_Grammar.Analysis_Modules.Labeler_Activity_Response import *
from Grammar.SORN_Grammar.Analysis_Modules.Labeler_Backpropagation import *


def add_all(ng):
    Classifier_Weights_Pre(ng)
    Classifier_Weights_Post(ng)
    Classifier_Backpropagation(ng)
    #Classifier_Activity_Response(ng)
    Labeler_Activity_Response(ng)
    Labeler_Backpropagation(ng)
