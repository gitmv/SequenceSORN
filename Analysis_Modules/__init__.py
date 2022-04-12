from Analysis_Modules.Classifier_Weights_Pre import *
from Analysis_Modules.Classifier_Weights_Post import *
from Analysis_Modules.Classifier_Backpropagation import *
from Analysis_Modules.Classifier_Activity_Response import *
from Analysis_Modules.Labeler_Activity_Response import *
from Analysis_Modules.Labeler_Backpropagation import *
from Analysis_Modules.Static_Classification import *

def add_all_analysis_modules(ng):
    Classifier_Weights_Pre(ng)
    Classifier_Weights_Post(ng)
    Classifier_Backpropagation(ng)
    #Classifier_Activity_Response(ng)
    Labeler_Activity_Response(ng)
    Labeler_Backpropagation(ng)
