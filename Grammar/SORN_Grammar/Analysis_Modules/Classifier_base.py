from PymoNNto import *

class Classifier_base(AnalysisModule):

    def initialize(self, neurons):
        self.add_tag('classifier')
        self.add_tag('cluster_matrix_classifier')
        self.corrMatrices={}

    def get_cluster_matrix(self, key):
        if key in self.corrMatrices:
            classification = self.get_results()[key]#self.last_call_result()
            idx = np.argsort(classification)
            return self.corrMatrices[key].iloc[idx, :].T.iloc[idx, :], idx
        else:
            print('module has to be executed first.')

    def get_data_matrix(self, neurons):
        return #overrride

    def execute(self, neurons, sensitivity=2):
        print('computing cluster classes...')
        import scipy.cluster.hierarchy as sch
        import pandas as pd

        data = self.get_data_matrix(neurons)

        mask = np.sum(data, axis=1) > 0
        self.update_progress(10)

        df = pd.DataFrame(data[mask])  # .T
        self.corrMatrix = df.corr()
        self.update_progress(40)

        self.corrMatrices[self.current_key] = self.corrMatrix

        pairwise_distances = sch.distance.pdist(self.corrMatrix)
        self.update_progress(60)
        linkage = sch.linkage(pairwise_distances, method='complete')
        self.update_progress(70)
        cluster_distance_threshold = pairwise_distances.max() / sensitivity
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
        self.update_progress(90)

        result = np.zeros(data.shape[0]) - 1
        result[mask] = idx_to_cluster_array

        return result
