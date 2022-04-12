import scipy
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt

def cluster_corr(corr_array, inplace=False):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    return corr_array[idx, :][:, idx]


m=np.random.rand(100, 100)>0.9

plt.imshow(m)
plt.show()
plt.imshow(cluster_corr(m))
plt.show()