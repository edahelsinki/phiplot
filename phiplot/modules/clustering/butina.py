import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from scipy.spatial.distance import pdist, squareform
from rdkit.ML.Cluster import Butina

class ButinaWrapper(BaseEstimator, ClusterMixin):
    """
    Scikit-learn wrapper for RDKit's Butina clustering algorithm.
    """
    def __init__(self, threshold=0.5, metric="euclidean"):
        self.threshold = threshold
        self.metric = metric

    def fit(self, X, y=None):
        if Butina is None:
            raise ImportError("RDKit is required to use Butina clustering.")
        
        n_samples = X.shape[0]
        
        # Calculate pairwise distances
        dist_array = pdist(X, metric=self.metric)
        sq_dists = squareform(dist_array)
        
        # RDKit expects a lower-triangular distance list
        tril_indices = np.tril_indices(n_samples, k=-1)
        rdkit_dists = sq_dists[tril_indices].tolist()
        
        # Returns a tuple of tuples: each sub-tuple is a cluster
        # The first element of each sub-tuple is the cluster medoid
        clusters = Butina.ClusterData(
            rdkit_dists,
            n_samples,
            self.threshold,
            isDistData=True,
            reordering=True 
        )
        
        self.labels_ = np.full(n_samples, -1, dtype=int)
        self.cluster_centers_ = []
        
        for cluster_idx, cluster in enumerate(clusters):
            centroid_idx = cluster[0]
            self.cluster_centers_.append(X[centroid_idx])
            for pt_idx in cluster:
                self.labels_[pt_idx] = cluster_idx
                
        self.cluster_centers_ = np.array(self.cluster_centers_)
        return self