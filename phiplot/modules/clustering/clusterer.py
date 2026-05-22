from sklearnex import patch_sklearn
patch_sklearn()
import time
import logging
import random
import numpy as np
import sklearn
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    Birch,
    BisectingKMeans
)
import bitbirch.bitbirch as bb
from sklearn.metrics import (
    silhouette_score,
    pairwise_distances,
    silhouette_samples
)
from sklearn.preprocessing import MaxAbsScaler
from kmodes.kmodes import KModes
from scipy.sparse import issparse
from scipy.ndimage import center_of_mass
from phiplot.modules.utils import *
from .param_parser import ParamParser
from .butina import ButinaWrapper

logger = logging.getLogger(__name__)

class Clusterer(ParamParser):
    def __init__(self, seed: int = 42) -> None:
        super().__init__("clustering_hyperparams.json")

        self.seed = seed
        self._set_global_seed()

        self.algorithm_map = {
            "KMeans": lambda hyperparams: KMeans(**self._inject_seed(hyperparams)),
            "KModes": lambda hyperparams: KModes(**self._inject_seed(hyperparams)),
            "BisectingKMeans": lambda hyperparams: BisectingKMeans(**self._inject_seed(hyperparams)),
            "Agglomerative": lambda hyperparams: AgglomerativeClustering(**hyperparams),
            "Birch": lambda hyperparams: Birch(**hyperparams),
            "BitBIRCH": lambda hyperparams: self._init_bitbirch(hyperparams),
            "Butina": lambda hyperparams: ButinaWrapper(**hyperparams)
        }

        self._X: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._medoid_idx: np.ndarray | None = None

    @property
    def X(self):
        if self._X is not None:
            return self._X.copy()
        
    @property
    def N(self):
        if self._X is not None:
            return self._X.shape[0]

    @property
    def labels(self):
        if self._labels is not None:
            return self._labels.copy()
    
    @property
    def centroids(self):
        if self._centroids is not None:
            return self._centroids.copy()
        
    @property
    def medoid_idx(self):
        if self._medoid_idx is not None:
            return self._medoid_idx.copy()

    def fit(self, X, medoid_metric: str = "euclidean", **kwargs) -> list:
        t_start = time.perf_counter()

        if self._algorithm == "Agglomerative":
            if "n_clusters" in kwargs:
                if kwargs["n_clusters"] == 0:
                    kwargs["n_clusters"] = None
        
        base: dict = self._hyperparams[self._algorithm]
        hyperparams: dict = base | {k: v for k, v in kwargs.items() if k in base}

        if issparse(X):
            is_binary = np.all(X.data == 1)
        else:
            is_binary = np.array_equal(np.unique(X), [0, 1])

        t_scale_start = time.perf_counter()
        if not is_binary:
            scaler = MaxAbsScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        logger.info(f"Diagnostics [fit]: Scaling took {time.perf_counter() - t_scale_start:.4f}s")

        t_fit_start = time.perf_counter()
        model: sklearn.base.ClusterMixin = self.algorithm_map[self._algorithm](hyperparams)
        X_scaled = X_scaled.astype(np.float32)
        model.fit(X_scaled)
        logger.info(f"Diagnostics [fit]: Core model fitting ({self._algorithm}) took {time.perf_counter() - t_fit_start:.4f}s")

        self._X = X_scaled
        if self._algorithm == "BitBIRCH":
            self._labels = model.get_assignments(X_scaled.shape[0])
        else:
            self._labels = model.labels_

        unique_labels = np.unique(self._labels)
        unique_labels = unique_labels[unique_labels != -1]

        t_centroid_start = time.perf_counter()
        if hasattr(model, "cluster_centers_"):
            self._centroids = model.cluster_centers_
        else:
            cluster_masks = (self._labels[:, None] == unique_labels)
            counts = cluster_masks.sum(axis=0)
            cluster_sums = cluster_masks.astype(X_scaled.dtype).T @ X_scaled
            self._centroids = cluster_sums / counts[:, None]
        logger.info(f"Diagnostics [fit]: Centroid calculation took {time.perf_counter() - t_centroid_start:.4f}s")
        
        t_medoid_start = time.perf_counter()
        medoid_indices = []
        for i, label_val in enumerate(unique_labels):
            mask = (self._labels == label_val)
            cluster_points = X_scaled[mask]
            real_indices = np.where(mask)[0]
            dists = pairwise_distances(self._centroids[i:i+1], cluster_points, metric=medoid_metric, n_jobs=-1)
            closest_in_cluster_idx = real_indices[np.argmin(dists)]
            medoid_indices.append(closest_in_cluster_idx)

        self._medoid_idx = np.array(medoid_indices)
        logger.info(f"Diagnostics [fit]: Medoid calculation took {time.perf_counter() - t_medoid_start:.4f}s")
        logger.info(f"Diagnostics [fit]: Total fit method took {time.perf_counter() - t_start:.4f}s")
    
    def eval_global_metrics(self, metric: str = "euclidean") -> dict[str, float] | None:
        t_start = time.perf_counter()
        
        X = self._X
        labels = self._labels
        unique_labels = np.unique(labels[labels != -1])
        K = len(unique_labels)
        N = len(X)
        
        if K < 2:
            return

        centroids = self._centroids
        global_centroid = X.mean(axis=0).reshape(1, -1)
        
        sep_dists = pairwise_distances(centroids, global_centroid, metric=metric, n_jobs=-1).flatten()
        separation = 0
        cohesion = 0
        
        cluster_radii = []
        
        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            n_i = len(cluster_points)
        
            separation += n_i * (sep_dists[i] ** 2)
            
            d_to_c = pairwise_distances(cluster_points, centroids[i:i+1], metric=metric, n_jobs=-1).flatten()
            cohesion += np.sum(d_to_c ** 2)
            
            cluster_radii.append(np.mean(d_to_c))

        ch_index = ((N - K) / (K - 1)) * (separation / cohesion) if cohesion != 0 else 0

        centroid_dists = pairwise_distances(centroids, metric=metric)
        db_ratios = []
        for i in range(K):
            max_ratio = 0
            for j in range(K):
                if i != j:
                    ratio = (cluster_radii[i] + cluster_radii[j]) / centroid_dists[i, j]
                    max_ratio = max(max_ratio, ratio)
            db_ratios.append(max_ratio)
        
        db_index = np.mean(db_ratios)
        logger.info(f"Diagnostics [global_metrics]: Distance matrix math and CH/DB Index took {time.perf_counter() - t_start:.4f}s")

        t_sil_start = time.perf_counter()
        MAX_SAMPLES = 5000
        if N > MAX_SAMPLES:
            sil_index = silhouette_score(X, labels, metric=metric, sample_size=MAX_SAMPLES, random_state=self.seed)
        else:
            sil_index = silhouette_score(X, labels, metric=metric)
        logger.info(f"Diagnostics [global_metrics]: Global Silhouette Score took {time.perf_counter() - t_sil_start:.4f}s")

        return {
            "Calinski-Harabasz Index": ch_index,
            "Davies-Bouldin Index": db_index,
            "Silhouette Index": sil_index
        }
        
    def eval_clusterwise_metrics(self, metric: str = "euclidean") -> dict[str | int, dict[str, float]]:
        t_start = time.perf_counter()
        stats = {}

        MAX_SAMPLES = 5000
        if self.N > MAX_SAMPLES:
            np.random.seed(self.seed)
            sample_indices = np.random.choice(self.N, MAX_SAMPLES, replace=False)
            X_sil = self._X[sample_indices]
            labels_sil = self._labels[sample_indices]
        else:
            X_sil = self._X
            labels_sil = self._labels

        t_sil_start = time.perf_counter()
        all_silhouette_scores = silhouette_samples(X_sil, labels_sil, metric=metric, n_jobs=-1)
        logger.info(f"Diagnostics [clusterwise_metrics]: Local Silhouette samples took {time.perf_counter() - t_sil_start:.4f}s")

        all_medoid_points = self._X[self._medoid_idx]
        medoid_to_medoid_dists = pairwise_distances(all_medoid_points, metric=metric, n_jobs=-1)

        unique_labels = np.unique(self._labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        t_loop_start = time.perf_counter()
        for i, label in enumerate(unique_labels):
            mask = self._labels == label
            cluster_points = self._X[mask]

            sil_mask = labels_sil == label
            if np.any(sil_mask):
                cluster_sil_score = float(np.mean(all_silhouette_scores[sil_mask]))
            else:
                cluster_sil_score = 0.0
            
            medoid_idx = self._medoid_idx[i]
            medoid_point = self._X[medoid_idx]
            dist_to_medoid = np.linalg.norm(cluster_points - medoid_point, axis=1)
            
            if len(cluster_points) > 1:
                MAX_DIAMETER_SAMPLES = 2000
                if len(cluster_points) > MAX_DIAMETER_SAMPLES:
                    np.random.seed(self.seed)
                    idx = np.random.choice(len(cluster_points), MAX_DIAMETER_SAMPLES, replace=False)
                    diam_points = cluster_points[idx]
                else:
                    diam_points = cluster_points
                    
                dists_internal = pairwise_distances(diam_points, metric=metric, n_jobs=-1)
                diameter = np.max(dists_internal)
            else:
                diameter = 0.0

            current_medoid_dists = medoid_to_medoid_dists[i] 
            
            non_zero_dists = current_medoid_dists[current_medoid_dists > 0]
            if len(non_zero_dists) > 0:
                nearest_medoid_dist = np.min(non_zero_dists)
                medoid_conf = np.mean(dist_to_medoid) / nearest_medoid_dist
            else:
                medoid_conf = np.nan

            stats[label] = {
                "Label": label,
                "count": int(len(cluster_points)),
                "silhouette_coeff": cluster_sil_score,
                "diameter": float(diameter),
                "radius": float(np.mean(dist_to_medoid)),
                "intraclust_variance": float(np.var(cluster_points)),
                "medoid_sep_ratio": float(medoid_conf),
                "max_radius": float(np.max(dist_to_medoid))
            }
        
        logger.info(f"Diagnostics [clusterwise_metrics]: Per-cluster stats loop took {time.perf_counter() - t_loop_start:.4f}s")
        logger.info(f"Diagnostics [clusterwise_metrics]: Total method took {time.perf_counter() - t_start:.4f}s")
            
        return stats
    
    def _init_bitbirch(self, params):
        criterion = params.pop("merge_criterion", "radius")
        tolerance = params.pop("tolerance", 0.05)
        bb.set_merge(criterion, tolerance=tolerance)
        return bb.BitBirch(**params)
    
    def _set_global_seed(self):
        """Sets the seed for Python and NumPy globals."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _inject_seed(self, params: dict) -> dict:
        """Helper to ensure random_state is in the hyperparams."""
        params.setdefault("random_state", self.seed)
        return params