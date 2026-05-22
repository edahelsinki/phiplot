import logging
import random
import numpy as np
import sklearn
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding, MDS, TSNE
from phiplot.modules.utils import *
from .param_parser import ParamParser

logger = logging.getLogger(__name__)

class ClusterEmbedder(ParamParser):
    """
    Docstring for ClusterEmbedder
    """

    def __init__(self, seed: int = 42):
        super().__init__("static_embedding_hyperparams.json")

        self.seed = seed
        self._set_global_seed()
        
        self._algorithm_map = {
            "PCA": lambda hp: PCA(**self._inject_seed(hp)),
            "KPCA": lambda hp: KernelPCA(**self._inject_seed(hp)),
            "ICA": lambda hp: FastICA(**self._inject_seed(hp)),
            "LLE": lambda hp: LocallyLinearEmbedding(**self._inject_seed(hp)),
            "MDS": lambda hp: MDS(**self._inject_seed(hp)),
            "tSNE": lambda hp: TSNE(**self._inject_seed(hp))
        }

        self._X: np.ndarray | None = None
        self._centroids: dict[str | int, np.ndarray] | None = None

    @property
    def X(self):
        if self._X is not None:
            return self._X.copy()
        
    @property
    def centroids(self):
        if self._centroids is not None:
            return self._centroids.copy()
              
    def embed(self, X, centroids, **kwargs) -> np.ndarray:
        base: dict = self._hyperparams[self._algorithm] | {"n_components": 2}
        hyperparams: dict = base | {k: v for k, v in kwargs.items() if k in base}
        model: sklearn.base.TransformerMixin = self._algorithm_map[self._algorithm](hyperparams)

        if self._can_transform(model):
            X_2d: np.ndarray = model.fit_transform(X)
            centroids_2d = model.transform(centroids)
        else:
            X_combined = np.vstack([X, centroids])
            combined_2d = model.fit_transform(X_combined)
            
            num_centroids = len(centroids) 
            X_2d = combined_2d[:-num_centroids]
            centroids_2d = combined_2d[-num_centroids:]

        self._X = X_2d
        self._centroids = centroids_2d
    
    def _can_transform(self, model):
        return callable(getattr(model, "transform", None))
    
    def _set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _inject_seed(self, params: dict) -> dict:
        """
        Injects random_state for stochastic algorithms. 
        Note: PCA is deterministic unless using 'arpack' or 'randomized' solvers, 
        but sklearn's PCA still accepts random_state safely.
        """
        params.setdefault("random_state", self.seed)
        return params