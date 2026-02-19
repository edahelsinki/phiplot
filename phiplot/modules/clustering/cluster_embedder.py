import logging
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

    def __init__(self):
        super().__init__("static_embedding_hyperparams.json")
        
        self._algorithm_map = {
            "PCA": lambda hyperparams: PCA(**hyperparams),
            "KPCA": lambda hyperparams: KernelPCA(**hyperparams),
            "ICA": lambda hyperparams: FastICA(**hyperparams),
            "LLE": lambda hyperparams: LocallyLinearEmbedding(**hyperparams),
            "MDS": lambda hyperparams: MDS(**hyperparams),
            "tSNE": lambda hyperparams: TSNE(**hyperparams)
        }

    def embed(self, X, **kwargs) -> np.ndarray:
        base: dict = self._hyperparams[self._algorithm] | {"n_components": 2}
        hyperparams: dict = base | {k: v for k, v in kwargs.items() if k in base}
        model: sklearn.base.TransformerMixin = self._algorithm_map[self._algorithm](hyperparams)
        coords: np.ndarray = model.fit_transform(X)
        return coords