import logging
import numpy as np
import sklearn
from sklearn.cluster import (
    KMeans,
    Birch,
    BisectingKMeans
)
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from phiplot.modules.utils import *
from .param_parser import ParamParser

logger = logging.getLogger(__name__)

class Clusterer(ParamParser):
    """
    Docstring for Clusterer
    """

    def __init__(self) -> None:
        super().__init__("clustering_hyperparams.json")

        self.algorithm_map = {
            "KMeans": lambda hyperparams: KMeans(**hyperparams),
            "Birch": lambda hyperparams: Birch(**hyperparams),
            "BisectingKMeans": lambda hyperparams: BisectingKMeans(**hyperparams)
        }

    def cluster(self, X, **kwargs) -> list:
        base: dict = self._hyperparams[self._algorithm]
        hyperparams: dict = base | {k: v for k, v in kwargs.items() if k in base}
        model: sklearn.base.ClusterMixin = self.algorithm_map[self._algorithm](hyperparams)
        model.fit(X)
        return model.labels_
    
    def eval_metrics(self, X, labels) -> dict[str, float]:
        try:
            return {
                "Silhouette Score": silhouette_score(X, labels),
                "Calinski-Harabasz Score": calinski_harabasz_score(X, labels),
                "Davis-Bouldin Score": davies_bouldin_score(X, labels)
            }
        except ValueError:
            logger.warning("Invalid input for clustering metrics.")
            return {
                "Silhouette Score": np.nan,
                "Calinski-Harabasz Score": np.nan,
                "Davis-Bouldin Score": np.nan
            }
        except Exception:
            logger.error("An unexpected error occurred while computing the clustering metrics.")
            return {}