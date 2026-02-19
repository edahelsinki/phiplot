import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class EmbeddingMetrics:
    """
    Provides embedding quality metrics computed by comparing the data in the original
    high dimensional space to its embedding in a lower dimensional space. Assumes
    the original and embedded data are aligned rowwise.

    The metrics are:
        - Trustworthiness (local structure preservation).
        - KNN Preservation (local structure preservation).
        - Sheppard Correlation (global structure preservation).
        - Stress (global structure preservation).

    Args:
        original (np.ndarray): The original data.
        embedding (np.ndarray): The embedded data.
        distance_measure (str): The name of the distance measure to use for computing the
            metrics. Should be one that is supported by all of the above metrics.
        n_neigbors (int): The number of neighbors to consider for the local preservation metrics.
            Should be fewer than half the number of data points.
    """

    def __init__(
        self,
        original: np.ndarray,
        embedding: np.ndarray,
        distance_measure: str = "euclidean",
        n_neighbors: int = 10,
    ):
        self._original = original
        self._embedding = embedding
        self._distance_measure = distance_measure
        self._n, self._m = original.shape
        self._n_neighbors = min(self._n // 2, n_neighbors)

    def get_metrics(self) -> dict[str, float]:
        """
        Get all the computed metrics.

        Returns:
            dict[str, float]: The name of the metric as the key and
                computed metric as the value.
        """

        return {
            "Trustworthiness": self._trustworthiness(self._n_neighbors),
            "KNN Preservation": self._knn_preservation(self._n_neighbors),
            "Shepard Correlation": self._shepard_correlation(),
            "Stress": self._stress(),
        }

    def _trustworthiness(self, n_neighbours=10) -> float:
        return trustworthiness(
            self._original, self._embedding, n_neighbors=n_neighbours
        )

    def _knn_preservation(self, k=10) -> float:
        nbrs_high = NearestNeighbors(
            n_neighbors=k + 1, metric=self._distance_measure
        ).fit(self._original)
        nbrs_low = NearestNeighbors(
            n_neighbors=k + 1, metric=self._distance_measure
        ).fit(self._embedding)

        _, indices_high = nbrs_high.kneighbors(self._original)
        _, indices_low = nbrs_low.kneighbors(self._embedding)

        indices_high = indices_high[:, 1:]
        indices_low = indices_low[:, 1:]

        overlap = np.array(
            [
                len(set(high).intersection(set(low))) / k
                for high, low in zip(indices_high, indices_low)
            ]
        )

        return overlap.mean()

    def _shepard_correlation(self) -> float:
        dist_high = pdist(self._original, metric=self._distance_measure)
        dist_low = pdist(self._embedding, metric=self._distance_measure)
        corr, _ = spearmanr(dist_high, dist_low)
        return corr

    def _stress(self) -> float:
        D_high = squareform(pdist(self._original, metric=self._distance_measure))
        D_low = squareform(pdist(self._embedding, metric=self._distance_measure))
        stress = np.sqrt(np.sum((D_high - D_low) ** 2) / np.sum(D_high**2))
        return stress
