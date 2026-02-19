import logging
import param
from typing import Any
import numpy as np
from scipy.spatial.distance import pdist
from .embedder import PCA, LLE, Isomap, ICA, tSNE, MDS, cKPCA, LSP
from .embedding_metrics import EmbeddingMetrics
from phiplot.modules.ui.utils import log_process, ProcessResult
from phiplot.modules.utils.default_param_parser import *

logger = logging.getLogger(__name__)


class EmbeddingHandler(param.Parameterized):
    """
    Manages initialization and updating of the embedding, including interactive updates.

    Acts as a middle-layer between embedders and the user interface, providing
    interfaces for retrieving point coordinates, adding embedding constraints,
    and projecting additional points. Works with positional indexing for points.

    Args:
        cp_init : dict[int, tuple[float, float]], optional
            Initial control points. Default is None.
        ml_init : list[set[int, int]], optional
            Initial must-link constraints. Default is None.
        cl_init : list[set[int, int]], optional
            Initial cannot-link constraints. Default is None.
        **params : dict
            Additional parameters for the `param.Parameterized` superclass.
    """

    must_reembed = param.Boolean(False)
    control_points = param.Dict({})
    must_links = param.List([])
    cannot_links = param.List([])

    def __init__(self, cp_init=None, ml_init=None, cl_init=None, **params):
        super().__init__(**params)

        self.control_points = cp_init or {}
        self.must_links = ml_init or []
        self.cannot_links = cl_init or []

        self._X = np.array([])
        self._embedding = dict(x=[], y=[])

        self._algorithm = None
        self._model = None
        self._embedding_params = None
        self._kernel_params = None
        self._link_strength_multiplier = 1

        self.model_initialized = False
        self.must_reembed = False
        self.update_existing = False

        self._static_emb_param_parser = DefaultParamParser("static_embedding_hyperparams.json")
        self._interactive_emb_param_parser = DefaultParamParser("interactive_embedding_hyperparams.json")

    @property
    def cp_indices(self) -> list:
        return list(self.control_points.keys())

    @property
    def supported_static_embeddings(self):
        return self._static_emb_param_parser.supported
    
    @property
    def supported_interactive_embeddings(self):
        return self._interactive_emb_param_parser.supported

    @property
    def supported_embeddings(self):
        return self.supported_interactive_embeddings + self.supported_static_embeddings

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algo) -> None:
        if algo in self.supported_algorithms:
            self._algorithm = algo
        else:
            self._algorithm = "PCA"
            log_process(
                ProcessResult(
                    False, "Unsupported algorithm. Using 'PCA' instead.", True, "error"
                )
            )

    @property
    def embedding(self) -> dict[str, list[float]]:
        return self._embedding

    @property
    def link_strengh_multiplier(self) -> float:
        return self._link_strength_multiplier

    @link_strengh_multiplier.setter
    def link_strengh_multiplier(self, multiplier: float) -> None:
        if isinstance(multiplier, (int, float)):
            if multiplier >= 0 and multiplier <= 100:
                self._link_strength_multiplier = multiplier
                if self.model_initialized and self._algorithm == "cKPCA":
                    self._model.adjust_link_strength(self._link_strength_multiplier)
                    self._update_embedding()
                res = ProcessResult(
                    True, "Link strength adjusted succesfully!", False, "debug"
                )
            else:
                res = ProcessResult(
                    False, "The multiplier must be in the range[0, 100]", False, "error"
                )
        else:
            res = ProcessResult(
                False, "The multiplier must be a number.", False, "error"
            )
        log_process(res)

    @property
    def X(self) -> np.ndarray:
        return self._X.copy()

    @X.setter
    def X(self, new_X) -> None:
        try:
            if isinstance(new_X, np.ndarray):
                self._X = new_X
            else:
                self._X = np.array(new_X)
        except Exception as e:
            logger.exception("Error during setting the feature matrix:")
            res = ProcessResult(
                False, "Could not set the feature matrix.", False, "error"
            )
        else:
            res = ProcessResult(True, "Feature matrix set successfully", False, "debug")
        log_process(res)

    def median_pairwise_dist(self, metric: str = "euclidean", n_samples: int = 10000) -> float | None:
        """
        Compute the median of all pairwise distances between a random 
        sample of rows in the feature matrix X.

        Args:
            metric (str): The distance measure to use. Should be one
                that is supported by `scipy.spatial.distance.pdist`.
            n_samples (int): The number of samples to take. Defaults to 10000 or
                the number of datapoints if there are fewer than 10000 of them.

        Returns:
            float: The median of the pairwise Euclidean distances between samples,
                or None if the feature matrix has not yet been set or is empty.
        """

        if self._X is None:
            return

        n = self._X.shape[0]
        if n == 0:
            return
        
        if n <= n_samples:
            sample = self._X
        else:
            sample = self._X[np.random.choice(n, n_samples, replace=False)]

        pairwise_dists = pdist(sample, metric=metric)

        return np.median(pairwise_dists)

    def get_params(self) -> dict[str, dict[str, str | int | float]]:
        """
        Get combined embedding and kernel hyperparamaters.

        Returns:
            (dict[str, dict[str, str | int | float]]): Constructed as
                - "embedding" (dict[str, str | int | float]): Embedding-specific hyperparameters.
                - "kernel" (dict[str, str | int | float]): Kernel-specific hyperparameters.
        """

        return dict(embedding=self._embedding_params, kernel=self._kernel_params)

    def get_embedding_metrics(self, distance_measure: str) -> dict[str, float]:
        """
        Get all the embedding quality metrics.

        Args:
            distance_measure: Any valid sklearn distance mesure for the metric computations.

        Returns:
            (dict[str, float]): The name of the metric as the key and the computed metric as the value.
        """

        embedding_metrics = EmbeddingMetrics(
            self._X, self._model.get_embedding().T, distance_measure
        )
        return embedding_metrics.get_metrics()

    def status(self) -> dict[str, Any]:
        """
        Get the current status of the embedding.

        Returns:
            (dict[str, Any]): Constructed as:
                - "intialized" (bool): True if the embedding has succesfully been initialized.
                - "n_points" (int): The number of points currently in the embedding.
                - "has_control_points" (bool): True if there is at least one control point.
                - "has_link_constraints" (bool): True if there is at least one must-link or cannot-link constraint.
                - "must_reembed" (bool): True if the next update should be a re-embedding of the current points.
        """

        return {
            "initialized": self.model_initialized,
            "n_points": len(self._X),
            "has_control_points": len(self.control_points) > 0,
            "has_link_constraints": self.must_links or self.cannot_links,
            "must_reembed": self.must_reembed,
        }

    def init_embedding(
        self,
        algorithm: str,
        emb_params: dict[str, (int | float | str)] | None = None
    ) -> ProcessResult:
        """
        Initialize the embedding model.

        Args:
            algorithm (str): The name of the projection algorithm to use.
            params (dict): The hyperparameters for the model.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self.model_initialized = False
        emb_params = emb_params or {}

        if algorithm not in self.supported_embeddings:
            res = ProcessResult(
                False,
                f"Unsupported algorithm {algorithm}. Must be one of {self.supported_embeddings}.",
                True,
                "error",
            )
            logger.error(f"Unsupported algorithm {algorithm}. Must be one of {self.supported_embeddings}.")
        elif len(self._X) == 0:
            res = ProcessResult(False, "The feature matrix is missing.", True, "error")
            logger.error(f"The feature matrix is missing.")
        else:
            model_map = {
                "PCA": lambda: PCA(data=self._X),
                "LLE": lambda: LLE(
                    data=self._X,
                    **emb_params,
                ),
                "Isomap": lambda: Isomap(
                    data=self._X,
                    **emb_params,
                ),
                "ICA": lambda: ICA(
                    data=self._X,
                    **emb_params
                ),
                "tSNE": lambda: tSNE(
                    data=self._X,
                    **emb_params
                ),
                "MDS": lambda: MDS(
                    data=self._X,
                    **emb_params
                ),
                "cKPCA": lambda: cKPCA(
                    data=self._X,
                    initial_control_points=self.control_points,
                    kernel_name=emb_params["kernel"].lower(),
                    **emb_params
                ),
                "LSP": lambda: LSP(
                    data=self._X,
                    initial_control_points=self.control_points,
                    **emb_params,
                ),
            }

            self._algorithm = algorithm
            self._model = model_map[algorithm]()

            if self.control_points and self._model.is_dynamic:
                self._model.update_control_points(self.control_points)

            if (self.must_links or self.cannot_links) and self._model.is_dynamic:
                self._model.update_must_and_cannot_link(
                    self.must_links, self.cannot_links
                )

            self._update_embedding()

            self.model_initialized = True
            self.must_reembed = False

            res = ProcessResult(True, "Embedding intialized succesfully!", True, "info")
            log_process(res)
        return res

    def project_new_points(self, new_X: list[np.ndarray]) -> ProcessResult:
        """
        Project new points to the embedding after the initial embedding.

        Args:
            new_X (list[np.ndarray]): The feature vectors of the points to project.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        plural = "s" if len(new_X) > 1 else ""
        try:
            for i, row in enumerate(new_X):
                x, y = self._model.add_new_point(row[0])
                self._X = np.vstack([self._X, row])
            self._update_embedding()
            res = ProcessResult(
                True, f"New point{plural} projected succesfully!", True, "info"
            )
        except Exception as e:
            logger.exception(f"Error in projecting a new point: {e}")
            res = ProcessResult(
                True, f"Could not project the point{plural}.", True, "error"
            )
        log_process(res)
        return res

    def add_control_point(self, idx: int, x: float, y: float) -> ProcessResult:
        """
        Make an existing point into a control point.

        Args:
            idx (int): The index of the point.
            x (float): The desired x-coordinate.
            y (float): The desired y-coordinate.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        if idx in self.control_points.keys():
            self.update_existing = True
            action = "update"
        else:
            self.update_existing = False
            action = "add"

        try:
            new = self.control_points.copy()
            new[idx] = [x, y]
            self.control_points = new

            if self.model_initialized and self._model.is_dynamic:
                self._model.update_control_points(self.control_points)
                self._update_embedding()

            res = ProcessResult(
                True, f"Control point {action}ed succesfully!", False, "debug"
            )
        except Exception as e:
            logger.exception(f"Error when trying to {action} the control point:")
            res = ProcessResult(
                False, f"The control point could not be {action}ed.", True, "error"
            )
        log_process(res)
        return res

    def remove_control_points(self, deleted: list[int]) -> ProcessResult:
        """
        Remove control points from the embedding.

        Args:
            deleted (list[int]): The indices of the points to remove.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        if not self.update_existing:
            for idx in deleted:
                if idx in self.control_points:
                    del self.control_points[idx]
                else:
                    logger.debug(f"Could not remove missing control point {idx}.")
            if not self.control_points:
                self.must_reembed = True
            elif self.model_initialized and self._model.is_dynamic:
                self._model.update_control_points(self.control_points)
                self._update_embedding()

    def add_must_link(self, pair: tuple[int, int]) -> ProcessResult:
        """
        Add a must-link constraint between two points.

        Args:
            pair (tuple[int, int]): The indices of the two points.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self._add_link(pair, "must_links")

    def remove_must_links(self, deleted: list[tuple[int, int]]) -> ProcessResult:
        """
        Remove must-link constraints between points.

        Args:
            pair (list[tuple[int, int]]): The list of index pairs with a must-link
                constraint between the corresponding points.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self._remove_links(deleted, "must_links")

    def add_cannot_link(self, pair: tuple[int, int]) -> ProcessResult:
        """
        Add a cannot-link constraint between two points.

        Args:
            pair (tuple[int, int]): The indices of the two points.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self._add_link(pair, "cannot_links")

    def remove_cannot_links(self, deleted: list[tuple[int, int]]) -> ProcessResult:
        """
        Remove cannot-link constraints between points.

        Args:
            pair (list[tuple[int, int]]): The list of index pairs with a cannot-link
                constraint between the corresponding points.

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        self._remove_links(deleted, "cannot_links")

    def clear(self) -> None:
        """
        Clear all the existing constraints.
        """

        self.control_points = {}
        self.must_links = []
        self.cannot_links = []

    def remap_constraints(self, pos_map: dict[int, int]) -> None:
        """
        Remap the constraints based on a new indexing.

        Args:
            pos_map (dict[int, int]): The mapping between the new and old index
        """

        def remap_index(idx):
            return pos_map[idx] if idx in pos_map else None

        self.control_points = {
            remap_index(cp_idx): cp_val
            for cp_idx, cp_val in self.control_points.items()
            if cp_idx in pos_map
        }

        self.must_links = [
            (remap_index(a), remap_index(b))
            for (a, b) in self.must_links
            if a in pos_map and b in pos_map
        ]

        self.cannot_links = [
            (remap_index(a), remap_index(b))
            for (a, b) in self.cannot_links
            if a in pos_map and b in pos_map
        ]

        mask = [i in pos_map.values() for i in range(len(self._X))]
        self.X = self._X[mask]

    def _add_link(self, pair: tuple[int, int], attr_name: str) -> ProcessResult:
        """
        Add a link constraint of specified type between two points.

        Args:
            pair (tuple[int, int]): The indices of the two points.
            attr_name (str): Either "must_links" or "cannot_links"

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        try:
            links = getattr(self, attr_name)
            if pair not in links:
                new = links + [pair]
                setattr(self, attr_name, new)
            if self.model_initialized and self._model.is_dynamic:
                self._model.update_must_and_cannot_link(
                    self.must_links, self.cannot_links
                )
                self._update_embedding()
            res = ProcessResult(
                True,
                f"Added a link between {pair} succesfully!",
                False,
                "debug",
            )
        except Exception as e:
            logger.exception(f"Error during adding {attr_name}:")
            res = ProcessResult(
                False, f"Could not add a link between {pair}.", True, "error"
            )
        log_process(res)
        return res

    def _remove_links(
        self, deleted: list[tuple[int, int]], attr_name: str
    ) -> ProcessResult:
        """
        Remove link constraints of specified type between points.

        Args:
            pair (tuple[int, int]): The list of index pairs with a link
                constraint between the corresponding points.
            attr_name (str): Either "must_links" or "cannot_links"

        Returns:
            ProcessResult: See :class:`ProcessResult` for details.
        """

        try:
            links = getattr(self, attr_name)
            new = [pair for pair in links if pair not in deleted]
            setattr(self, attr_name, new)
            if self.model_initialized and self._model.is_dynamic:
                self._model.update_must_and_cannot_link(
                    self.must_links, self.cannot_links
                )
                self._update_embedding()
                res = ProcessResult(
                    True,
                    f"Removed {attr_name} between {deleted} succesfully!",
                    False,
                    "debug",
                )
        except Exception as e:
            logger.exception(f"Error during removingg {attr_name}s:")
            res = ProcessResult(
                False, f"Could not remove {attr_name} between {deleted}.", True, "error"
            )
        log_process(res)
        return res

    def _update_embedding(self) -> None:
        """
        Retrieve and update the new embedding coordinates of points.
        """

        x, y = self._model.get_embedding()
        self._embedding["x"] = x
        self._embedding["y"] = y
