from abc import ABC, abstractmethod
import concurrent.futures
import copy
from functools import partial
import logging
import os
import psutil
import threading
import time
from typing import Tuple, Optional
import warnings
import jax
from jax import numpy as jnp, jit, lax, random, Array, config
import numpy as np
from sklearn import decomposition
from sklearn.metrics import pairwise_distances
import sklearn.manifold as manifold
from .kernels import Kernel
from .nystroem import Nystroem, KernelKMeansSQ

config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def timing_logger(method):
    def wrapper(self, *args, **kwargs):
        logger.info("Intializing the embedder...")
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        elapsed_ms = int(1000 * (time.perf_counter() - start))
        logger.info(f"{self.name} embedding initialized in: {elapsed_ms:.2f} ms")
        return result

    return wrapper


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models that provides a standardized interface for
    retrieving the 2D embedding and includes a method for normalizing the input data.

    Args:
        data (Array): Original high-dimensional data points to embed, shape (n_samples, n_features).
    """

    def __init__(self, data: Array) -> None:
        self.name = ""
        self.is_dynamic = False
        self.verbose = False

        self.normalize = True
        self.data = data
        if self.normalize:
            self.data = self.normalize_data(data)

        self.n, self.d = data.shape

        self.projection_matrix = None
        self.embedding = None

    @abstractmethod
    def get_embedding(self) -> Array:
        """Return the 2D embedding for the model."""
        pass

    def normalize_data(self, X: Array) -> jax.Array:
        """
        Normalize each feature to zero mean and unit variance.

        Args:
            X (Array): Input array of shape (n_samples, n_features).

        Returns:
            Array: Normalized array with same shape as `X`.
        """

        self.mean = jnp.mean(X, axis=0)
        self.std = jnp.clip(jnp.std(X, axis=0), 1e-12)
        return (X - self.mean) / self.std

    def normalize_new_point(self, x_new: Array) -> Array:
        """
        Normalize a single additional point.

        Args:
            x_new (Array): The new point (1, n_features).

        Returns:
            Array: Normalized array with same shape as `x_new`.
        """

        return (x_new - self.mean) / self.std


class StaticEmbedding(BaseEmbedding):
    """
    Static embedding model that assumes the embedding is fixed after initialization.

    The `get_embedding` method returns the stored embedding transposed.

    Args:
        data (Array): Input data to embed, shape (n_samples, n_features).
    """

    def __init__(self, data: Array) -> None:
        super().__init__(data)

    def get_embedding(self) -> Array:
        return self.embedding.T


class InteractiveEmbedding(BaseEmbedding):
    """
    Interactive embedding model that supports adding constraints to the embedding,
    including control points, must-link, and cannot-link constraints.

    Args:
        data (Array): Input data to embed, shape (n_samples, n_features).
    """

    def __init__(self, data: Array) -> None:
        super().__init__(data)
        self.is_dynamic = True

        self.control_points = {}
        self.X = jnp.array([])
        self.y = jnp.array([])

        self.must_link = set()
        self.cannot_link = set()
        self.has_ml_cl_constraints = False

        self._frame_rate = 30

    @property
    def control_point_indices(self):
        return list(self.control_points.keys())

    def _normalize_constraints(self, constraints: list) -> set[tuple[int, int]]:
        """
        Normalize a list of link constraints into a set of ordered index tuples.
        Each constraint must be a set containing exactly two integer indices.

        Args:
            constraints (list): List of constraints, each expected to be a set of two indices.

        Returns:
            set[tuple[int, int]]: A set of normalized constraints represented as sorted tuples of indices.

        Raises:
            TypeError: If any constraint is not a set or does not contain exactly two elements.
            ValueError: If any index in constraints is not an integer or out of valid range.
        """

        if not isinstance(constraints, list):
            raise TypeError(f"Constraints should be a list, got {type(constraints)}")

        normalized_constraints = set()
        max_index = len(self.data) - 1

        for c in constraints:
            if len(c) != 2:
                raise ValueError(
                    f"Each constraint must have exactly two indices, got {len(c)}"
                )
            idx1, idx2 = tuple(c)
            if not (
                isinstance(idx1, (int, jnp.integer))
                and isinstance(idx2, (int, jnp.integer))
            ):
                raise TypeError(
                    f"Constraint indices must be integers, got {type(idx1)} and {type(idx2)}"
                )
            if idx1 < 0 or idx1 > max_index or idx2 < 0 or idx2 > max_index:
                raise ValueError(
                    f"Constraint indices {idx1}, {idx2} out of valid range [0, {max_index}]"
                )
            normalized_constraints.add(tuple(sorted((idx1, idx2))))

        return normalized_constraints

    def _on_constraints_updated(self):
        """
        Hook method called after constraints are updated. Can be overridden by subclasses.
        """
        pass

    def update_must_and_cannot_link(self, must_link: list, cannot_link: list) -> None:
        """
        Normalize and update must-link and cannot-link constraints if they have changed,
        update the flag indicating whether any link constraints exist,
        and invoke a callback to handle the update.

        Args:
            must_link (list): List of must-link constraints, each as a tuple of two indices.
            cannot_link (list): List of cannot-link constraints, each as a tuple of two indices.
        """

        new_ml_norm = self._normalize_constraints(must_link)
        new_cl_norm = self._normalize_constraints(cannot_link)

        if new_ml_norm == self.must_link and new_cl_norm == self.cannot_link:
            logger.debug("Constraints unchanged. Skipping update.")
            return

        self.must_link = new_ml_norm
        self.cannot_link = new_cl_norm
        self.has_ml_cl_constraints = bool(self.must_link or self.cannot_link)

        self._on_constraints_updated()

    def update_control_points(self, points: dict[int, Array]) -> None:
        """
        Update control points with their feature indices and corresponding 2D embedding coordinates.

        Args:
            points (dict[int, Array]): Dictionary mapping control point indices in the original data
                to their 2D embedding coordinates.

        Raises:
            ValueError: If any index in `points` is out of bounds of the input data.
            TypeError: If `points` is not a dictionary or keys are not integers.

        Notes:
            - If `points` is empty or None, control points and related attributes are reset.
            - Assumes embedding coordinates are JAX arrays or convertible to such.
        """
        if points is None or len(points) == 0:
            self.control_points = {}
            self.X = jnp.array([])
            self.y = jnp.array([])
            return

        if not isinstance(points, dict):
            raise TypeError(f"Expected points to be a dict, got {type(points)}")

        max_index = len(self.data) - 1
        for idx in points.keys():
            if not isinstance(idx, (int, jnp.integer)):
                raise TypeError(f"Control point index must be int, got {type(idx)}")
            if idx < 0 or idx > max_index:
                raise ValueError(
                    f"Control point index {idx} is out of bounds [0, {max_index}]"
                )

        self.control_points = copy.deepcopy(points)
        self.X = self.data[jnp.array(list(points.keys()))]
        self.y = jnp.array(list(points.values()))

    def augment_control_points(self) -> None:
        """
        Update the embedding positions based on the must-link and cannot-link constraints
        by moving the pairs of points along their difference vector and augment the control
        points to include these adjusted points.
        """

        embedding = self.get_embedding().T
        avg_median = jnp.mean(jnp.abs(jnp.median(embedding, axis=0)))

        indices = jnp.array(list(self.control_points.keys()))
        control_point_mask = (
            jnp.zeros(embedding.shape[0], dtype=bool).at[indices].set(True)
        )

        tmp_points = {}

        # Cannot-link update
        if self.cannot_link:
            for i, j in self.cannot_link:
                if control_point_mask[i] and control_point_mask[j]:
                    continue

                x1 = embedding[i]
                x2 = embedding[j]
                diff = x1 - x2
                norm = jnp.linalg.norm(diff) + 1e-8
                direction = (diff / norm) * 5 * avg_median

                if not control_point_mask[i]:
                    embedding = embedding.at[i].set(x1 + direction)
                    tmp_points[i] = x1 + direction
                if not control_point_mask[j]:
                    embedding = embedding.at[j].set(x2 - direction)
                    tmp_points[j] = x2 - direction

        # Must-link update
        if self.must_link:
            for i, j in self.must_link:
                if control_point_mask[i] and control_point_mask[j]:
                    continue

                x1 = embedding[i]
                x2 = embedding[j]
                diff = x1 - x2
                adjustment = 0.45 * diff

                if not control_point_mask[i]:
                    embedding = embedding.at[i].set(x1 - adjustment)
                    tmp_points[i] = x1 - adjustment
                if not control_point_mask[j]:
                    embedding = embedding.at[j].set(x2 + adjustment)
                    tmp_points[j] = x2 + adjustment

        # Augment control points to include adjusted points
        for idx, val in tmp_points.items():
            if idx not in self.control_points:
                self.control_points[idx] = jnp.array(val)

        self.X = self.data[jnp.array(list(self.control_points.keys()))]
        self.y = jnp.array(list(self.control_points.values()))


class PCA(StaticEmbedding):
    """
    Provides a static 2D embedding using the first two principal
    components from Principal Component Analysis (PCA).
    """

    @timing_logger
    def __init__(self, data: Array, **kwargs) -> None:
        super(PCA, self).__init__(data)
        self.name = "PCA"

        pca = decomposition.PCA(n_components=2, **kwargs)
        pca.fit(data)
        self.projection_matrix = pca.components_
        self.embedding = jnp.array(pca.transform(data))

class KPCA(StaticEmbedding):
    """
    Provides a static 2D embedding using the first two principal
    components from kernel Principal Component Analysis (KPCA).
    """

    @timing_logger
    def __init__(self, data: Array, **kwargs) -> None:
        super(KPCA, self).__init__(data)
        self.name = "KPCA"

        kpca = decomposition.KernelPCA(n_components=2, **kwargs)
        kpca.fit(data)
        self.projection_matrix = kpca.components_
        self.embedding = jnp.array(kpca.transform(data))

class LLE(StaticEmbedding):
    """
    Provides a static 2D embedding using Locally Linear Embedding (LLE),
    a non-linear manifold learning method based on local neighborhood geometry.
    """

    @timing_logger
    def __init__(self, data: Array, **kwargs) -> None:
        super(LLE, self).__init__(data)
        self.name = "LLE"

        lle = manifold.LocallyLinearEmbedding(
            n_components=2, **kwargs
        )
        lle.fit(data)
        self.embedding = jnp.array(lle.transform(data))


class Isomap(StaticEmbedding):
    """
    Provides a static 2D embedding using Isomap, a non-linear
    dimensionality reduction method based on geodesic distances.
    """

    @timing_logger
    def __init__(self, data: jax.Array, **kwargs) -> None:
        super(Isomap, self).__init__(data)
        self.name = "Isomap"

        iso = manifold.Isomap(n_components=2, **kwargs)
        iso.fit(data)
        self.embedding = jnp.array(iso.transform(data))


class ICA(StaticEmbedding):
    """
    Provides a static 2D embedding using Independent Component Analysis (ICA),
    which separates multivariate signals into independent components.
    """

    @timing_logger
    def __init__(self, data: Array, **kwargs) -> None:
        super(ICA, self).__init__(data)
        self.name = "ICA"

        ica = decomposition.FastICA(n_components=2, **kwargs)
        ica.fit(data)
        self.embedding = jnp.array(ica.transform(data))


class tSNE(StaticEmbedding):
    """
    Provides a static 2D embedding using t-distributed Stochastic Neighbor Embedding (t-SNE),
    a non-linear technique for preserving local structure in high-dimensional data.
    """

    @timing_logger
    def __init__(self, data: Array, **kwargs) -> None:
        super(tSNE, self).__init__(data)
        self.name = "tSNE"

        tsne = manifold.TSNE(n_components=2, **kwargs)
        self.embedding = jnp.array(tsne.fit_transform(data))


class MDS(StaticEmbedding):
    """
    Provides a static 2D embedding using Multidimensional Scaling (MDS),
    preserving pairwise distances in a lower-dimensional space.
    """

    @timing_logger
    def __init__(self, data: Array, **kwargs) -> None:
        super(MDS, self).__init__(data)
        self.name = "MDS"

        dists = pairwise_distances(data, **kwargs)
        dists = (dists + dists.T) / 2.0
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed")
        self.embedding = mds.fit_transform(dists)


class cKPCA(InteractiveEmbedding):
    """
    Constrained Kernel PCA (cKPCA) for interactive 2D embedding.

    This class implements an interactive dimensionality reduction technique using
    kernel PCA, where the embedding is iteratively updated based on user-defined
    control points and pairwise constraints (must-link and cannot-link).

    Args:
        data (Array): Input data of shape (n_samples, n_features).
        initial_control_points (dict[int, Array]): Initial control points mapping
            sample indices to 2D positions.
        kernel_name (str): Name of the kernel function to use (e.g., 'rbf', 'polynomial').
        seed (Optional[int]): Random seed for reproducibility. Default is 42.
        **kwargs: Additional kernel-specific parameters (gamma, degree, coef0)
    """

    @timing_logger
    def __init__(
        self,
        data: Array,
        initial_control_points: dict[int, Array],
        kernel_name: str,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> None:
        super(cKPCA, self).__init__(data)
        self.name = "cKPCA"
        self.projection_matrix = jnp.zeros((2, self.n))

        self._rng_key = random.PRNGKey(seed)

        self.cp_selector_m_by_n = None
        self.num_landmarks = int(self.n**0.5)  # A heuristic

        self.kernel_name = kernel_name
        params = self._resolve_kwargs(**kwargs)
        self.kernel_fn = Kernel(kernel_name, **params)

        self._nystroem_fit_transform()
        self._init_principal_axes()
        self._init_objective()
        self._init_principal_axes

        self.set_objective_hyperparameters()
        self.set_solver_hyperparameters()

        self.update_control_points(initial_control_points)

    def _resolve_kwargs(self, **kwargs) -> dict:
        """
        Validate and configure kernel-specific hyperparameters based on the current kernel type.

        This method reads optional keyword arguments to set the parameters `degree`, `gamma`,
        and `coef0` depending on the selected kernel. If a required parameter is not provided,
        a default is computed or used.

        Supported kernels and their hyperparameters:
            - "polynomial":
                - kernel_degree (int): Degree of the polynomial (default: 3).
                - kernel_gamma (float): Scaling factor for input features (default: 1.0 / input_dim).
                - kernel_coef0 (float): Independent term in the polynomial kernel (default: 1.0).

            - "rbf", "rbf_bitvector":
                - kernel_gamma (float): Bandwidth of the RBF kernel. If not provided, it is estimated
                using the median of pairwise distances (default: 1 / (2 * median_dist^2)).

            - "manhattan", "hamming":
                - kernel_gamma (float): Scaling factor (default: 1.0).

            - "tanimoto", "dice", "hamming":
                - These kernels are intended for binary (0/1) data. A warning is issued if the data
                contains values outside [0, 1].

            - "linear", "cosine":
                - No hyperparameters are required.

        Returns:
            dict: The dictionary of kernel parameters.

        Raises:
            ValueError: If an unsupported kernel name is specified.
        """

        params = {}
        if self.kernel_name == "polynomial":
            params["degree"] = int(kwargs.get("degree", 3))
            params["gamma"] = kwargs.get("gamma", 1.0 / self.d)
            params["coef0"] = kwargs.get("coef0", 1.0)
        elif self.kernel_name in {"rbf", "rbf_bitvector"}:
            num_samples = min(10000, self.n)
            self._rng_key, subkey = random.split(self._rng_key)
            indices = random.choice(subkey, self.n, (num_samples,), replace=False)
            params["gamma"] = kwargs.get(
                "gamma",
                1 / (2 * self.median_pairwise_distances(self.data[indices]) ** 2),
            )
        elif self.kernel_name in {"manhattan", "hamming"}:
            params["gamma"] = kwargs.get("gamma", 1.0)
        elif self.kernel_name in {"tanimoto", "dice", "hamming"}:
            if not jnp.array_equal(self.data, jnp.clip(self.data, 0, 1)):
                warnings.warn(
                    f"Kernel '{self.kernel_name}' is intended for binary (0/1) data, "
                    f"but input values are outside [0, 1]. Results may be invalid.",
                    UserWarning,
                )
        elif self.kernel_name in {"linear", "cosine"}:
            pass
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_name}")

        return params

    def _nystroem_fit_transform(self, time_out_sec=30, memory_limit_mb=4096):
        """
        Compute the Nyström approximation of the kernel using kernel k-means for landmark selection.
        Execution is subject to a timeout and memory usage limit that kill the interpreter if they are exceeded.

        Args:
            time_out_sec (int): Maximum allowed execution time in seconds.
            memory_limit_mb (int): Maximum allowed memory usage in megabytes.
        """

        def body():
            start = time.perf_counter()
            kkmeanspp = KernelKMeansSQ(self.kernel_fn)
            nystroem = Nystroem(self.kernel_fn)
            try:
                nystroem.fit(self.data, kkmeanspp, self.num_landmarks)
                self.S = nystroem.transform(self.data)
                self.S.block_until_ready()
                logger.info(
                    f"{self.kernel_name.title()} kernel approximation finished in "
                    f"{1000*(time.perf_counter() - start):.2f} ms"
                )
                self.nystroem = nystroem
            except Exception as e:
                logger.error(f"Kernel approximation failed: {e}")

        def monitor_memory(stop_event):
            process = psutil.Process(os.getpid())
            while not stop_event.is_set():
                mem = process.memory_info().rss / (1024**2)
                if mem > memory_limit_mb:
                    logger.error(
                        f"Memory limit exceeded: {mem:.2f} MB > {memory_limit_mb} MB"
                    )
                    os._exit(1)
                time.sleep(0.1)

        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_memory, args=(stop_event,))
        monitor_thread.start()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(body)
            try:
                future.result(timeout=time_out_sec)
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"Timeout: kernel approximation took longer than {time_out_sec} seconds"
                )
                os._exit(1)
            finally:
                stop_event.set()
                monitor_thread.join()

    def _init_objective(self) -> None:
        """
        Intialize the quadratic part of the objective function.
        """

        temp = self.S.T @ jnp.ones((self.n, 1))
        self.C_var = (1 / self.n) * (
            (self.S.T @ self.S) - ((1 / self.n) * jnp.outer(temp, temp))
        )
        self.C_cp = 0
        self.C_ml_cl = 0

    def _init_principal_axes(self, n_axes: int = 2) -> None:
        """
        Intialize the principal axes randomly.

        Args:
            n_axes (int): The number of principal axes to compute, defaults to 2.
        """

        self.principal_axes = []
        for _ in range(n_axes):
            self._rng_key, subkey = random.split(self._rng_key)
            v = random.uniform(
                subkey, shape=(self.num_landmarks, 1), minval=0.0, maxval=1.0
            )
            self.principal_axes.append((v / jnp.linalg.norm(v)).reshape(-1, 1))

    def set_objective_hyperparameters(self, **kwargs) -> None:
        """
        Set hyperparameters that control the quadratic part of the objective function.

        Keyword Args:
            cp_const_mu (float): Weight for the control point alignment term.
                Encourages the embedding to match control points. Default is 100.
            cl_ml_const_mu (float): Weight for the cannot-link and must-link constraint terms.
                Controls how strongly to enforce pairwise constraints. Default is 1.
            orth_mu (float): Weight for the orthogonality penalty, which encourages
                orthogonal projection directions. Default is 10.
        """

        self.cp_const_mu = kwargs.get("cp_const_mu", 100)
        self.cl_ml_const_mu = kwargs.get("cl_ml_const_mu", 1)
        self.orth_mu = kwargs.get("orth_mu", 10)

        self.cl_ml_const_mu_original = self.cl_ml_const_mu

    def adjust_link_strength(self, m: float) -> None:
        """
        Adjust must-link and cannot-link constraint strength.

        Args:
            m (float): Multiplier for must-link and cannot-link strength
        """

        self.cl_ml_const_mu = m * self.cl_ml_const_mu_original
        self._on_constraints_updated()

    def set_solver_hyperparameters(self, **kwargs) -> None:
        """
        Set hyperparameters for the optimization solver.

        Keyword Args:
            learning_rate (float): Learning rate used in the optimization algorithm.
                Controls the step size during updates. Default is 0.1.
            beta1 (float): Exponential decay rate for the first moment estimate in
                optimizers like Adam. Default is 0.1.
        """

        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.beta1 = kwargs.get("beta1", 0.95)

    def _on_constraints_updated(self) -> None:
        self._update_ml_cl_params()
        self.iteration()
        self.projection_matrix = self._construct_projection_matrix(self.principal_axes)

    @staticmethod
    def _constraints_to_jax(constraints: set[set[int]]) -> Array:
        constraint_list = [tuple(sorted(c)) for c in constraints if len(c) == 2]
        return jnp.array(constraint_list)

    def _update_ml_cl_params(self) -> None:
        """
        Compute the regularization matrix associated with must-link and cannot-link constraints.

        This method constructs a normalized Laplacian matrix that encodes pairwise
        relationships based on the provided must-link and cannot-link constraints.
        The matrix is scaled by `cl_ml_const_mu` and used as part of the overall
        optimization objective to guide the embedding.

        If no constraints are present, the penalty term is set to zero.
        """

        if self.must_link or self.cannot_link:
            # Construct a weighted Laplacian matrix
            self.C_ml_cl = (
                self.cl_ml_const_mu / (len(self.must_link) + len(self.cannot_link))
            ) * self._construct_ml_cl_laplacian_matrix(
                self._constraints_to_jax(self.must_link),
                self._constraints_to_jax(self.cannot_link),
                self.S,
                self.num_landmarks,
            )
        else:
            self.C_ml_cl = 0

    def update_control_points(self, points: dict[int, Array]) -> None:
        """
        Update control point features and their 2D embeddings and trigger the re-computation
        of the control parameters, the principal axes, and the projection matrix.

        Args:
            points (dict): Mapping of control point indices to 2D coordinates.
        """

        new_indices = points.keys() != self.control_points.keys()

        super().update_control_points(points)

        if new_indices:
            self._update_cp_params()
        self.iteration()
        self.projection_matrix = self._construct_projection_matrix(self.principal_axes)

    @staticmethod
    @jit
    def _compute_C_cp(S: Array, indices: Array, mu: float) -> Array:
        """
        Compute the control point regularization matrix.

        This matrix is formed by summing the outer products of selected rows from
        the similarity matrix `S`, corresponding to control points. It encourages
        the learned embedding to align with the structure induced by those control points.

        Args:
            S (Array): Sample-to-landmark similarity matrix of shape (n_samples, n_landmarks).
            indices (Array): Indices of control points within the sample set.
            mu (float): Regularization weight for the control point term.

        Returns:
            Array: A (n_landmarks x n_landmarks) regularization matrix scaled by `mu / len(indices)`.
        """

        def body(i, acc):
            s_i = S[i]
            return acc + jnp.outer(s_i, s_i)

        C_cp = lax.fori_loop(
            0,
            len(indices),
            lambda i, acc: body(indices[i], acc),
            jnp.zeros((S.shape[1], S.shape[1])),
        )
        return (mu / len(indices)) * C_cp

    def _update_cp_params(self):
        """
        Constructs a selector matrix `cp_selector_m_by_n` mapping control points to the
        full sample space and compute the control point regularization matrix `C_cp`.

        If no control points are present, `C_cp` is set to a zero matrix.
        """

        indices = jnp.array(list(self.control_points.keys()), dtype=jnp.int64)
        self.cp_selector_m_by_n = self.construct_mn_selector_matrix(self.n, indices)
        self.C_cp = lax.cond(
            indices.size > 0,
            lambda idx: self._compute_C_cp(self.S, idx, self.cp_const_mu),
            lambda _: jnp.zeros((self.S.shape[1], self.S.shape[1])),
            indices,
        )

    def iteration(self) -> None:
        """
        Solve for the new principal axes.
        """

        start = time.perf_counter()

        for i in range(len(self.principal_axes)):
            if i == 0:
                prev = None
            else:
                prev = self.principal_axes[i - 1]
            self.principal_axes[i] = self._solve_iteratively(
                prev, i, self.principal_axes[i]
            )

        logger.debug(f"Solver converged in {time.perf_counter()-start:.2f} ms")

    @staticmethod
    @jit
    def _construct_projection_matrix(principal_axes: list) -> Array:
        """
        Construct the new projection matrix from the principal axes by stacking them.

        Args:
            principal_axes (list): List of the principal axes with shape (n_landmarks,)

        Returns:
            Array: The projection matrix of shape (n_landmarks, n_axes)
        """

        return jnp.hstack(principal_axes).T

    @staticmethod
    @jit
    def _adjust_learning_rate(
        v_new: Array, v: Array, learning_rate: float
    ) -> Tuple[Array, float, bool]:
        """
        Dynamically adjust the learning rate if the solution diverging or collapsing
        and normalize the new solution.

        Args:
            v_new (Array): The new principal axis after the solver iteration, shape (n_landmarks, 1)
            v (Array): The principal axis before the solver iteration, shape (n_landmarks, 1)
            learning_rate (float): The current learning rate

        Returns:
            Tuple[Array, float, bool]
                -Normalized principal axis (shape (n_landmarks, 1)).
                - Updated learning rate (halved if diverging/collapsing, else unchanged).
                - Flag indicating whether the learning rate was adjusted; when True, convergence
                check should be skipped for this iteration.
        """

        diff_norm = jnp.linalg.norm(v_new - v)
        v_new_norm = jnp.linalg.norm(v_new)

        diverging = diff_norm > 1.5
        collapsing = v_new_norm < 1e-8
        continue_flag = diverging | collapsing

        new_lr = jnp.where(continue_flag, learning_rate / 2.0, learning_rate)
        v_new_normalized = jnp.where(v_new_norm > 1e-8, v_new / v_new_norm, v)

        return v_new_normalized, new_lr, continue_flag

    @staticmethod
    @partial(jit, static_argnames=["max_iters"])
    def _solve_iteratively_jit(
        C: Array,
        b: Array,
        v_prev: Array,
        v_init: Array,
        orth_mu: float,
        learning_rate: float,
        beta1: float,
        tol: Optional[float] = 1e-9,
        max_iters: Optional[int] = 10000,
    ) -> Array:
        """
        JIT-compiled wrapper for the `_solve_iteratively` method.
        """

        # Orthogonalize to the previous component
        if v_prev is not None:
            C = C - orth_mu * jnp.outer(v_prev, v_prev)

        b = b.reshape(1, -1)

        def cond_fn(state):
            _, _, i, converged, _ = state
            return jnp.logical_and(i < max_iters, jnp.logical_not(converged))

        def body_fn(state):
            v, v_dw, i, _, lr = state

            v_lookahead = v + beta1 * v_dw
            dw = C @ v_lookahead - b.T
            v_dw_new = beta1 * v_dw + lr * dw
            v_new = v + v_dw_new

            # Possibly adjust the learning rate
            v_new_normalized, new_lr, continue_flag = cKPCA._adjust_learning_rate(
                v_new, v, lr
            )

            # Zero the gradient if the learning rate has been adjusted
            v_dw_reset = jnp.where(continue_flag, jnp.zeros_like(v_dw), v_dw_new)

            delta = jnp.linalg.norm(v_new_normalized - v)
            converged = (delta < tol) & jnp.logical_not(continue_flag)

            return v_new_normalized, v_dw_reset, i + 1, converged, new_lr

        # Normalize the initial value with safety epsilon to avoid div by zero
        v_init = v_init / jnp.linalg.norm(v_init + 1e-12)

        # Zero the gradient
        v_dw_init = jnp.zeros_like(v_init)

        state = (v_init, v_dw_init, 0, False, learning_rate)
        v_final, _, _, _, _ = lax.while_loop(cond_fn, body_fn, state)

        return v_final

    def _solve_iteratively(self, v_prev: Array, dim: int, v_init: Array) -> Array:
        """
        Solve the constrained optimization problem for one principal axis.

        Args:
            v_prev (Array): The previous principal axis, shape (n_landmarks, 1).
            dim (int): The dimension to solve for.
            v_init (Array): The current principal axis, shape (n_landmarks, 1)
        """

        # The quadratic part of the optimization objective
        C = self.C_var - self.C_cp - self.C_ml_cl

        # The linear part of the optimization objective
        b = jnp.zeros((1, self.num_landmarks))
        n_control_points = len(self.control_points)
        if n_control_points > 0:
            y_s = self.y[:, dim]
            b = (
                -1
                * self.cp_const_mu
                / n_control_points
                * (y_s.T @ self.cp_selector_m_by_n @ self.S)
            ).reshape(1, -1)

        return self._solve_iteratively_jit(
            C=C,
            b=b,
            v_prev=v_prev,
            v_init=v_init,
            orth_mu=self.orth_mu,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
        )

    @staticmethod
    @jit
    def median_pairwise_distances(X: Array) -> Array:
        """
        Compute the median of all pairwise Euclidean distances between rows of `X`.

        Args:
            X (Array): Input data matrix of shape (n_samples, n_features).

        Returns:
            Array: The median of the pairwise Euclidean distances between samples.
        """

        def pairwise_dists(X):
            # Compute squared Euclidean distances between all pairs
            diffs = jnp.expand_dims(X, 0) - jnp.expand_dims(X, 1)
            dist_matrix = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
            return dist_matrix

        distances = pairwise_dists(X)
        # Only take the upper triangle without the diagonal
        i, j = jnp.triu_indices(distances.shape[0], k=1)
        upper_tri_values = distances[i, j]
        return jnp.median(upper_tri_values)

    @staticmethod
    @partial(jit, static_argnames=["n"])
    def construct_mn_selector_matrix(n, control_point_indices: Array) -> Array:
        """
        Construct a selector matrix that maps control points to the full sample space.

        This matrix `M` has shape (m, n), where `m` is the number of control points and
        `n` is the total number of samples. Each row of `M` contains a single 1.0 at the
        column corresponding to the control point index, and zeros elsewhere.

        Args:
            n (int): Total number of samples.
            control_point_indices (Array): Indices of the control points, shape (m, 1).

        Returns:
            Array: A binary selector matrix of shape (m, n), where each row selects a control point.
        """

        m = len(control_point_indices)
        m_matrix = jnp.zeros((m, n))
        m_matrix = m_matrix.at[jnp.arange(m), control_point_indices].set(1.0)
        return m_matrix

    @staticmethod
    @partial(jit, static_argnames=["num_landmarks"])
    def _construct_ml_cl_laplacian_matrix(
        ml: Array, cl: Array, S: Array, num_landmarks: int
    ) -> Array:
        """
        Construct a Laplacian-like matrix based on must-link and cannot-link constraints.

        This matrix is computed as a sum of outer products of difference vectors between
        pairs of points in the similarity matrix `S`. Must-link (ml) constraints contribute
        positively and cannot-link (cl) constraints contribute negatively.

        Args:
            ml (Array): Must-link constraint pairs, shape (num_ml_constraints, 2), each row contains indices (i, j).
            cl (Array): Cannot-link constraint pairs, shape (num_cl_constraints, 2), each row contains indices (i, j).
            S (Array): Similarity matrix between samples and landmarks, shape (n_samples, num_landmarks).
            num_landmarks (int): Number of landmarks.

        Returns:
            Array: A (num_landmarks x num_landmarks) matrix representing the weighted Laplacian
            based on the provided constraints.
        """

        def update_with_constraint(C, pair, sign):
            i, j = pair
            diff = S[i] - S[j]
            outer = jnp.outer(diff, diff)
            return C + sign * outer

        def apply_constraints(C, constraints, sign):
            if constraints.shape[0] == 0:
                return C

            def body_fn(i, C):
                return update_with_constraint(C, constraints[i], sign)

            return lax.fori_loop(0, constraints.shape[0], body_fn, C)

        C_ml_cl = jnp.zeros((num_landmarks, num_landmarks))
        C_ml_cl = apply_constraints(C_ml_cl, ml, +1.0)
        C_ml_cl = apply_constraints(C_ml_cl, cl, -1.0)

        return C_ml_cl

    def add_new_point(self, point: Array) -> Array:
        """
        Add a new data point, update the Nyström kernel approximation,
        and project the point into the current 2D embedding.

        Args:
            point (Array): A single point in the original feature space, shape (1, n_features)

        Returns:
            Array: The projected 2D coordinates of the point, shape (1, 2).
        """

        # Normalize if the original data was normalized
        if self.normalize:
            point = self.normalize_new_point(point)

        self.n += 1
        self.data = jnp.vstack([self.data, point])

        # Find the kernel approximation at the new point
        c = self.nystroem.transform(point[None, :])
        self.S = jnp.vstack([self.S, c])

        # Recompute variance
        sum_S = jnp.sum(self.S, axis=0)
        self.C_var = (1 / self.n) * (self.S.T @ self.S) - (1 / self.n**2) * jnp.outer(
            sum_S, sum_S
        )

        # Need to compute new control point selector
        indices = jnp.array(list(self.control_points.keys()), dtype=jnp.int64)
        self.cp_selector_m_by_n = self.construct_mn_selector_matrix(self.n, indices)

        # Construct the new projection matrix
        self.iteration()
        self.projection_matrix = self._construct_projection_matrix(self.principal_axes)

        projection = self.projection_matrix @ c.T
        return projection

    def get_embedding(self) -> Array:
        """
        Project the data via the projection matrix.

        Returns:
            The 2D embedding of the data as an array of shape (2, n_samples)
        """

        return self.projection_matrix @ self.S.T


class LSP(InteractiveEmbedding):
    """
    Provides an interactive 2D embedding by performing the
    Least Squared Error Projection (LSP) with constraints.
    """

    @timing_logger
    def __init__(self, data: Array, initial_control_points: dict[int, Array]):
        super().__init__(data)
        self.name = "LSP"

        self.update_control_points(initial_control_points)

    def get_embedding(self, X=[]):
        if X == []:
            X = self.data.T
        return jnp.dot(self.projection_matrix, X)

    def update_control_points(self, points: dict[int, Array]):
        super(LSP, self).update_control_points(points)
        if len(self.y) > 0:
            self.projection_matrix = jnp.dot(self.y.T, jnp.linalg.pinv(self.X.T))
        else:
            self.projection_matrix = jnp.zeros((2, len(self.data[0])))
        if self.has_ml_cl_constraints:
            self.augment_control_points()
            if len(self.y) > 0:
                self.projection_matrix = jnp.dot(self.y.T, jnp.linalg.pinv(self.X.T))
            else:
                self.projection_matrix = jnp.zeros((2, len(self.data[0])))