from functools import partial
import logging
from typing import Callable, Tuple, Optional
from jax import numpy as jnp, jit, lax, random, vmap, Array
import numpy as np

logger = logging.getLogger(__name__)


class Nystroem:
    """
    Implements the Nystroem method for approximate kernel mapping using a
    low-rank factorization based on a subset of landmark points.

    Args:
        kernel_fn (Callable[[Array, Array], Array]): A function that computes the
            kernel matrix between two inputs.
    """

    def __init__(self, kernel_fn: Callable[[Array, Array], Array]) -> None:
        self.kernel_fn = kernel_fn

    @staticmethod
    @jit
    def center_kernel(K: Array) -> Array:
        """
        Center the kernel matrix K using the centering trick:
            `K_centered = (I - 1_n) @ K @ (I - 1_n)`

        where:
            - `I` is the n x n identity matrix
            - `1_n` is an n x n matrix with all elements equal to 1/n

        This operation ensures that the centered kernel has zero mean in feature space.

        Args:
            K (Array): Square kernel matrix of shape (n, n) to be centered.

        Returns:
            Array: The centered kernel matrix of shape (n, n).
        """

        n = K.shape[0]
        H = jnp.eye(n) - jnp.ones((n, n)) / n
        return H @ K @ H

    @staticmethod
    @jit
    def center_cross_kernel(K_xl: Array, K_ll: Array) -> Array:
        """
        Center the cross-kernel matrix `K_xl` using the centering information
        from the landmark kernel matrix `K_ll`.

        The centered matrix is computed as:
           `K_xl_centered = K_xl - mean_K_xl - mean_K_ll.T + mean_total`

        where means are computed over the columns (landmarks).

        Args:
            K_xl (Array): Cross-kernel matrix of shape (n_samples, n_landmarks)
            K_ll (Array): Kernel matrix of the landmarks (n_landmarks, n_landmarks)

        Returns:
            Array: Centered cross-kernel matrix of shape (n_samples, n_landmarks)
        """

        n_landmarks = K_ll.shape[0]
        one_l = jnp.ones((n_landmarks, 1)) / n_landmarks

        mean_K_ll = K_ll @ one_l
        mean_K_xl = K_xl @ one_l
        mean_total = one_l.T @ K_ll @ one_l

        return K_xl - mean_K_xl - mean_K_ll.T + mean_total

    @staticmethod
    @partial(jit, static_argnames=["kernel_fn"])
    def _transform_jit(
        kernel_fn: Callable[[Array, Array], Array], X: Array, landmarks: Array
    ) -> Array:
        """
        JIT-compiled wrapper for the `transform` method.
        """

        # Compute and center kernel of the landmarks
        K_ll = kernel_fn(landmarks, landmarks)
        K_ll_centered = Nystroem.center_kernel(K_ll)

        # Eigendecomposition
        eig_vals, eig_vecs = jnp.linalg.eigh(K_ll_centered)

        # Regularize and clip eigenvalues for numerical stability
        clipped_eig_vals = jnp.clip(eig_vals, a_min=1e-8)

        # Normalize eigenvectors
        sqrt_eig_vals = jnp.sqrt(clipped_eig_vals)
        normed_eig_vecs = eig_vecs / sqrt_eig_vals

        # Centered cross-kernel between X and landmarks
        K_xl = kernel_fn(X, landmarks)
        K_xl_centered = Nystroem.center_cross_kernel(K_xl, K_ll_centered)

        # Project X
        return K_xl_centered @ normed_eig_vecs

    def fit(self, X: Array, landmark_selector=None, n_landmarks: int = 100):
        """
        Selects landmark points for the Nyström approximation.

        If no landmark_selector is provided, the first `n_landmarks` samples from `X` are used
        as landmarks by default.

        Args:
            X (Array): Input data matrix of shape (n_samples, n_features).
            landmark_selector (optional): An object with a `.select(X, n_landmarks)` method
                that returns landmark points. Defaults to None.
            n_landmarks (int): Number of landmarks to select. Defaults to 100.

        Returns:
            self: Returns the instance itself to allow method chaining.
        """

        if landmark_selector is None:
            self.landmarks = X[:n_landmarks]
            logger.info("Using naive landmark selector")
        else:
            self.landmarks = landmark_selector.select(X, n_landmarks)
        return self

    def transform(self, X: Array) -> Array:
        """
        Approximate the kernel feature map for input `X` using the Nyström method.

        Requires that `fit` has been called beforehand to select landmark points.

        Args:
            X (Array): Input data matrix of shape (n_samples, n_features).

        Returns:
            Array: Projected feature representation of `X`, shape (n_samples, n_landmarks).

        Raises:
            ValueError: If `fit` has not been called before `transform`.
        """

        if not hasattr(self, "landmarks"):
            raise ValueError("Call fit() before transform()")
        return self._transform_jit(self.kernel_fn, X, self.landmarks)


class KernelKMeansSQ:
    """
    Approximates kernel k-means++ landmark selection using
    kernel distances and a probabilistic sampling strategy.

    Args:
        kernel_fn (Callable[[Array, Array], Array]): A function that computes the
            kernel matrix between two inputs.
        seed (Optional[int]): Random seed for reproducibility. Default is 42.
    """

    def __init__(
        self,
        kernel_fn: Callable[[Array, Array], Array],
        seed: int = 42,
    ) -> None:
        self.kernel_fn = kernel_fn
        self._rng_key = random.PRNGKey(seed)

    @staticmethod
    @partial(jit, static_argnames=["kernel_fn", "num_local_trials"])
    def _sample_cluster_centers(
        rng_key: Array,
        X: Array,
        kmat_diagonal: Array,
        cp_distances: Array,
        cp_potential: Array,
        num_local_trials: int,
        kernel_fn: Callable[[Array, Array], Array],
    ) -> Tuple[int, Array, float]:
        """
        Samples a new cluster center using probabilistic selection based on kernel distances.

        Args:
            rng_key (Array): PRNG key for random sampling.
            X (Array): Input data of shape (n_samples, n_features).
            kmat_diagonal (Array): Kernel diagonal values (k(x_i, x_i) for all i).
            cp_distances (Array): Current distances from selected landmarks to all points.
            cp_potential (float): Current sum of distances, used for sampling distribution.
            num_local_trials (int): Number of candidate centers to consider in each iteration.

        Returns:
            Tuple[int, Array, float]
                - Index of selected new landmark.
                - Updated cp_distances.
                - Updated cp_potential.
        """

        # Select candidates via inverse CDF method
        rand_vals = random.uniform(rng_key, shape=(num_local_trials,)) * cp_potential
        cp_dist_cumsum = jnp.cumsum(cp_distances + 1e-10)
        candidate_ids = jnp.searchsorted(cp_dist_cumsum, rand_vals).astype(jnp.int64)
        X_candidates = X[candidate_ids]

        # Compute squared distances to canditates via the kernel trick
        K_cand_X = kernel_fn(X_candidates, X)
        k_cand = kmat_diagonal[candidate_ids]
        distance_to_candidates = k_cand[:, None] + kmat_diagonal[None, :] - 2 * K_cand_X

        def body_fn(i, state):
            best_candidate, best_cp_distances, best_cp_potential = state
            candidate = candidate_ids[i]

            # Compute canditate potential from updated distances
            dists = distance_to_candidates[i]
            candidate_cp_distances = jnp.minimum(cp_distances, dists)
            candidate_potential = candidate_cp_distances.sum()

            def update():
                return (candidate, candidate_cp_distances, candidate_potential)

            def keep():
                return (best_candidate, best_cp_distances, best_cp_potential)

            # Update best candidate if none selected yet or found better potential
            cond = jnp.logical_or(
                best_candidate == -1, candidate_potential < best_cp_potential
            )
            return lax.cond(cond, update, keep)

        # Initialize the best candidate state with invalid candidate and infinite potential
        best_candidate_init = jnp.array(-1, dtype=jnp.int64)
        init_state = (best_candidate_init, cp_distances, jnp.inf)

        return lax.fori_loop(0, candidate_ids.shape[0], body_fn, init_state)

    @staticmethod
    @partial(jit, static_argnames=["kernel_fn", "num_landmarks", "num_local_trials"])
    def _select_jit(
        rng_key: Array,
        X: Array,
        num_landmarks: int,
        num_local_trials: int,
        kernel_fn: Callable[[Array, Array], Array],
    ) -> Array:
        """
        JIT-compiled wrapper for the `select` method.
        """

        n, d = X.shape

        # Compute kernel diagonal elements k(x_i, x_i) for all points x_i in X
        kmat_diagonal = vmap(lambda x: kernel_fn(x[None, :], x[None, :])[0])(X)
        kmat_diagonal = jnp.squeeze(kmat_diagonal)

        # Randomly select initial landmark uniformly from data points
        rng_keys = random.split(rng_key, num_landmarks)
        init_landmark_id = random.randint(rng_keys[0], shape=(), minval=0, maxval=n)
        x_i = lax.dynamic_slice(X, (init_landmark_id, 0), (1, d))

        # Compute initial distances from the selected landmark to all points via using the kernel trick
        k_init = kernel_fn(x_i, X)[0]
        cp_dists = kmat_diagonal[init_landmark_id] + kmat_diagonal - 2 * k_init
        cp_potential = cp_dists.sum()

        # Initialize selected IDs
        selected_ids = jnp.full((num_landmarks,), -1, dtype=jnp.int64)
        selected_ids = selected_ids.at[0].set(init_landmark_id)

        # Iteratively sample new landmarks to update distances and potentials,
        # refining the selection of representative points using the kernel k-means++ logic
        def scan_body(state, i):
            rng_key, cp_dists, cp_potential, selected_ids = state

            new_landmark_id, new_cp_dists, new_cp_potential = (
                KernelKMeansSQ._sample_cluster_centers(
                    rng_key,
                    X,
                    kmat_diagonal,
                    cp_dists,
                    cp_potential,
                    num_local_trials,
                    kernel_fn,
                )
            )

            selected_ids = lax.dynamic_update_index_in_dim(
                selected_ids, new_landmark_id, i + 1, axis=0
            )

            return (
                rng_key,
                new_cp_dists,
                new_cp_potential,
                selected_ids,
            ), new_landmark_id

        init_state = (rng_keys[1], cp_dists, cp_potential, selected_ids)

        # Run scan for remaining landmarks
        final_state, _ = lax.scan(
            scan_body, init_state, xs=jnp.arange(num_landmarks - 1)
        )
        _, _, _, selected_ids = final_state
        return X[selected_ids]

    def select(
        self, X: Array, num_landmarks: int, params: Optional[dict] = None
    ) -> Array:
        """
        Selects landmark points using a kernel k-means++ inspired sampling scheme.

        Args:
            X (Array): Input dataset of shape (n_samples, n_features).
            num_landmarks (int): Number of landmark points to select.
            params (Optional[dict]): Dictionary with optional key 'num_local_trials'.
                Defaults to None, which uses heuristic value 2 + log(num_landmarks).

        Returns:
            Array: Selected landmark points of shape (num_landmarks, n_features).
        """

        if params is None:
            params = {}

        num_local_trials = params.get(
            "num_local_trials", int(2 + np.log(num_landmarks))
        )

        self._rng_key, subkey = random.split(self._rng_key)

        return self._select_jit(
            subkey, X, num_landmarks, num_local_trials, self.kernel_fn
        )
