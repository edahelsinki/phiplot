from functools import partial
from typing import Optional, Callable
from jax import numpy as jnp, jit, Array


class Kernel:
    """
    A unified interface for computing various kernel similarity functions using JAX.

    Supports kernels for both continuous and binary feature spaces, including:
        - linear
        - polynomial
        - RBF (Gaussian)
        - cosine
        - manhattan (L1-based RBF)
        - tanimoto (for bit vectors)
        - dice (for binary similarity)
        - hamming (bitwise exponential decay)

    Attributes:
        name (str): The name of the kernel function to use.
        **params: Additional kernel-specific hyperparameters (e.g., gamma, degree, coef0).

    Example:
        k = Kernel("rbf", gamma=0.5)
        similarity = k(X, Y)
    """

    AVAILABLE_KERNELS = sorted(
        [
            "linear",
            "polynomial",
            "rbf",
            "rbf_bitvector",
            "tanimoto",
            "manhattan",
            "cosine",
            "dice",
            "hamming",
        ]
    )

    def __init__(self, name: str, **params):
        self.name = name
        params = self._resolve_params(params)
        self.kernel_fn = self._build_kernel_fn(name, params)

    def __call__(self, X: Array, Y: Optional[Array] = None) -> Array:
        """
        Evaluate the kernel between two input sets X and Y.

        Args:
            X (Array): Input array of shape (n_samples_X, n_features).
            Y (Optional[Array], optional): Input array of shape (n_samples_Y, n_features).
                If None, defaults to X (i.e., self-similarity).

        Returns:
            Array: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """

        X = jnp.asarray(X, dtype=jnp.float64)
        Y = jnp.asarray(Y if Y is not None else X, dtype=jnp.float64)

        # Reshape to 2D if the input is an 1D array
        if X.ndim == 1:
            X = X[None, :]
        if Y.ndim == 1:
            Y = Y[None, :]

        return self.kernel_fn(X, Y)

    def _resolve_params(self, params):
        """
        Extracts and assigns default hyperparameters for kernel functions.

        Args:
            params (dict): Dictionary of parameters passed to the kernel class.

        Returns:
            dict: A dictionary with keys 'gamma', 'degree', and 'coef0', each set to
                either the provided value or a default:
                    - gamma: default = 1.0
                    - degree: default = 3
                    - coef0: default = 1.0
        """

        return {
            "gamma": params.get("gamma", 1.0),
            "degree": params.get("degree", 3),
            "coef0": params.get("coef0", 1.0),
        }

    def _build_kernel_fn(
        self, name: str, params: dict
    ) -> Callable[[Array, Array], Array]:
        """
        Constructs a callable kernel function based on the kernel name and parameters.

        Args:
            name (str): The name of the kernel to use.
            params (dict): Dictionary of resolved kernel parameters ('gamma', 'degree', 'coef0').

        Returns:
            Callable[[Array, Array], Array]: A JAX-compatible function that computes the kernel matrix.

        Raises:
            ValueError: If an unknown kernel name is provided.
        """

        gamma = params.get("gamma", None)
        degree = params.get("degree", None)
        coef0 = params.get("coef0", None)

        kernel_map = {
            "linear": self.linear_kernel,
            "polynomial": partial(
                self.polynomial_kernel, degree=degree, gamma=gamma, coef0=coef0
            ),
            "rbf": partial(self.rbf_kernel, gamma=gamma),
            "rbf_bitvector": partial(self.rbf_kernel_bitvector, gamma=gamma),
            "tanimoto": self.tanimoto_kernel,
            "manhattan": partial(self.manhattan_kernel, gamma=gamma),
            "cosine": self.cosine_kernel,
            "dice": self.dice_kernel,
            "hamming": partial(self.hamming_kernel, gamma=gamma),
        }

        if name not in kernel_map:
            raise ValueError(
                f"Unknown kernel: '{name}'.  Available: {', '.join(self.AVAILABLE_KERNELS)}"
            )
        return kernel_map[name]

    @staticmethod
    @jit
    def linear_kernel(X: Array, Y: Array) -> Array:
        """
        Computes:
            K(x, y) = x · y

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        return jnp.dot(X, Y.T)

    @staticmethod
    @jit
    def polynomial_kernel(
        X: Array, Y: Array, degree: int, gamma: float, coef0: float
    ) -> Array:
        """
        Computes:
            K(x, y) = (γ · x · y + c)^d

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        return (gamma * jnp.dot(X, Y.T) + coef0) ** degree

    @staticmethod
    @jit
    def rbf_kernel(X: Array, Y: Array, gamma: float) -> Array:
        """
        Computes:
            K(x, y) = exp(-γ · ||x - y||²)

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)
            gamma (float): Kernel width parameter

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        diff = X[:, None, :] - Y[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        return jnp.exp(-gamma * dist_sq)

    @staticmethod
    @jit
    def rbf_kernel_bitvector(X: Array, Y: Array, gamma: float) -> Array:
        """
        Efficient RBF kernel for binary vectors:
            K(x, y) = exp(-γ · (||x||² + ||y||² - 2x·y))
        """
        dot = jnp.dot(X, Y.T)
        norm_X = jnp.sum(X, axis=1, keepdims=True)
        norm_Y = jnp.sum(Y, axis=1, keepdims=True).T
        dist_sq = norm_X + norm_Y - 2 * dot
        return jnp.exp(-gamma * dist_sq)

    @staticmethod
    @jit
    def tanimoto_kernel(X: Array, Y: Array) -> Array:
        """
        Computes:
            K(x, y) = (x · y) / (||x||² + ||y||² - x · y)

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """
        numerator = jnp.dot(X, Y.T)
        norm_X = jnp.sum(X * X, axis=1, keepdims=True)
        norm_Y = jnp.sum(Y * Y, axis=1, keepdims=True).T
        denominator = norm_X + norm_Y - numerator
        return jnp.where(denominator <= 1e-12, 1.0, numerator / denominator)

    @staticmethod
    @jit
    def dice_kernel(X: Array, Y: Array) -> Array:
        """
        Computes:
            K(x, y) = 2 · (x · y) / (||x||₁ + ||y||₁)

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        intersection = jnp.dot(X, Y.T)
        sum_X = jnp.sum(X, axis=1, keepdims=True)
        sum_Y = jnp.sum(Y, axis=1, keepdims=True).T
        denominator = sum_X + sum_Y
        return jnp.where(denominator <= 1e-12, 1.0, 2.0 * intersection / denominator)

    @staticmethod
    @jit
    def hamming_kernel(X: Array, Y: Array, gamma: float) -> Array:
        """
        Computes:
            K(x, y) = exp(-γ · H(x, y))

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        diff = jnp.abs(X[:, None, :] - Y[None, :, :])
        hamming_dist = jnp.mean(diff, axis=-1)
        return jnp.exp(-gamma * hamming_dist)

    @staticmethod
    @jit
    def manhattan_kernel(X: Array, Y: Array, gamma) -> Array:
        """
        Computes:
            K(x, y) = exp(-γ · ||x - y||₁)

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        diff = jnp.abs(X[:, None, :] - Y[None, :, :])
        dists = jnp.sum(diff, axis=2)
        return jnp.exp(-gamma * dists)

    @staticmethod
    @jit
    def cosine_kernel(X: Array, Y: Array) -> Array:
        """
        Computes:
            K(x, y) = (x · y) / (||x|| · ||y||)

        Args:
            X (Array): Shape (n_samples_X, n_features)
            Y (Array): Shape (n_samples_Y, n_features)

        Returns:
            Array: Shape (n_samples_X, n_samples_Y)
        """

        dot = jnp.dot(X, Y.T)
        norm_X = jnp.linalg.norm(X, axis=1, keepdims=True)
        norm_Y = jnp.linalg.norm(Y, axis=1, keepdims=True)
        norm_product = jnp.outer(norm_X.squeeze(-1), norm_Y.squeeze(-1))
        return jnp.where(norm_product <= 1e-12, 0.0, dot / norm_product)
