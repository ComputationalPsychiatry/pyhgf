# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized deep predictive coding network.

This module provides a vectorized implementation of deep HGF networks that uses
layer-wise matrix operations instead of per-node updates.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from pyhgf.model.vectorized_types import LayerParams, LayerState, NetworkState
from pyhgf.updates.vectorized import (
    vectorized_layer_posterior_update,
    vectorized_layer_prediction,
    vectorized_layer_prediction_error,
    vectorized_weight_update,
)


def linear(x):
    """Linear (identity) activation function."""
    return x


class VectorizedDeepNetwork:
    """Deep predictive coding network with vectorized operations.

    This class implements a deep hierarchical Gaussian filter using layer-wise
    vectorized operations for efficient scaling to large networks.

    Unlike the standard Network which uses per-node updates with Python loops, this
    implementation uses JAX matrix operations to update all nodes in a layer
    simultaneously.

    Examples
    --------
    >>> # Build a network with method chaining
    >>> net = (
    ...     VectorizedDeepNetwork()
    ...     .add_nodes(n_nodes=10)  # Output layer
    ...     .add_layer(size=8)      # Hidden layer 1
    ...     .add_layer(size=6)      # Hidden layer 2
    ...     .add_layer(size=4)      # Input layer
    ... )
    >>>
    >>> # Fit to data
    >>> net.fit(x_train, y_train, lr=0.2)
    >>>
    >>> # Make predictions
    >>> predictions = net.predict(x_test)

    Notes
    -----
    The network uses volatile nodes internally, which have two implicit levels:
    - Value level: represents the node's belief about its value
    - Volatility level: represents uncertainty about the value level

    Layer indexing follows the convention:
    - Layer 0 is the output layer (receives observations)
    - Layer N is the input layer (receives predictors)
    """

    def __init__(
        self,
        coupling_fn: Callable = linear,
    ):
        """Initialize a VectorizedDeepNetwork.

        Parameters
        ----------
        coupling_fn :
            Coupling function applied between layers. Default is linear (identity).
            This function is applied to parent means before the weighted sum to
            predict child means. Use e.g. jnp.tanh for nonlinear coupling.
        """
        self.coupling_fn = coupling_fn
        self.coupling_fn_grad = jax.grad(lambda x: coupling_fn(x))
        self.layer_sizes: list[int] = []
        self.tonic_volatilities: list[float] = []
        self.state: Optional[NetworkState] = None
        self._propagation_fn: Optional[Callable] = None
        self._prediction_fn: Optional[Callable] = None

    def add_nodes(
        self,
        n_nodes: int,
        tonic_volatility: float = -4.0,
    ) -> "VectorizedDeepNetwork":
        """Add input/output layer nodes.

        This is typically used to add the first (output) layer.

        Parameters
        ----------
        n_nodes :
            Number of nodes in the layer.
        tonic_volatility :
            Tonic volatility for the value level (log scale).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        self.layer_sizes.append(n_nodes)
        self.tonic_volatilities.append(tonic_volatility)

        return self

    def add_layer(
        self,
        size: int,
        tonic_volatility: float = -4.0,
    ) -> "VectorizedDeepNetwork":
        """Add a fully-connected hidden layer.

        Parameters
        ----------
        size :
            Number of nodes in the layer.
        tonic_volatility :
            Tonic volatility for the value level (log scale).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        self.layer_sizes.append(size)
        self.tonic_volatilities.append(tonic_volatility)
        return self

    def add_layer_stack(
        self,
        layer_sizes: list[int],
        tonic_volatility: float = -4.0,
    ) -> "VectorizedDeepNetwork":
        """Add multiple hidden layers at once.

        Parameters
        ----------
        layer_sizes :
            List of layer sizes.
        tonic_volatility :
            Tonic volatility for all layers.

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        for size in layer_sizes:
            self.add_layer(size=size, tonic_volatility=tonic_volatility)
        return self

    def _reset_layer_states(self) -> None:
        """Reset all layer states to defaults while preserving learned weights.

        This resets means to zero, precisions to one, and errors to zero for all layers.
        Weight matrices and parameters are kept unchanged. Used between fit() calls in
        epoch-based training to prevent precision accumulation and hidden state
        carryover.
        """
        fresh_layers = tuple(LayerState.create(size) for size in self.layer_sizes)
        assert self.state is not None
        self.state = NetworkState(
            layers=fresh_layers,
            weights=self.state.weights,
            params=self.state.params,
            time_step=self.state.time_step,
        )

    def _init_state(self, key: jax.random.PRNGKey) -> NetworkState:
        """Initialize network state with random weights.

        Parameters
        ----------
        key :
            JAX random key for weight initialization.

        Returns
        -------
        NetworkState
            Initialized network state.
        """
        init_key = key

        layers = []
        weights = []
        params = []

        for i, size in enumerate(self.layer_sizes):
            # Create layer state
            layers.append(LayerState.create(size))

            # Create layer parameters
            params.append(
                LayerParams.create(
                    n_nodes=size,
                    tonic_volatility=self.tonic_volatilities[i],
                    tonic_volatility_vol=-4.0,
                    volatility_coupling=1.0,
                )
            )

            # Create weights connecting to next layer (if not first layer)
            # weights[i] connects layer[i] to layer[i+1]
            if i > 0:
                prev_size = self.layer_sizes[i - 1]
                # Xavier initialization
                scale = jnp.sqrt(2.0 / (prev_size + size))
                init_key, subkey = jax.random.split(init_key)
                w = jax.random.normal(subkey, (prev_size, size)) * scale
                weights.append(w)

        return NetworkState(
            layers=tuple(layers),
            weights=tuple(weights),
            params=tuple(params),
            time_step=1.0,
        )

    def _create_propagation_fn(self, lr: float, T: int = 1):
        """Create the jitted propagation function.

        Parameters
        ----------
        lr :
            Learning rate for weight updates.
        T :
            Number of inference iterations before weight update. T=1 is the original
            single-pass behavior. T>1 enables PC-style iterative inference.

        Returns
        -------
        Callable
            JIT-compiled propagation function.
        """
        coupling_fn = self.coupling_fn
        coupling_fn_grad = self.coupling_fn_grad

        def propagation_step(state: NetworkState, inputs):
            """Single propagation step through the network."""
            x, y = inputs
            layers = list(state.layers)
            weights = list(state.weights)
            params = state.params

            n_layers = len(layers)

            # 1. Set predictors (top layer = input)
            # Must set both expected_mean AND mean for proper learning
            layers[-1] = layers[-1]._replace(expected_mean=x, mean=x)

            # 2. Set observations (bottom layer = output)
            layers[0] = layers[0]._replace(
                mean=y,
                observed=jnp.ones_like(y, dtype=jnp.int32),
            )

            # ========== INFERENCE PHASE (T iterations) ==========
            # Let prediction errors propagate through the network
            # by iteratively updating node activities (posteriors)
            for _t in range(T):
                # 3. Prediction: top-down (using current parent means)
                for i in range(n_layers - 1, 0, -1):
                    layers[i - 1] = vectorized_layer_prediction(
                        child_state=layers[i - 1],
                        parent_state=layers[i],
                        weights=weights[i - 1],
                        params=params[i - 1],
                        time_step=state.time_step,
                        coupling_fn=coupling_fn,
                    )

                # 4. Prediction errors: bottom-up
                for i in range(n_layers - 1):
                    n_parents = weights[i].shape[1]
                    layers[i] = vectorized_layer_prediction_error(
                        state=layers[i],
                        n_parents=n_parents,
                    )

                # 5. Posterior updates: bottom-up (skip input layer = layers[-1])
                for i in range(1, n_layers - 1):
                    # n_value_parents = number of parent nodes (next layer up)
                    n_vp = weights[i].shape[1] if i < n_layers - 1 else 1
                    layers[i] = vectorized_layer_posterior_update(
                        parent_state=layers[i],
                        child_state=layers[i - 1],
                        weights=weights[i - 1],
                        params=params[i],
                        coupling_fn_grad=coupling_fn_grad,
                        n_value_parents=n_vp,
                    )

            # ========== LEARNING PHASE (after inference converges) ==========
            # Update weights once using converged activities
            for i in range(1, n_layers):
                weights[i - 1] = vectorized_weight_update(
                    parent_state=layers[i],
                    child_state=layers[i - 1],
                    weights=weights[i - 1],
                    coupling_fn=coupling_fn,
                    lr=lr,
                )

            new_state = NetworkState(
                layers=tuple(layers),
                weights=tuple(weights),
                params=params,
                time_step=state.time_step,
            )

            # Return output prediction for monitoring
            output_pred = layers[0].expected_mean

            return new_state, (new_state, output_pred)

        return jax.jit(propagation_step)

    def _create_prediction_fn(self):
        """Create the jitted prediction function (forward pass only).

        Returns
        -------
        Callable
            JIT-compiled prediction function.
        """
        coupling_fn = self.coupling_fn

        def prediction_step(state: NetworkState, x):
            """Forward prediction without learning."""
            layers = list(state.layers)
            params = state.params

            n_layers = len(layers)

            # Set input
            layers[-1] = layers[-1]._replace(expected_mean=x)

            # Top-down prediction
            for i in range(n_layers - 1, 0, -1):
                layers[i - 1] = vectorized_layer_prediction(
                    child_state=layers[i - 1],
                    parent_state=layers[i],
                    weights=state.weights[i - 1],
                    params=params[i - 1],
                    time_step=state.time_step,
                    coupling_fn=coupling_fn,
                )

            # Return output layer expected mean
            return layers[0].expected_mean

        return jax.jit(prediction_step)

    def fit(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        lr: float = 0.2,
        T: int = 1,
        key: Optional[jax.random.PRNGKey] = None,
        reset_state: bool = True,
    ) -> "VectorizedDeepNetwork":
        """Fit network to data.

        Parameters
        ----------
        x :
            Input data, shape (n_samples, n_input_features).
        y :
            Target data, shape (n_samples, n_output_features).
        lr :
            Learning rate for weight updates.
        T :
            Number of inference iterations per sample before weight update.
            T=1 is single-pass (original behavior).
            T>1 enables PC-style iterative inference where errors propagate
            through the network before weights are updated.
        key :
            JAX random key for initialization. If None, uses PRNGKey(0).
        reset_state :
            If True (default), reset layer states (means, precisions) to defaults before
            fitting, while preserving learned weights. This prevents precision
            accumulation and hidden state carryover between fit() calls, enabling stable
            epoch-based training. Set to False to preserve full state between calls.

        Returns
        -------
        VectorizedDeepNetwork
            Self with updated state.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        # Initialize state if needed
        if self.state is None:
            self.state = self._init_state(key)
        elif reset_state:
            self._reset_layer_states()

        # Create propagation function (recreate if T changed)
        self._propagation_fn = self._create_propagation_fn(lr, T)

        # Convert to JAX arrays
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Run scan over data
        self.state, (self.trajectories, self.predictions) = jax.lax.scan(
            self._propagation_fn, self.state, (x, y)
        )

        return self

    def predict(
        self,
        x: Union[np.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Forward pass without learning.

        Parameters
        ----------
        x :
            Input data, shape (n_samples, n_input_features) or (n_input_features,).

        Returns
        -------
        jnp.ndarray
            Predictions, shape (n_samples, n_output_features) or (n_output_features,).
        """
        if self.state is None:
            raise ValueError("Network must be fit before predicting.")

        if self._prediction_fn is None:
            self._prediction_fn = self._create_prediction_fn()

        x = jnp.asarray(x)
        prediction_fn = self._prediction_fn

        # Handle single sample vs batch
        if x.ndim == 1:
            return prediction_fn(self.state, x)
        else:
            # Vectorize over samples
            return jax.vmap(lambda xi: prediction_fn(self.state, xi))(x)

    def get_layer_sizes(self) -> list[int]:
        """Get the size of each layer.

        Returns
        -------
        list[int]
            List of layer sizes.
        """
        return self.layer_sizes.copy()

    def get_weights(self) -> tuple:
        """Get the current weight matrices.

        Returns
        -------
        tuple
            Tuple of weight matrices.
        """
        if self.state is None:
            raise ValueError("Network must be initialized first.")
        return self.state.weights

    def reset(
        self, key: Optional[jax.random.PRNGKey] = None
    ) -> "VectorizedDeepNetwork":
        """Reset the network state.

        Parameters
        ----------
        key :
            JAX random key for reinitialization.

        Returns
        -------
        VectorizedDeepNetwork
            Self with reset state.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        self.state = self._init_state(key)
        self._propagation_fn = None
        self._prediction_fn = None
        return self

    @property
    def n_layers(self) -> int:
        """Number of layers in the network."""
        return len(self.layer_sizes)

    @property
    def n_nodes(self) -> int:
        """Total number of nodes in the network."""
        return sum(self.layer_sizes)

    def __repr__(self) -> str:
        """Print string representation."""
        return f"VectorizedDeepNetwork(nodes={self.n_nodes}, layers={self.layer_sizes})"
