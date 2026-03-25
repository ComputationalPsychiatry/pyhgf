# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized deep predictive coding network.

This module provides a vectorized implementation of deep HGF networks
that uses layer-wise matrix operations instead of per-node updates.
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


class VectorizedDeepNetwork:
    """Deep predictive coding network with vectorized operations.

    This class implements a deep hierarchical Gaussian filter using
    layer-wise vectorized operations for efficient scaling to large networks.

    Unlike the standard DeepNetwork which uses per-node updates with
    Python loops, this implementation uses JAX matrix operations
    to update all nodes in a layer simultaneously.

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
    The network uses volatile nodes internally, which have two levels:
    - Value level (external): represents the node's belief about its value
    - Volatility level (internal): represents uncertainty about the value level

    Layer indexing follows the convention:
    - Layer 0 is the output layer (receives observations)
    - Layer N is the input layer (receives predictors)
    """

    def __init__(
        self,
        coupling_fn: Callable = jnp.tanh,
    ):
        """Initialize a VectorizedDeepNetwork.

        Parameters
        ----------
        coupling_fn :
            Coupling function applied between layers. Default is tanh.
            This function is applied to parent means before the weighted
            sum to predict child means.
        """
        self.coupling_fn = coupling_fn
        self.coupling_fn_grad = jax.grad(lambda x: coupling_fn(x))
        self.layer_sizes: list[int] = []
        self.tonic_volatilities: list[float] = []
        self.tonic_volatilities_vol: list[float] = []
        self.volatility_couplings: list[float] = []
        self.use_biases: list[bool] = []
        self.coupling_fns: list[Callable] = []  # per-layer coupling functions
        self.state: Optional[NetworkState] = None
        self._propagation_fn = None
        self._prediction_fn = None

    def add_nodes(
        self,
        n_nodes: int,
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 0.0,
        use_bias: bool = False,
        coupling_fn: Optional[Callable] = None,
    ) -> "VectorizedDeepNetwork":
        """Add input/output layer nodes.

        This is typically used to add the first (output) layer.

        Parameters
        ----------
        n_nodes :
            Number of nodes in the layer.
        tonic_volatility :
            Tonic volatility for the value level (log scale).
        tonic_volatility_vol :
            Tonic volatility for the volatility level (log scale).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        self.layer_sizes.append(n_nodes)
        self.tonic_volatilities.append(tonic_volatility)
        self.tonic_volatilities_vol.append(tonic_volatility_vol)
        self.volatility_couplings.append(volatility_coupling)
        self.use_biases.append(use_bias)
        self.coupling_fns.append(coupling_fn if coupling_fn is not None else self.coupling_fn)
        return self

    def add_layer(
        self,
        size: int,
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 0.0,
        use_bias: bool = False,
        coupling_fn: Optional[Callable] = None,
    ) -> "VectorizedDeepNetwork":
        """Add a fully-connected hidden layer.

        Parameters
        ----------
        size :
            Number of nodes in the layer.
        tonic_volatility :
            Tonic volatility for the value level (log scale).
        tonic_volatility_vol :
            Tonic volatility for the volatility level (log scale).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        self.layer_sizes.append(size)
        self.tonic_volatilities.append(tonic_volatility)
        self.tonic_volatilities_vol.append(tonic_volatility_vol)
        self.volatility_couplings.append(volatility_coupling)
        self.use_biases.append(use_bias)
        self.coupling_fns.append(coupling_fn if coupling_fn is not None else self.coupling_fn)
        return self

    def add_layer_stack(
        self,
        layer_sizes: list[int],
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 0.0,
        use_bias: bool = False,
        coupling_fn: Optional[Callable] = None,
    ) -> "VectorizedDeepNetwork":
        """Add multiple hidden layers at once.

        Parameters
        ----------
        layer_sizes :
            List of layer sizes.
        tonic_volatility :
            Tonic volatility for all layers (value level).
        tonic_volatility_vol :
            Tonic volatility for all layers (volatility level).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        for size in layer_sizes:
            self.add_layer(
                size=size,
                tonic_volatility=tonic_volatility,
                tonic_volatility_vol=tonic_volatility_vol,
                volatility_coupling=volatility_coupling,
                use_bias=use_bias,
                coupling_fn=coupling_fn,
            )
        return self

    def _reset_layer_states(self) -> None:
        """Reset all layer states to defaults while preserving learned weights.

        This resets means to zero, precisions to one, and errors to zero
        for all layers. Weight matrices and parameters are kept unchanged.
        Used between fit() calls in epoch-based training to prevent
        precision accumulation and hidden state carryover.
        """
        fresh_layers = tuple(
            LayerState.create(size) for size in self.layer_sizes
        )
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
                    tonic_volatility_vol=self.tonic_volatilities_vol[i],
                    volatility_coupling=self.volatility_couplings[i],
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

    def _create_propagation_fn(
        self,
        lr: float,
        T: int = 1,
        sqrt_normalization: bool = False,
        input_precision: float = 1.0,
        output_precision: float = 1.0,
        reset_hidden: bool = False,
        update_input_layer: bool = False,
        normalize_vol_pe: bool = True,
    ):
        """Create the jitted propagation function.

        Parameters
        ----------
        lr :
            Learning rate for weight updates.
        T :
            Number of inference iterations before weight update.
            T=1 is the original single-pass behavior.
            T>1 enables PC-style iterative inference.
        sqrt_normalization :
            If True, normalize value PE by sqrt(n_parents) instead of n_parents.
        input_precision :
            Precision to pin the input layer to each step (1.0 = unpinned).
        output_precision :
            Precision to pin the output layer to each step (1.0 = unpinned).
        reset_hidden :
            If True, reset hidden layer states (means, precisions) to defaults
            at the start of each sample. Prevents precision accumulation across
            unrelated samples (recommended for i.i.d. data like MNIST).
        update_input_layer :
            If True, include the input layer in the posterior update loop,
            allowing it to accumulate a learned volatility state. Default False
            (input layer is a pure data source).
        normalize_vol_pe :
            If True (default), divide fresh_value_pe by n_value_parents when
            computing the volatility PE, matching the upstream HGF derivation.
            If False, use n_value_parents=1 for all layers, allowing the
            volatility level to update more aggressively.

        Returns
        -------
        Callable
            JIT-compiled propagation function.
        """
        # Per-layer coupling functions and their gradients.
        # coupling_fns[i] is applied to layer[i].expected_mean when layer[i]
        # acts as a parent (i.e. when predicting layer[i-1]).
        coupling_fns = self.coupling_fns
        coupling_fn_grads = [jax.grad(lambda x, fn=fn: fn(x)) for fn in coupling_fns]
        use_biases = self.use_biases  # captured for bias updates
        layer_sizes = self.layer_sizes  # captured for per-sample reset

        def propagation_step(state: NetworkState, inputs):
            """Single propagation step through the network."""
            x, y = inputs
            layers = list(state.layers)
            weights = list(state.weights)
            params = list(state.params)

            n_layers = len(layers)

            # Per-sample hidden state reset (layers 1 to n_layers-2)
            # Resets means and precisions to defaults, preserving weights/params.
            # Prevents hidden precision from accumulating across unrelated samples.
            if reset_hidden:
                for i in range(1, n_layers - 1):
                    layers[i] = LayerState.create(layer_sizes[i])

            # 1. Set predictors (top layer = input)
            # Must set both expected_mean AND mean for proper learning
            if input_precision != 1.0:
                layers[-1] = layers[-1]._replace(
                    expected_mean=x, mean=x,
                    precision=jnp.full_like(layers[-1].precision, input_precision),
                    expected_precision=jnp.full_like(layers[-1].expected_precision, input_precision),
                )
            else:
                layers[-1] = layers[-1]._replace(expected_mean=x, mean=x)

            # 2. Set observations (bottom layer = output)
            if output_precision != 1.0:
                layers[0] = layers[0]._replace(
                    mean=y,
                    observed=jnp.ones_like(y, dtype=jnp.int32),
                    precision=jnp.full_like(layers[0].precision, output_precision),
                    expected_precision=jnp.full_like(layers[0].expected_precision, output_precision),
                )
            else:
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
                        coupling_fn=coupling_fns[i],  # parent i's coupling fn
                        add_bias=use_biases[i - 1],
                    )

                # 4. Interleaved PE → posterior (matches standard Network ripple order).
                #
                # Standard Network enforces: a node computes its PE only AFTER its
                # own posterior has been updated; the updated posterior mean is what
                # propagates upward as the prediction error signal.
                #
                # Previous wave-based implementation (all PEs then all posteriors)
                # computed PE(hidden) from the PRE-posterior mean, which is incorrect.
                #
                # Correct order for an N-layer network:
                #   PE(layer 0)                  ← mean is pinned to y, always correct
                #   posterior(layer 1)           ← uses layer 0 PE
                #   PE(layer 1)                  ← uses UPDATED layer 1 mean
                #   posterior(layer 2)           ← uses layer 1 updated PE
                #   ...
                #   [optional] posterior(input)  ← controlled by update_input_layer

                # Step 4a: PE for output layer (mean = y, observation-pinned)
                layers[0] = vectorized_layer_prediction_error(
                    state=layers[0],
                    n_parents=weights[0].shape[1],
                    sqrt_normalization=sqrt_normalization,
                )

                # Step 4b: per hidden layer — posterior then PE (interleaved)
                for i in range(1, n_layers - 1):
                    n_vp = weights[i].shape[1] if normalize_vol_pe else 1
                    layers[i] = vectorized_layer_posterior_update(
                        parent_state=layers[i],
                        child_state=layers[i - 1],
                        weights=weights[i - 1],
                        params=params[i],
                        coupling_fn_grad=coupling_fn_grads[i],  # parent i's grad
                        n_value_parents=n_vp,
                    )
                    # Recompute PE using updated posterior mean so the layer
                    # above receives the correct (post-posterior) error signal.
                    layers[i] = vectorized_layer_prediction_error(
                        state=layers[i],
                        n_parents=weights[i].shape[1],
                        sqrt_normalization=sqrt_normalization,
                    )

                # Step 4c: optional posterior update for input layer
                if update_input_layer:
                    n_vp = 1  # input layer has no parents above it
                    layers[-1] = vectorized_layer_posterior_update(
                        parent_state=layers[-1],
                        child_state=layers[-2],
                        weights=weights[-1],
                        params=params[-1],
                        coupling_fn_grad=coupling_fn_grads[-1],  # input layer's grad
                        n_value_parents=n_vp,
                    )

            # ========== LEARNING PHASE (after inference converges) ==========
            # Update weights once using converged activities
            for i in range(1, n_layers):
                weights[i - 1] = vectorized_weight_update(
                    parent_state=layers[i],
                    child_state=layers[i - 1],
                    weights=weights[i - 1],
                    coupling_fn=coupling_fns[i],  # parent i's coupling fn
                    lr=lr,
                )
                # Bias update: delta_b = lr * pe (bias = weight with constant input 1)
                # Only update layers where use_bias=True
                if use_biases[i - 1]:
                    pe = layers[i - 1].mean - layers[i - 1].expected_mean
                    new_bias = params[i - 1].bias + lr * pe
                    params[i - 1] = params[i - 1]._replace(bias=new_bias)

            new_state = NetworkState(
                layers=tuple(layers),
                weights=tuple(weights),
                params=tuple(params),
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
        coupling_fns = self.coupling_fns
        use_biases = self.use_biases

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
                    coupling_fn=coupling_fns[i],  # parent i's coupling fn
                    add_bias=use_biases[i - 1],
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
        pe_normalization: str = "n_parents",
        input_precision: float = 1.0,
        output_precision: float = 1.0,
        update_input_layer: bool = False,
        normalize_vol_pe: bool = True,
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
            If True (default), reset layer states (means, precisions) to
            defaults before fitting, while preserving learned weights.
            This prevents precision accumulation and hidden state carryover
            between fit() calls, enabling stable epoch-based training.
            Set to False to preserve full state between calls.
        pe_normalization :
            How to normalize the value prediction error. One of:
            ``"n_parents"`` (default) — divide by n_parents (upstream baseline).
            ``"sqrt_n_parents"`` — divide by sqrt(n_parents), which can improve
            gradient flow in wide networks.
        input_precision :
            Precision to pin the input layer to at each step. Default 1.0
            (unpinned). High values (e.g. 1000.0) fix the input layer beliefs
            to the data, concentrating weight updates in lower layers.
        output_precision :
            Precision to pin the output layer to at each step. Default 1.0
            (unpinned). High values (e.g. 100.0) fix the output layer beliefs,
            concentrating weight updates in upper layers.
        update_input_layer :
            If True, include the input layer in the posterior update loop.
            Default False.
        normalize_vol_pe :
            If True (default), normalize fresh_value_pe by n_value_parents.
            If False, use n_value_parents=1 (more aggressive volatility updates).

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

        # Create propagation function
        sqrt_norm = pe_normalization == "sqrt_n_parents"
        self._propagation_fn = self._create_propagation_fn(
            lr, T,
            sqrt_normalization=sqrt_norm,
            input_precision=input_precision,
            output_precision=output_precision,
            reset_hidden=reset_state,
            update_input_layer=update_input_layer,
            normalize_vol_pe=normalize_vol_pe,
        )

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

        # Handle single sample vs batch
        if x.ndim == 1:
            return self._prediction_fn(self.state, x)
        else:
            # Vectorize over samples
            return jax.vmap(lambda xi: self._prediction_fn(self.state, xi))(x)

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

    def reset(self, key: Optional[jax.random.PRNGKey] = None) -> "VectorizedDeepNetwork":
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
