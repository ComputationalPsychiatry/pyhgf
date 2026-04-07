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

from pyhgf.typing import LayerParams, LayerState, NetworkState
from pyhgf.updates.vectorized.volatile import (
    vectorized_layer_prediction,
)
from pyhgf.utils.vectorized_belief_propagation import propagation_step
from pyhgf.utils.weight_initialisation import (
    he_init,
    orthogonal_init,
    sparse_init,
    xavier_init,
)


class VectorizedDeepNetwork:
    """Deep predictive coding network with vectorized operations.

    This class implements a deep hierarchical Gaussian filter using layer-wise
    vectorized operations for efficient scaling to large networks.

    Unlike the standard DeepNetwork which uses per-node updates with Python loops, this
    implementation uses JAX matrix operations to update all nodes in a layer
    simultaneously.

    Examples
    --------
    >>> # Build a network with method chaining
    >>> net = (
    ...     VectorizedDeepNetwork()
    ...     .add_layer(size=10)  # Output layer
    ...     .add_layer(size=8)   # Hidden layer 1
    ...     .add_layer(size=6)   # Hidden layer 2
    ...     .add_layer(size=4)   # Input layer
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
            Coupling function applied between layers. Default is tanh. This function is
            applied to parent means before the weighted sum to predict child means.
        """
        self.coupling_fn = coupling_fn
        self.layer_sizes: list[int] = []
        self.tonic_volatilities: list[float] = []
        self.tonic_volatilities_vol: list[float] = []
        self.volatility_couplings: list[float] = []
        self.add_constant_inputs: list[bool] = []
        self.coupling_fns: list[Callable] = []  # per-layer coupling functions
        self.state: Optional[NetworkState] = None
        self.trajectories: Optional[NetworkState] = None
        self.predictions: Optional[jnp.ndarray] = None
        self._propagation_fn: Optional[Callable] = None
        self._prediction_fn: Optional[Callable] = None

    def add_layer(
        self,
        size: int,
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 1.0,
        add_constant_input: bool = False,
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
        volatility_coupling :
            Coupling strength between the value and volatility levels.
        add_constant_input :
            If True, add a bias term to the layer's predictions.
        coupling_fn :
            Coupling function for this layer. If None, uses the network-level coupling
            function.

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        self.layer_sizes.append(size)
        self.tonic_volatilities.append(tonic_volatility)
        self.tonic_volatilities_vol.append(tonic_volatility_vol)
        self.volatility_couplings.append(volatility_coupling)
        self.add_constant_inputs.append(add_constant_input)
        self.coupling_fns.append(
            coupling_fn if coupling_fn is not None else self.coupling_fn
        )
        return self

    def add_layer_stack(
        self,
        layer_sizes: list[int],
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 1.0,
        add_constant_input: bool = False,
        coupling_fn: Optional[Callable] = None,
    ) -> "VectorizedDeepNetwork":
        """Add multiple hidden layers at once.

        Parameters
        ----------
        layer_sizes :
            List of layer sizes.
        tonic_volatility :
            Tonic volatility for all layers (value level, log scale).
        tonic_volatility_vol :
            Tonic volatility for all layers (volatility level, log scale).
        volatility_coupling :
            Coupling strength between the value and volatility levels for all layers.
        add_constant_input :
            If True, add a bias term to each layer's predictions.
        coupling_fn :
            Coupling function for all layers. If None, uses the network-level coupling
            function.

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
                add_constant_input=add_constant_input,
                coupling_fn=coupling_fn,
            )
        return self

    def _init_state(self) -> NetworkState:
        """Initialize network state with uniform weights.

        All inter-layer weights are set to ``1.0``, matching the DeepNetwork and
        Rust backends.  Use :meth:`weight_initialisation` after construction to
        apply Xavier, He, orthogonal, or sparse initialisation.

        Returns
        -------
        NetworkState
            Initialized network state.
        """
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
            # weights[i-1] connects layer[i-1] (child) to layer[i] (parent)
            # If the parent layer has add_constant_input=True, an extra column
            # is appended for the bias node (constant mean = 1.0).
            if i > 0:
                prev_size = self.layer_sizes[i - 1]
                n_parent_cols = size + (1 if self.add_constant_inputs[i] else 0)
                weights.append(jnp.ones((prev_size, n_parent_cols)))

        return NetworkState(
            layers=tuple(layers),
            weights=tuple(weights),
            params=tuple(params),
            time_step=1.0,
        )

    def weight_initialisation(
        self,
        strategy: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "VectorizedDeepNetwork":
        """Initialise inter-layer weight matrices.

        Parameters
        ----------
        strategy :
            Initialisation strategy.  One of ``"xavier"``, ``"he"``,
            ``"orthogonal"``, or ``"sparse"``.  If *None*, weights are left
            unchanged (all ``1.0``).
        seed :
            Optional random seed passed to the initialisation function.
        **kwargs
            Extra keyword arguments forwarded to the initialisation function
            (e.g. ``gain`` for orthogonal, ``sparsity`` / ``std`` for sparse).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If the strategy name is not recognised or the state has not been
            initialised yet.
        """
        if strategy is None:
            return self

        valid = {"xavier", "he", "orthogonal", "sparse"}
        if strategy not in valid:
            raise ValueError(
                f"Invalid weight initialisation strategy '{strategy}'. "
                f"Choose from {sorted(valid)}."
            )

        if self.state is None:
            raise ValueError(
                "State must be initialised before calling weight_initialisation. "
                "Call fit() first or initialise the state manually."
            )

        _init_fns: dict[str, Callable[..., np.ndarray]] = {
            "xavier": xavier_init,
            "he": he_init,
            "orthogonal": orthogonal_init,
            "sparse": sparse_init,
        }
        init_fn = _init_fns[strategy]

        new_weights = list(self.state.weights)
        for i, w in enumerate(new_weights):
            n_children, n_parents = w.shape  # (prev_size, size)
            flat = init_fn(n_parents, n_children, seed=seed, **kwargs)
            new_weights[i] = jnp.array(flat.reshape(w.shape))

        self.state = NetworkState(
            layers=self.state.layers,
            weights=tuple(new_weights),
            params=self.state.params,
            time_step=self.state.time_step,
        )
        return self

    def _create_propagation_fn(
        self,
        lr: Union[float, str],
    ):
        """Create the jitted propagation function.

        Parameters
        ----------
        lr :
            Learning rate for weight updates.

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
        add_constant_inputs = self.add_constant_inputs  # captured for bias updates

        def _step(state: NetworkState, inputs):
            return propagation_step(
                state,
                inputs,
                coupling_fns,
                coupling_fn_grads,
                add_constant_inputs,
                lr,
            )

        return jax.jit(_step)

    def _create_prediction_fn(self):
        """Create the jitted prediction function (forward pass only).

        Returns
        -------
        Callable
            JIT-compiled prediction function.
        """
        coupling_fns = self.coupling_fns
        add_constant_inputs = self.add_constant_inputs

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
                    parent_has_constant=add_constant_inputs[i],
                )

            # Return output layer expected mean
            return layers[0].expected_mean

        return jax.jit(prediction_step)

    def fit(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        lr: Union[float, str] = 0.2,
    ) -> "VectorizedDeepNetwork":
        """Fit network to data.

        Parameters
        ----------
        x :
            Input data, shape (n_samples, n_input_features).
        y :
            Target data, shape (n_samples, n_output_features).
        lr :
            Learning rate for weight updates, or ``"dynamic"`` for
            Kalman-gain updates.

        Returns
        -------
        VectorizedDeepNetwork
            Self with updated state.
        """
        # Initialize state if needed
        if self.state is None:
            self.state = self._init_state()

        self._propagation_fn = self._create_propagation_fn(
            lr,
        )

        # Convert to JAX arrays
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Run scan over data
        assert self._propagation_fn is not None
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

        prediction_fn = self._prediction_fn
        x = jnp.asarray(x)

        # Handle single sample vs batch
        if x.ndim == 1:
            return prediction_fn(self.state, x)
        else:
            # Vectorize over samples
            return jax.vmap(lambda xi: prediction_fn(self.state, xi))(x)

    def reset(self) -> "VectorizedDeepNetwork":
        """Reset the network state.

        Returns
        -------
        VectorizedDeepNetwork
            Self with reset state.
        """
        self.state = self._init_state()
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
