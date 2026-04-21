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
from pyhgf.updates.vectorized.binary import vectorized_binary_prediction
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


class DeepNetwork:
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
        coupling_fn: Callable = lambda x: x,
        update_type: str = "eHGF",
    ):
        """Initialize a VectorizedDeepNetwork.

        Parameters
        ----------
        coupling_fn :
            Coupling function applied between layers. Default is linear (identity),
            matching the Rust backend and the Network class. This function is
            applied to parent means before the weighted sum to predict child means.
        update_type :
            The type of volatility-level posterior update. Can be ``"eHGF"``
            (default), ``"standard"`` or ``"unbounded"``. Matches the Network
            class and Rust backend.
        """
        self.coupling_fn = coupling_fn
        self.update_type = update_type
        self.layer_sizes: list[int] = []
        self.layer_kinds: list[str] = []
        self.tonic_volatilities: list[float] = []
        self.tonic_volatilities_vol: list[float] = []
        self.volatility_couplings: list[float] = []
        self.add_constant_inputs: list[bool] = []
        self.fully_connected: list[bool] = []
        self.coupling_fns: list[Callable] = []  # per-layer coupling functions
        self.volatility_parents: list[bool] = []
        self.state: Optional[NetworkState] = None
        self.trajectories: Optional[NetworkState] = None
        self.predictions: Optional[jnp.ndarray] = None
        self._propagation_fn: Optional[Callable] = None
        self._propagation_lr: Optional[Union[float, str]] = None
        self._propagation_optimizer: Optional[str] = None
        self._record_trajectories: bool = False
        self._prediction_fn: Optional[Callable] = None

    def add_layer(
        self,
        size: int,
        kind: str = "volatile",
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 1.0,
        add_constant_input: bool = True,
        fully_connected: bool = True,
        coupling_fn: Optional[Callable] = None,
        volatility_parent: bool = True,
    ) -> "DeepNetwork":
        """Add a layer of nodes.

        Parameters
        ----------
        size :
            Number of nodes in the layer.
        kind :
            Type of nodes in this layer. ``"volatile"`` (default) uses volatile nodes
            with value and volatility levels. ``"binary"`` uses binary state nodes.
        tonic_volatility :
            Tonic volatility for the value level (log scale). Only used for
            ``"volatile"`` layers.
        tonic_volatility_vol :
            Tonic volatility for the volatility level (log scale). Only used for
            ``"volatile"`` layers.
        volatility_coupling :
            Coupling strength between the value and volatility levels. Only used for
            ``"volatile"`` layers.
        add_constant_input :
            If `True`, add a bias term to the layer's predictions.
        fully_connected :
            If `True` (default), each node in this layer connects to every node in the
            child layer (dense weight matrix).  If False, nodes connect one-to-one with
            the child layer (diagonal weight matrix). This requires both layers to have
            the same size and ``add_constant_input=False``.
        coupling_fn :
            Coupling function for this layer. If None, uses the network-level coupling
            function.
        volatility_parent :
            If True (default), this layer has an implied internal volatility parent:
            mean_vol and precision_vol are predicted and updated each step. If False,
            the volatility level is frozen and only tonic_volatility determines the
            expected precision for the value level.

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If *kind* is not a recognised node type.
        ValueError
            If *fully_connected* is False with ``add_constant_input=True``.
        ValueError
            If *fully_connected* is False and this layer's size differs from
            the preceding child layer (for non-binary child layers).
        """
        # Normalise Rust-style kind names to short form
        _kind_aliases = {
            "volatile-state": "volatile",
            "binary-state": "binary",
        }
        kind = _kind_aliases.get(kind, kind)

        valid_kinds = {"volatile", "binary"}
        if kind not in valid_kinds:
            raise ValueError(
                f"Invalid layer kind '{kind}'. Choose from {sorted(valid_kinds)}."
            )

        if not fully_connected:
            if add_constant_input:
                raise ValueError(
                    "One-to-one layers (fully_connected=False) cannot use "
                    "add_constant_input=True."
                )
            if self.layer_sizes and self.layer_sizes[-1] != size:
                raise ValueError(
                    f"One-to-one layers require the same size as the child "
                    f"layer ({self.layer_sizes[-1]}), got {size}."
                )

        self.layer_sizes.append(size)
        self.layer_kinds.append(kind)
        self.tonic_volatilities.append(tonic_volatility)
        self.tonic_volatilities_vol.append(tonic_volatility_vol)
        self.volatility_couplings.append(volatility_coupling)
        self.add_constant_inputs.append(add_constant_input)
        self.fully_connected.append(fully_connected)
        self.coupling_fns.append(
            coupling_fn if coupling_fn is not None else self.coupling_fn
        )
        self.volatility_parents.append(volatility_parent)
        return self

    def add_layer_stack(
        self,
        layer_sizes: list[int],
        kind: str = "volatile",
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 1.0,
        add_constant_input: bool = True,
        fully_connected: bool = True,
        coupling_fn: Optional[Callable] = None,
    ) -> "DeepNetwork":
        """Add multiple hidden layers at once.

        Parameters
        ----------
        layer_sizes :
            List of layer sizes.
        kind :
            Type of nodes for all layers (``"volatile"`` or ``"binary"``).
        tonic_volatility :
            Tonic volatility for all layers (value level, log scale).
        tonic_volatility_vol :
            Tonic volatility for all layers (volatility level, log scale).
        volatility_coupling :
            Coupling strength between the value and volatility levels for all layers.
        add_constant_input :
            If True, add a bias term to each layer's predictions.
        fully_connected :
            If True (default), layers are fully connected. If False, layers use
            one-to-one connections (requires equal sizes).
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
                kind=kind,
                tonic_volatility=tonic_volatility,
                tonic_volatility_vol=tonic_volatility_vol,
                volatility_coupling=volatility_coupling,
                add_constant_input=add_constant_input,
                fully_connected=fully_connected,
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
                if self.fully_connected[i]:
                    weights.append(jnp.ones((prev_size, n_parent_cols)))
                else:
                    weights.append(jnp.eye(prev_size, n_parent_cols))

        # Adam moment buffers (zeros, same shapes as weights)
        adam_m = tuple(jnp.zeros_like(w) for w in weights)
        adam_v = tuple(jnp.zeros_like(w) for w in weights)

        return NetworkState(
            layers=tuple(layers),
            weights=tuple(weights),
            params=tuple(params),
            time_step=1.0,
            adam_m=adam_m,
            adam_v=adam_v,
            adam_t=0,
        )

    def weight_initialisation(
        self,
        strategy: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "DeepNetwork":
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
            If the strategy name is not recognised.
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
            self.state = self._init_state()

        _init_fns: dict[str, Callable[..., np.ndarray]] = {
            "xavier": xavier_init,
            "he": he_init,
            "orthogonal": orthogonal_init,
            "sparse": sparse_init,
        }
        init_fn = _init_fns[strategy]

        new_weights = list(self.state.weights)
        for i, w in enumerate(new_weights):
            # Binary child layers always use 1.0 weights — skip initialisation.
            if self.layer_kinds[i] == "binary":
                continue
            n_children, n_parents = w.shape  # (prev_size, size)
            flat = init_fn(n_parents, n_children, seed=seed, **kwargs)
            new_weights[i] = jnp.array(flat.reshape(w.shape))

        self.state = NetworkState(
            layers=self.state.layers,
            weights=tuple(new_weights),
            params=self.state.params,
            time_step=self.state.time_step,
            adam_m=self.state.adam_m,
            adam_v=self.state.adam_v,
            adam_t=self.state.adam_t,
        )
        return self

    def _create_propagation_fn(
        self,
        lr: Union[float, str],
        optimizer: Optional[str] = None,
        params: Optional[dict] = None,
        record_trajectories: bool = False,
    ):
        """Create the jitted propagation function.

        Parameters
        ----------
        lr :
            Learning rate for weight updates.
        optimizer :
            Optimizer name.  ``"adam"`` enables Adam filtering of weight
            gradients.  *None* uses plain gradient updates.
        params :
            Hyper-parameters for the optimizer (e.g. ``beta1``, ``beta2``,
            ``epsilon``, and ``lr`` for Adam).  See :meth:`fit` for defaults.
        record_trajectories :
            If True, the scan output includes the full ``NetworkState`` at
            every time step (useful for inspection but significantly slower).
            If False (default), only predictions are accumulated.

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
        layer_kinds = self.layer_kinds
        update_type = self.update_type
        volatility_parents = self.volatility_parents

        # Build Adam parameters tuple (or None)
        # Format: (beta1, beta2, epsilon, adam_lr_override)
        if optimizer == "adam":
            p = params or {}
            adam_params: Optional[tuple[float, float, float, Optional[float]]] = (
                p.get("beta1", 0.9),
                p.get("beta2", 0.999),
                p.get("epsilon", 1e-8),
                p.get("lr", 1e-3),
            )
        else:
            adam_params = None

        if record_trajectories:

            def _step(state: NetworkState, inputs):
                new_state, output_pred = propagation_step(
                    state,
                    inputs,
                    coupling_fns,
                    coupling_fn_grads,
                    add_constant_inputs,
                    lr,
                    layer_kinds,
                    adam_params,
                    update_type,
                    volatility_parents,
                )
                return new_state, (new_state, output_pred)

        else:

            def _step(state: NetworkState, inputs):
                return propagation_step(
                    state,
                    inputs,
                    coupling_fns,
                    coupling_fn_grads,
                    add_constant_inputs,
                    lr,
                    layer_kinds,
                    adam_params,
                    update_type,
                    volatility_parents,
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
        layer_kinds = self.layer_kinds
        volatility_parents = self.volatility_parents

        def prediction_step(state: NetworkState, x):
            """Forward prediction without learning."""
            layers = list(state.layers)
            params = state.params

            n_layers = len(layers)

            # Set input
            layers[-1] = layers[-1]._replace(expected_mean=x)

            # Top-down prediction
            for i in range(n_layers - 1, 0, -1):
                if layer_kinds[i - 1] == "binary":
                    layers[i - 1] = vectorized_binary_prediction(
                        child_state=layers[i - 1],
                        parent_state=layers[i],
                        weights=state.weights[i - 1],
                        coupling_fn=coupling_fns[i],
                        parent_has_constant=add_constant_inputs[i],
                    )
                else:
                    layers[i - 1] = vectorized_layer_prediction(
                        child_state=layers[i - 1],
                        parent_state=layers[i],
                        weights=state.weights[i - 1],
                        params=params[i - 1],
                        time_step=state.time_step,
                        coupling_fn=coupling_fns[i],
                        parent_has_constant=add_constant_inputs[i],
                        has_volatility_parent=volatility_parents[i - 1],
                    )

            # Return output layer expected mean
            return layers[0].expected_mean

        return jax.jit(prediction_step)

    def fit(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        lr: Union[float, str] = 0.2,
        optimizer: Optional[str] = "adam",
        params: Optional[dict] = None,
        record_trajectories: bool = False,
    ) -> "DeepNetwork":
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
        optimizer :
            Optimizer name.  ``"adam"`` (default) filters weight gradients through the
            Adam algorithm (Kingma & Ba, 2015).  *None* uses plain gradient or
            Kalman-gain updates.
        params :
            Dictionary of optimizer hyper-parameters.  For Adam: ``beta1`` (default
            0.9), ``beta2`` (default 0.999), ``epsilon`` (default 1e-8), and ``lr``
            (default 1e-3, the Adam step size).
        record_trajectories :
            If True, record the full ``NetworkState`` at every time step (accessible
            via ``self.trajectories``).  This is useful for inspection but significantly
            increases memory usage and slows training. Default is False.

        Returns
        -------
        VectorizedDeepNetwork
            Self with updated state.

        Raises
        ------
        ValueError
            If *optimizer* is not ``None`` or ``"adam"``.
        """
        if optimizer is not None and optimizer != "adam":
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Supported: 'adam' or None."
            )

        # Initialize state if needed
        if self.state is None:
            self.state = self._init_state()

        # Recreate (and retrace) the propagation fn when settings change
        needs_retrace = (
            self._propagation_fn is None
            or self._propagation_lr != lr
            or self._propagation_optimizer != optimizer
            or self._record_trajectories != record_trajectories
        )
        if needs_retrace:
            self._propagation_fn = self._create_propagation_fn(
                lr, optimizer, params, record_trajectories
            )
            self._propagation_lr = lr
            self._propagation_optimizer = optimizer
            self._record_trajectories = record_trajectories

        # Convert to JAX arrays
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Run scan over data
        assert self._propagation_fn is not None
        if record_trajectories:
            self.state, (self.trajectories, self.predictions) = jax.lax.scan(
                self._propagation_fn, self.state, (x, y)
            )
        else:
            self.trajectories = None
            self.state, self.predictions = jax.lax.scan(
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

    def reset(self) -> "DeepNetwork":
        """Reset the network state.

        Returns
        -------
        VectorizedDeepNetwork
            Self with reset state.
        """
        self.state = self._init_state()
        self._propagation_fn = None
        self._propagation_lr = None
        self._propagation_optimizer = None
        self._record_trajectories = False
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
