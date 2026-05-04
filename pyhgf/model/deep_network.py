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

# Default per-layer parameter values (matching ``LayerParams.create``).
_LAYER_PARAM_DEFAULTS: dict[str, float] = {
    "tonic_volatility": -4.0,
    "tonic_volatility_vol": -4.0,
    "volatility_coupling": 1.0,
    "autoconnection_strength_vol": 1.0,
}

# Names of fields that can be overridden per layer.
_LAYER_STATE_FIELDS: frozenset[str] = frozenset(LayerState._fields)
_LAYER_PARAM_FIELDS: frozenset[str] = frozenset(LayerParams._fields)
_LAYER_OVERRIDE_FIELDS: frozenset[str] = _LAYER_STATE_FIELDS | _LAYER_PARAM_FIELDS


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
        # Per-layer overrides for fields of ``LayerState`` and ``LayerParams``.
        # Populated from the ``**kwargs`` of :meth:`add_layer`.
        self.layer_overrides: list[dict[str, float]] = []
        self.add_constant_inputs: list[bool] = []
        self.fully_connected: list[bool] = []
        self.coupling_fns: list[Callable] = []  # per-layer coupling functions
        self.volatility_parents: list[bool] = []
        self.state: Optional[NetworkState] = None
        self.trajectories: Optional[NetworkState] = None
        self.predictions: Optional[jnp.ndarray] = None
        self._propagation_fn: Optional[Callable] = None
        self._propagation_lr: Optional[Union[float, str]] = None
        self._propagation_learning_kind: Optional[str] = None
        self._record_trajectories: bool = False
        self._propagation_weight_update: bool = True
        self._prediction_fn: Optional[Callable] = None

    def add_layer(
        self,
        size: int,
        kind: str = "volatile",
        add_constant_input: bool = True,
        fully_connected: bool = True,
        coupling_fn: Optional[Callable] = None,
        volatility_parent: bool = True,
        **kwargs,
    ) -> "DeepNetwork":
        """Add a layer of nodes.

        Parameters
        ----------
        size :
            Number of nodes in the layer.
        kind :
            Type of nodes in this layer. ``"volatile"`` (default) uses volatile nodes
            with value and volatility levels. ``"binary"`` uses binary state nodes.
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
            the volatility level is frozen and only ``tonic_volatility`` determines the
            expected precision for the value level.
        **kwargs :
            Per-layer overrides for any field of :class:`pyhgf.typing.LayerState`
            (e.g. ``mean``, ``precision``, ``expected_mean``, ``expected_precision``,
            ``mean_vol``, ``precision_vol``, ...) or :class:`pyhgf.typing.LayerParams`
            (``tonic_volatility``, ``tonic_volatility_vol``, ``volatility_coupling``,
            ``autoconnection_strength_vol``).
            Each value is broadcast to the layer's ``size``. Unknown names raise
            ``ValueError``. Defaults match ``LayerState.create`` and
            ``LayerParams.create``.

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If *kind* is not a recognised node type, if a key in *kwargs* does not match
            a ``LayerState`` or ``LayerParams`` field, if *fully_connected* is False
            with ``add_constant_input=True``, or if *fully_connected* is False and this
            layer's size differs from the preceding child layer.
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

        invalid_keys = [k for k in kwargs if k not in _LAYER_OVERRIDE_FIELDS]
        if invalid_keys:
            raise ValueError(
                f"Unknown layer override(s): {invalid_keys}. "
                f"Valid fields are {sorted(_LAYER_OVERRIDE_FIELDS)}."
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
        self.layer_overrides.append(dict(kwargs))
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
        add_constant_input: bool = True,
        fully_connected: bool = True,
        coupling_fn: Optional[Callable] = None,
        **kwargs,
    ) -> "DeepNetwork":
        """Add multiple hidden layers at once.

        Parameters
        ----------
        layer_sizes :
            List of layer sizes.
        kind :
            Type of nodes for all layers (``"volatile"`` or ``"binary"``).
        add_constant_input :
            If True, add a bias term to each layer's predictions.
        fully_connected :
            If True (default), layers are fully connected. If False, layers use
            one-to-one connections (requires equal sizes).
        coupling_fn :
            Coupling function for all layers. If None, uses the network-level coupling
            function.
        **kwargs :
            Per-layer overrides forwarded to :meth:`add_layer` (any field of
            ``LayerState`` or ``LayerParams``, e.g. ``tonic_volatility``,
            ``precision``).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        for size in layer_sizes:
            self.add_layer(
                size=size,
                kind=kind,
                add_constant_input=add_constant_input,
                fully_connected=fully_connected,
                coupling_fn=coupling_fn,
                **kwargs,
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
            overrides = self.layer_overrides[i]

            # Layer state with per-layer overrides (broadcast scalar -> (n_nodes,))
            state = LayerState.create(size)
            state_overrides = {
                k: jnp.full(size, v)
                for k, v in overrides.items()
                if k in _LAYER_STATE_FIELDS
            }
            if state_overrides:
                state = state._replace(**state_overrides)
            layers.append(state)

            # Layer parameters with defaults overridden per layer
            param_kwargs = dict(_LAYER_PARAM_DEFAULTS)
            for k, v in overrides.items():
                if k in _LAYER_PARAM_FIELDS:
                    param_kwargs[k] = v
            params.append(LayerParams.create(n_nodes=size, **param_kwargs))

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
        learning_kind: str = "precision_weighted",
        params: Optional[dict] = None,
        record_trajectories: bool = False,
        weight_update: bool = True,
    ):
        """Create the jitted propagation function.

        Parameters
        ----------
        lr :
            How the gradient is applied: a non-negative float for direct scaling, or
            ``"adam"`` for the Adam optimiser.
        learning_kind :
            Gradient computation mode: ``"standard"``, ``"precision_weighted"``
            (default), or ``"precision_ratio"``.
        params :
            Hyper-parameters for the Adam optimiser (``beta1``, ``beta2``, ``epsilon``,
            ``lr``). Only used when ``lr="adam"``.
        record_trajectories :
            If True, the scan output includes the full ``NetworkState`` at every time
            step (useful for inspection but significantly slower). If False (default),
            only predictions are accumulated.
        weight_update :
            If ``True`` (default), the learning phase updates the weights at the
            end of each step. If ``False``, weights are frozen — the network
            performs inference only.

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

        # Resolve Adam hyper-parameters when lr="adam".
        if lr == "adam":
            p = params or {}
            adam_params: Optional[tuple[float, float, float, float]] = (
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
                    learning_kind,
                    weight_update,
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
                    learning_kind,
                    weight_update,
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
        lr: Union[float, str] = "adam",
        learning_kind: str = "precision_weighted",
        params: Optional[dict] = None,
        record_trajectories: bool = False,
        weight_update: bool = True,
    ) -> "DeepNetwork":
        """Fit network to data.

        Parameters
        ----------
        x :
            Input data, shape (n_samples, n_input_features).
        y :
            Target data, shape (n_samples, n_output_features).
        lr :
            How the gradient is applied:

            - **float ≥ 0** — direct scaling of the gradient.
            - ``"adam"`` (default) — Adam optimiser; step size set by ``params["lr"]``
            (default 1e-3).
        learning_kind :
            Gradient computation mode:

            - ``"precision_weighted"`` (default) — gradient is weighted by the child
            layer's posterior precision before applying *lr*.
            - ``"standard"`` — raw prediction-error outer product, no precision
            weighting.
            - ``"precision_ratio"`` — Kalman-gain rule.
        params :
            Hyper-parameters for the Adam optimiser: ``beta1`` (default 0.9), ``beta2``
            (default 0.999), ``epsilon`` (default 1e-8), and ``lr`` (default 1e-3, the
            Adam step size).
        record_trajectories :
            If True, record the full ``NetworkState`` at every time step (accessible
            via ``self.trajectories``).  This is useful for inspection but significantly
            increases memory usage and slows training. Default is False.
        weight_update :
            If ``True`` (default), the learning phase updates the network's
            weights at the end of each step. Set to ``False`` to freeze the
            weights and run only the inference (prediction → PE → posterior)
            cycle — useful for evaluating a fixed model on new data while
            still recording trajectories.

        Returns
        -------
        VectorizedDeepNetwork
            Self with updated state.

        Raises
        ------
        ValueError
            If *learning_kind* or *lr* is an unrecognised value.
        """
        # Initialize state if needed
        if self.state is None:
            self.state = self._init_state()

        # Recreate (and retrace) the propagation fn when settings change
        needs_retrace = (
            self._propagation_fn is None
            or self._propagation_lr != lr
            or self._propagation_learning_kind != learning_kind
            or self._record_trajectories != record_trajectories
            or self._propagation_weight_update != weight_update
        )
        if needs_retrace:
            self._propagation_fn = self._create_propagation_fn(
                lr, learning_kind, params, record_trajectories, weight_update
            )
            self._propagation_lr = lr
            self._propagation_learning_kind = learning_kind
            self._record_trajectories = record_trajectories
            self._propagation_weight_update = weight_update

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
        self._propagation_learning_kind = None
        self._record_trajectories = False
        self._propagation_weight_update = True
        self._prediction_fn = None
        return self

    def plot_layers(
        self,
        layers: Optional[Union[int, list[int]]] = None,
        variables=("expected_mean",),
        mode: str = "all",
        figsize: Optional[tuple] = None,
        color: Optional[Union[tuple, str]] = None,
        axs=None,
    ):
        """Plot layer-wise parameter trajectories.

        Each row of the figure corresponds to a variable (a field of
        :class:`pyhgf.typing.LayerState`) and each column to a layer.
        ``mode="all"`` draws one line per node; ``mode="mean_ci"`` draws the
        across-node mean and a 95% normal-approximation confidence band.

        Parameters
        ----------
        layers :
            Index or indices of the layers to plot. A single ``int`` is
            accepted as shorthand for a one-element list. ``None`` (default)
            plots all layers.
        variables :
            Name (or sequence of names) of ``LayerState`` fields to plot,
            e.g. ``"expected_mean"``, ``"precision"``, ``"mean_vol"``. The
            derived name ``"PWPE"`` is also accepted: it plots the magnitude
            of the precision-weighted prediction error,
            ``|mean - expected_mean| * expected_precision``.
        mode :
            ``"all"`` for one line per node, ``"mean_ci"`` for mean ± 95% CI.
        figsize :
            Matplotlib figure size in inches.
        color :
            Colour of the lines (``"all"`` mode) or of the mean curve and
            confidence band (``"mean_ci"`` mode). When ``None`` (default),
            Matplotlib's default colour cycle is used.
        axs :
            A 2D array of Matplotlib axes (rows = variables, cols = layers)
            where to draw the trajectories. The default is ``None`` (create a
            new figure).

        Returns
        -------
        axs :
            2D array of Matplotlib axes, shape ``(n_variables, n_layers)``.

        Raises
        ------
        ValueError
            If ``self.trajectories`` is ``None`` (run ``fit(...,
            record_trajectories=True)`` first), or if a variable / layer index
            / mode is invalid.
        """
        from pyhgf.plots.matplotlib import plot_layers as _plot_layers

        return _plot_layers(
            network=self,
            layers=layers,
            variables=variables,
            mode=mode,
            figsize=figsize,
            color=color,
            axs=axs,
        )

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
