# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized deep predictive coding network.

This module provides a vectorized implementation of deep HGF networks that uses layer-
wise matrix operations instead of per-node updates.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from pyhgf.model.builder import LayerConfig, resolve_coupling_fn
from pyhgf.typing.vectorised import (
    VOLATILITY_STATE_FIELDS,
    Layer,
    LayerParams,
    LayerState,
    Network,
    stack_layers,
)
from pyhgf.utils.vectorized_belief_propagation import (
    batch_step,
    batched_prediction_pass,
    batched_prediction_states,
    input_prediction_error,
    learn_sweep,
    prediction_pass,
    prediction_sweep,
    run_scan,
    update_sweep,
)
from pyhgf.utils.weight_initialisation import (
    he_init,
    orthogonal_init,
    sparse_init,
    xavier_init,
)

# Default per-layer parameter values (matching ``LayerParams.create``).
_LAYER_PARAM_DEFAULTS: dict[str, float] = {
    "tonic_volatility_vol": -4.0,
}

# Names of fields that can be overridden per layer (eqx.Modules expose dataclass fields).
_LAYER_STATE_FIELDS: frozenset[str] = frozenset(LayerState.__dataclass_fields__.keys())
_LAYER_PARAM_FIELDS: frozenset[str] = frozenset(LayerParams.__dataclass_fields__.keys())
# Volatility-level state fields, absent (``None``) on a layer without a
# volatility parent — a state override for one of these on such a layer is
# dropped rather than re-allocating the array.
_VOL_STATE_FIELDS: frozenset[str] = frozenset(VOLATILITY_STATE_FIELDS)

# Minimum identical-layer count for ``add_layer_stack`` to auto-collapse the
# block into a ``LayerStack`` (i.e. switch the propagation kernels to
# ``jax.lax.scan``). Below this, the savings on JIT trace cost don't
# outweigh the scan setup overhead.
_SCAN_AUTO_THRESHOLD: int = 5
_LAYER_OVERRIDE_FIELDS: frozenset[str] = _LAYER_STATE_FIELDS | _LAYER_PARAM_FIELDS


class DeepNetwork:
    """Deep predictive coding network with vectorized operations.

    This class implements a deep hierarchical Gaussian filter using layer-wise
    vectorized operations for efficient scaling to large networks.

    Unlike the standard DeepNetwork which uses per-node updates with Python loops, this
    implementation uses JAX matrix operations to update all nodes in a layer
    simultaneously.

    Parameters
    ----------
    coupling_fn :
        Coupling function applied between layers. Default is linear (identity). This
        function is applied to parent means before the weighted sum to predict child
        means.
    volatility_updates :
        The type of volatility-level posterior update. Can be ``"unbounded"``
        (default), ``"eHGF"`` or ``"standard"``.
    max_posterior_precision :
        Upper bound applied to every posterior precision write. Defaults to ``1e10``.

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
        volatility_updates: str = "unbounded",
        max_posterior_precision: float = 1e10,
        precision_clipping_value: float = 1e-6,
    ):
        r"""Initialize a VectorizedDeepNetwork.

        Parameters
        ----------
        coupling_fn :
            Coupling function applied between layers. Default is linear (identity),
            matching the Rust backend and the Network class. This function is
            applied to parent means before the weighted sum to predict child means.
        volatility_updates :
            The type of volatility-level posterior update. Can be ``"unbounded"``
            (default), ``"eHGF"`` or ``"standard"``. Matches the Network
            class and Rust backend.
        max_posterior_precision :
            Upper bound applied to every posterior precision write (value level and
            volatility level). Defaults to ``1e10`` and is shared with the nodalised
            ``Network`` and the Rust backend. Increase it to relax the cap, or lower it
            to be more conservative against precision blow-up.
        precision_clipping_value :
            Bound applied to binary-layer predicted means
            (``[v, 1 - v]``) so the implied binary precision
            :math:`\hat{\mu}(1 - \hat{\mu})` never collapses. A larger value (e.g.
            ``1e-3``, matching TAPAS) stabilises the forward filter in high-volatility
            regimes; a very small value (default ``1e-6``) avoids flat, zero-gradient
            plateaus that hurt gradient-based inference. Shared with the nodalised
            ``Network`` and the Rust backend.
        """
        self.coupling_fn = coupling_fn
        self.volatility_updates = volatility_updates
        self.max_posterior_precision = float(max_posterior_precision)
        self.precision_clipping_value = float(precision_clipping_value)
        self.layer_sizes: list[int] = []
        self.layer_kinds: list[str] = []
        # Per-layer overrides for fields of ``LayerState`` and ``LayerParams``.
        # Populated from the ``**kwargs`` of :meth:`add_layer`.
        self.layer_overrides: list[dict[str, float]] = []
        self.add_constant_inputs: list[bool] = []
        self.fully_connected: list[bool] = []
        self.coupling_fns: list[Callable] = []  # per-layer coupling functions
        self.volatility_parents: list[bool] = []
        # Indices of consecutive layers to collapse into a ``LayerStack`` at
        # ``_init_state`` time. Each entry is a ``(start, end_exclusive)`` half-
        # open interval over ``self.layer_sizes``, populated automatically by
        # ``add_layer_stack`` when ≥ ``_SCAN_AUTO_THRESHOLD`` identical layers
        # sit on a compatible (matching-width, non-binary) layer below.
        self.scan_blocks: list[tuple[int, int]] = []
        self.state: Optional[Network] = None
        self.opt_state: Optional[optax.OptState] = None
        self._optimizer: Optional[optax.GradientTransformation] = None
        self.trajectories: Optional[Network] = None
        self.predictions: Optional[jnp.ndarray] = None
        # Per-sample input-layer errors from the last batch_update call,
        # shape (batch, n_input_features).
        self.input_errors: Optional[jnp.ndarray] = None

    @classmethod
    def from_configs(
        cls,
        configs: list[LayerConfig],
        coupling_fn: Optional[Callable | str] = None,
        volatility_updates: str = "unbounded",
        max_posterior_precision: float = 1e10,
        precision_clipping_value: float = 1e-6,
    ) -> "DeepNetwork":
        """Build a network from a list of layer configurations.

        Constructs a DeepNetwork by adding layers in order from a list of
        :class:`LayerConfig` objects. This is useful for reproducible experiments,
        hyperparameter sweeps, and programmatic model construction.

        Parameters
        ----------
        configs : list[LayerConfig]
            List of layer configurations, outermost (output) layer first.
        coupling_fn : Optional[Callable | str], default None
            Default coupling function applied between layers. Can be a callable
            (e.g., ``jax.nn.gelu``) or a string name (e.g., ``"gelu"``).
            Per-layer overrides in configs take precedence.
        volatility_updates : str, default "unbounded"
            The type of volatility-level posterior update.
        max_posterior_precision : float, default 1e10
            Upper bound applied to posterior precision writes.
        precision_clipping_value : float, default 1e-6
            Bound applied to binary-layer predicted means.

        Returns
        -------
        DeepNetwork
            Fully constructed network with all layers added.

        Raises
        ------
        ValueError
            If any layer config is invalid or a coupling function name
            is not recognized.

        Examples
        --------
        >>> from pyhgf.model import LayerConfig, DeepNetwork
        >>> configs = [
        ...     LayerConfig(size=10),
        ...     LayerConfig(size=20, coupling_fn="gelu"),
        ...     LayerConfig(size=15)
        ... ]
        >>> net = DeepNetwork.from_configs(configs)
        >>> net.n_layers
        3
        """
        if not configs:
            raise ValueError("configs must not be empty")

        # Resolve the network-level coupling function
        net_coupling_fn = resolve_coupling_fn(coupling_fn)

        # Create the network with the resolved coupling function
        net = cls(
            coupling_fn=net_coupling_fn or (lambda x: x),
            volatility_updates=volatility_updates,
            max_posterior_precision=max_posterior_precision,
            precision_clipping_value=precision_clipping_value,
        )

        # Add layers in order
        for config in configs:
            # Resolve per-layer coupling function
            layer_coupling_fn = resolve_coupling_fn(config.coupling_fn)
            if layer_coupling_fn is None:
                layer_coupling_fn = net_coupling_fn

            # Prepare per-layer parameter overrides
            overrides = {}
            if config.tonic_volatility_vol is not None:
                overrides["tonic_volatility_vol"] = config.tonic_volatility_vol

            # Add the layer
            net = net.add_layer(
                size=config.size,
                kind=config.kind,
                add_constant_input=config.add_constant_input,
                fully_connected=config.fully_connected,
                coupling_fn=layer_coupling_fn,
                volatility_parent=config.volatility_parent,
                **overrides,
            )

        return net

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> "DeepNetwork":
        """Build a network from a dictionary configuration.

        Loads a network configuration from a dictionary (e.g., from JSON or YAML),
        making it easy to version control and reproduce experiments.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary with keys:
            - "layers" (required): list of layer configs (each a dict or LayerConfig)
            - "coupling_fn" (optional): default coupling function name or callable
            - "volatility_updates" (optional): volatility update scheme
            - "max_posterior_precision" (optional): precision upper bound
            - "precision_clipping_value" (optional): binary-layer clipping bound

        Returns
        -------
        DeepNetwork
            Fully constructed network.

        Raises
        ------
        ValueError
            If "layers" is missing or invalid.
        KeyError
            If required layer fields are missing.

        Examples
        --------
        >>> config = {
        ...     "layers": [
        ...         {"size": 10},
        ...         {"size": 20, "coupling_fn": "gelu"},
        ...         {"size": 15}
        ...     ],
        ...     "coupling_fn": "identity",
        ...     "volatility_updates": "unbounded"
        ... }
        >>> net = DeepNetwork.from_dict(config)
        >>> net.n_layers
        3

        You can also save and load from JSON:

        >>> import json
        >>> json_str = json.dumps(config)
        >>> loaded_config = json.loads(json_str)
        >>> net = DeepNetwork.from_dict(loaded_config)
        """
        if "layers" not in config:
            raise ValueError("config must contain a 'layers' key")

        # Convert layer dicts to LayerConfig objects
        layers_data = config["layers"]
        if not layers_data:
            raise ValueError("'layers' must not be empty")

        configs = [
            LayerConfig.from_dict(layer) if isinstance(layer, dict) else layer
            for layer in layers_data
        ]

        # Extract network-level parameters
        net_config = {
            k: v
            for k, v in config.items()
            if k
            in (
                "coupling_fn",
                "volatility_updates",
                "max_posterior_precision",
                "precision_clipping_value",
            )
        }

        return cls.from_configs(configs, **net_config)

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
            with value and volatility levels. ``"binary"`` uses binary state nodes (one
            independent Bernoulli belief per node). ``"categorical"`` makes the whole
            layer one joint choice over its ``size`` mutually-exclusive classes: the
            expected mean is the softmax across the layer's nodes, and the observation
            clamped during :meth:`fit` must be one-hot. Only valid as the output
            (bottom) layer. See ``docs/source/notebooks/1.1-Binary_HGF.ipynb`` for the
            binary case.
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
            the value level has no volatility source and does not undergo a Gaussian
            random walk (its conditional predicted precision equals the prior
            precision).
        **kwargs :
            Per-layer overrides for any field of :class:`pyhgf.typing.LayerState`
            (e.g. ``mean``, ``precision``, ``expected_mean``, ``expected_precision``,
            ``mean_vol``, ``precision_vol``, ...) or :class:`pyhgf.typing.LayerParams`
            (``tonic_volatility_vol``).
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
            "categorical-state": "categorical",
        }
        kind = _kind_aliases.get(kind, kind)

        valid_kinds = {"volatile", "binary", "categorical"}
        if kind not in valid_kinds:
            raise ValueError(
                f"Invalid layer kind '{kind}'. Choose from {sorted(valid_kinds)}."
            )
        if kind == "categorical" and self.layer_sizes:
            # A softmax couples every node of the layer into one choice; that
            # is only meaningful for the observed (bottom) layer, where the
            # one-hot truth is clamped.
            raise ValueError(
                "Categorical layers represent a single N-way choice and are "
                "only supported as the output (bottom) layer — add it first."
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
        # Eagerly rebuild the Network so ``self.state`` is always queryable
        # after construction. Cheap for typical depths; only triggers fresh
        # array allocations, no JIT trace.
        self.state = self._init_state()
        # Optimiser state's shape depends on the weights tuple, so invalidate
        # any previously initialised ``opt_state``.
        self.opt_state = None
        self._optimizer = None
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

        When the block contains ≥ ``_SCAN_AUTO_THRESHOLD`` (currently 5) identical
        layers sitting on a matching-width volatile layer below, the layers are
        automatically collapsed into a single ``LayerStack`` at network-build time, and
        the propagation kernels ``jax.lax.scan`` over them with a single trace.

        If any eligibility condition is not met (mixed sizes, fewer than 5 layers, width
        mismatch with the layer below, direct binary- or categorical-leaf adjacency, or
        no layer below), the layers are simply added one by one the usual way.

        Parameters
        ----------
        layer_sizes :
            List of layer sizes.
        kind :
            Type of nodes for all layers (``"volatile"``, ``"binary"``, or
            ``"categorical"``).
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
            ``LayerState`` or ``LayerParams``, e.g. ``tonic_volatility_vol``,
            ``precision``).

        Returns
        -------
        VectorizedDeepNetwork
            Self for method chaining.
        """
        auto_scan = (
            len(layer_sizes) >= _SCAN_AUTO_THRESHOLD
            and len(set(layer_sizes)) == 1
            and len(self.layer_sizes) > 0
            and self.layer_sizes[-1] == layer_sizes[0]
            and not (
                len(self.layer_sizes) == 1
                and self.layer_kinds[0] in ("binary", "categorical")
            )
        )

        start = len(self.layer_sizes)
        for size in layer_sizes:
            self.add_layer(
                size=size,
                kind=kind,
                add_constant_input=add_constant_input,
                fully_connected=fully_connected,
                coupling_fn=coupling_fn,
                **kwargs,
            )

        if auto_scan:
            self.scan_blocks.append((start, len(self.layer_sizes)))
        return self

    def _init_state(self) -> Network:
        """Initialize network with uniform weights, as an Equinox ``Network`` PyTree.

        All inter-layer weights are set to ``1.0``. Use :meth:`weight_initialisation`
        after construction to apply Xavier, He, orthogonal, or sparse initialisation.
        """
        layers: list[Layer] = []

        for i, size in enumerate(self.layer_sizes):
            overrides = self.layer_overrides[i]

            # Per-layer state with overrides (broadcast scalar -> (n_nodes,)).
            # Without a volatility parent the volatility fields are ``None``; a
            # state override for one of them is dropped, since the frozen
            # volatility level never reads it.
            has_vol = self.volatility_parents[i]
            state = LayerState.create(size, has_volatility_parent=has_vol)
            state_overrides = {
                k: jnp.full(size, v)
                for k, v in overrides.items()
                if k in _LAYER_STATE_FIELDS and (has_vol or k not in _VOL_STATE_FIELDS)
            }
            if state_overrides:
                state = dataclasses.replace(state, **state_overrides)

            # Per-layer params with overrides
            param_kwargs = dict(_LAYER_PARAM_DEFAULTS)
            for k, v in overrides.items():
                if k in _LAYER_PARAM_FIELDS:
                    param_kwargs[k] = v
            params = LayerParams.create(n_nodes=size, **param_kwargs)

            # `weights_in` lives on the parent (this Layer); layer 0 has none.
            if i > 0:
                prev_size = self.layer_sizes[i - 1]
                n_parent_cols = size + (1 if self.add_constant_inputs[i] else 0)
                if self.fully_connected[i]:
                    weights_in = jnp.ones((prev_size, n_parent_cols))
                else:
                    weights_in = jnp.eye(prev_size, n_parent_cols)
            else:
                weights_in = None

            coupling_fn = self.coupling_fns[i]
            layers.append(
                Layer(
                    state=state,
                    params=params,
                    weights_in=weights_in,
                    coupling_fn=coupling_fn,
                    add_constant_input=self.add_constant_inputs[i],
                    has_volatility_parent=self.volatility_parents[i],
                    is_input_layer=(i == 0),
                    fully_connected=self.fully_connected[i],
                    kind=self.layer_kinds[i],
                )
            )

        # Collapse any registered scan blocks (auto-registered by
        # ``add_layer_stack`` when N≥_SCAN_AUTO_THRESHOLD identical layers were
        # added) into ``LayerStack`` elements. Blocks are appended in order,
        # so no sorting is needed.
        if self.scan_blocks:
            elements: list = []
            block_iter = iter(self.scan_blocks)
            next_block = next(block_iter, None)
            i = 0
            while i < len(layers):
                if next_block is not None and i == next_block[0]:
                    start, end = next_block
                    elements.append(stack_layers(layers[start:end]))
                    i = end
                    next_block = next(block_iter, None)
                else:
                    elements.append(layers[i])
                    i += 1
        else:
            elements = layers

        return Network(
            layers=tuple(elements),
            volatility_updates=self.volatility_updates,
            max_posterior_precision=self.max_posterior_precision,
            precision_clipping_value=self.precision_clipping_value,
        )

    def weight_initialisation(
        self,
        strategy: Optional[str] = None,
        key: Optional[jax.Array] = None,
        **kwargs,
    ) -> "DeepNetwork":
        """Initialise inter-layer weight matrices.

        Parameters
        ----------
        strategy :
            Initialisation strategy. One of ``"xavier"``, ``"he"``, ``"orthogonal"``, or
            ``"sparse"``. If *None*, weights are left unchanged (all ``1.0``).
        key :
            ``jax.random.PRNGKey`` controlling the randomness. Defaults to
            ``jax.random.key(0)``. Replaces the legacy ``seed: int`` argument
            (breaking change).
        **kwargs
            Extra keyword arguments forwarded to the initialisation function (e.g.
            ``gain`` for orthogonal, ``sparsity`` / ``std`` for sparse).

        Returns
        -------
        DeepNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If the strategy name is not recognised or no layer has been added.
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
                "Add at least one layer before calling weight_initialisation."
            )

        _init_fns: dict[str, Callable[..., np.ndarray]] = {
            "xavier": xavier_init,
            "he": he_init,
            "orthogonal": orthogonal_init,
            "sparse": sparse_init,
        }
        init_fn = _init_fns[strategy]

        # Convert the JAX PRNG key to a single integer seed for the
        # numpy-backed init helpers.
        if key is None:
            key = jax.random.key(0)
        seed = int(jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.int32))

        from pyhgf.typing.vectorised import LayerStack

        new_elements = list(self.state.layers)
        for i, elem in enumerate(new_elements):
            if elem.weights_in is None:
                continue
            if isinstance(elem, LayerStack):
                # Stack has weights_in shape (N, n_child, n_parent[+1]). Use
                # the same seed for every slice — matches the unrolled init's
                # "same seed across all layers" semantics (necessary for
                # byte-parity with the unrolled path; the underlying "all
                # layers identical at init" pattern is a separate concern).
                n_slices, n_children, n_parents = elem.weights_in.shape
                flat = init_fn(n_parents, n_children, seed=seed, **kwargs)
                per_slice = jnp.array(flat.reshape(n_children, n_parents))
                new_weights = jnp.broadcast_to(
                    per_slice, (n_slices, n_children, n_parents)
                )
            else:
                n_children, n_parents = elem.weights_in.shape
                flat = init_fn(n_parents, n_children, seed=seed, **kwargs)
                new_weights = jnp.array(flat.reshape(elem.weights_in.shape))
            new_elements[i] = dataclasses.replace(elem, weights_in=new_weights)

        self.state = dataclasses.replace(self.state, layers=tuple(new_elements))
        return self

    def fit(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        optimizer: optax.GradientTransformation,
        learning_kind: str = "precision_weighted",
        record: Optional[tuple] = None,
        weight_update: bool = True,
        time_step: float = 1.0,
    ) -> "DeepNetwork":
        r"""Fit network to data.

        Parameters
        ----------
        x :
            Input data, shape (n_samples, n_input_features).
        y :
            Target data, shape (n_samples, n_output_features).
        optimizer :
            Any ``optax.GradientTransformation`` — e.g. ``optax.sgd(0.2)``,
            ``optax.adam(1e-3)``, or any chain. Migration from the legacy
            ``lr=`` API: ``lr=0.2`` → ``optimizer=optax.sgd(0.2)``,
            ``lr="adam"`` → ``optimizer=optax.adam(1e-3)``.
        learning_kind :
            Gradient computation mode passed to
            :func:`~pyhgf.updates.vectorized.learning.vectorized_weight_gradient`:
            ``"standard"`` or ``"precision_weighted"`` (default). Both are
            strictly local. Each weight update reads only its own
            connection's prediction error, activation, and precision.
        record :
            Tuple of ``LayerState`` field names to record at every time step,
            e.g. ``("expected_mean", "precision")``. ``None`` (default) skips
            recording. Use the :data:`~pyhgf.typing.RECORD_ALL` constant
            for the legacy "record everything" behaviour. The result lands in
            ``self.trajectories`` as ``dict[field_name, tuple[Array, ...]]``
            where each per-layer array has a leading ``(T,)`` axis.
        weight_update :
            If True (default), the learning phase updates weights each step. If False,
            weights and ``opt_state`` are frozen. Useful for evaluating a fixed model on
            new data while still recording trajectories.
        time_step :
            Uniform inference time step :math:`\\Delta t` applied at every
            ``propagation_step``. Replaces the legacy ``net.state.time_step``
            attribute (``NetworkState`` no longer carries it). Default ``1.0``.

        Returns
        -------
        DeepNetwork
            Self with updated state and optimiser state.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before calling fit.")

        # Normalise + validate ``record``. A ``None`` or empty tuple disables
        # recording. Otherwise each name must be a real ``LayerState`` field.
        record_tuple: tuple = tuple(record) if record else ()
        if record_tuple:
            valid_fields = set(LayerState.__dataclass_fields__.keys())
            invalid = [f for f in record_tuple if f not in valid_fields]
            if invalid:
                raise ValueError(
                    f"Unknown record field(s) {invalid}. Valid: {sorted(valid_fields)}."
                )

        # Initialise opt_state on first fit, or when the optimiser changes.
        if self.opt_state is None or self._optimizer is not optimizer:
            self.opt_state = optimizer.init(self.state.weights_tuple())
            self._optimizer = optimizer

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        (self.state, self.opt_state), step_output = run_scan(
            (self.state, self.opt_state),
            (x, y),
            optimizer,
            learning_kind,
            weight_update,
            record_tuple,
            float(time_step),
        )

        if record_tuple:
            self.trajectories, self.predictions = step_output
        else:
            self.trajectories = None
            self.predictions = step_output

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
            raise ValueError("Add at least one layer before calling predict.")

        x = jnp.asarray(x)

        # Handle single sample vs batch.
        if x.ndim == 1:
            return prediction_pass(self.state, x)
        return batched_prediction_pass(self.state, x)

    def prediction(self, x: Union[np.ndarray, jnp.ndarray]) -> "DeepNetwork":
        """Run the top-down prediction sweep only.

        Clamps the predictors ``x`` on the input (top) layer, predicts every layer
        from the one above, and writes the resulting beliefs back to ``self.state``.
        No prediction errors, posterior updates, or weight learning are performed —
        use :meth:`update` for the bottom-up belief update and :meth:`fit` for weight
        learning.

        Unlike :meth:`predict` (which returns the bottom layer's ``expected_mean`` and
        leaves the network untouched), this advances and stores the network state, so
        a subsequent :meth:`update` sees the predicted beliefs.

        Parameters
        ----------
        x :
            Predictors clamped on the top layer, shape ``(n_input_features,)``. Single
            sample only; use :meth:`predict` for a batched, read-only forward pass.

        Returns
        -------
        DeepNetwork
            Self, with ``self.state`` advanced by the prediction sweep.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before calling prediction.")
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError(
                "prediction() operates on a single sample (1D x); "
                "use predict() for a batched read-only forward pass."
            )
        self.state = prediction_sweep(self.state, x)
        return self

    def update(
        self,
        y: Union[np.ndarray, jnp.ndarray],
        optimizer: Optional[optax.GradientTransformation] = None,
        learning_kind: str = "precision_weighted",
    ) -> "DeepNetwork":
        """Run the bottom-up prediction-error + posterior-update sweep.

        Clamps the observations ``y`` on the output (bottom) layer, computes the leaf
        prediction error, then performs the interleaved posterior-update + prediction-
        error for every interior layer, in the correct bottom-up order. Belief states
        (means and precisions) are always updated.

        If an ``optimizer`` is supplied, a weight-learning phase runs *after* the belief
        sweep: PE-driven gradients are formed and applied to every inter-layer weight
        matrix, with the optimiser state held on ``self.opt_state``. With
        ``optimizer=None`` (default) the weights are left unchanged, exactly as
        :meth:`fit` would leave them with ``weight_update=False``.

        Typically called right after :meth:`prediction`, which sets up the predicted
        beliefs this sweep corrects, so a full local learning step without backprop is::

            net.prediction(x).update(y, optimizer=optax.sgd(1e-2))

        Parameters
        ----------
        y :
            Observations clamped on the bottom layer, shape ``(n_output_features,)``.
            Single sample only.
        optimizer :
            Optional ``optax.GradientTransformation``. If given, run the weight-learning
            phase and update ``self.opt_state``. If ``None``, only beliefs are updated.
        learning_kind :
            Weight-gradient mode used when ``optimizer`` is supplied. One of
            ``"standard"`` or ``"precision_weighted"`` (default). Both are
            strictly local. Each weight update reads only its own
            connection's prediction error, activation, and precision.

        Returns
        -------
        DeepNetwork
            Self, with ``self.state`` (and, if learning, ``self.opt_state``) advanced.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before calling update.")
        y = jnp.asarray(y)
        if y.ndim != 1:
            raise ValueError("update() operates on a single sample (1D y).")
        self.state = update_sweep(self.state, y)

        if optimizer is not None:
            # Initialise opt_state on first use, or when the optimiser changes.
            if self.opt_state is None or self._optimizer is not optimizer:
                self.opt_state = optimizer.init(self.state.weights_tuple())
                self._optimizer = optimizer
            self.state, self.opt_state = learn_sweep(
                self.state, self.opt_state, optimizer, learning_kind
            )
        return self

    def input_error(self) -> jnp.ndarray:
        """Prediction error at the input (top) layer.

        The bottom-up sweep leaves the top layer untouched (its values are
        clamped to the predictors), so this is the only way to read the error
        message the input would receive from the layer below — the child
        layer's prediction error, weighted by its confidence (precision) and
        routed back through the connecting weights. Typical usage runs the
        two sweeps first::

            net.prediction(x).update(y)
            error_at_input = net.input_error()

        Because prediction errors follow the ``observed - predicted``
        convention, the result is the negative of the gradient of a
        squared-error loss with respect to the predictors.

        Returns
        -------
        jnp.ndarray
            The prediction error at the top layer, shape
            ``(n_input_features,)``.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before calling input_error.")
        return input_prediction_error(self.state)

    def predict_states(
        self, x: Union[np.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, tuple]:
        """Batched forward pass that also returns the per-sample swept states.

        Returns the same predictions as :meth:`predict` on 2D input, plus the
        per-sample layer states the sweep computed. Passing those states to
        :meth:`batch_update` (as ``predicted``) skips its internal forward
        sweep, so a caller that has already predicted does not pay for the
        sweep twice.

        Parameters
        ----------
        x :
            Predictors, shape ``(batch, n_input_features)``.

        Returns
        -------
        predictions :
            Shape ``(batch, n_output_features)``.
        states :
            One batched ``LayerState`` per element — an opaque value to hand
            back to :meth:`batch_update`.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before calling predict_states.")
        states = batched_prediction_states(self.state, jnp.asarray(x))
        return states[0].expected_mean, states

    def batch_update(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        optimizer: Optional[optax.GradientTransformation] = None,
        learning_kind: str = "precision_weighted",
        update_confidences: bool = True,
        time_step: float = 1.0,
        predicted: Optional[tuple] = None,
    ) -> "DeepNetwork":
        """One batch-synchronous learning step over many samples at once.

        Every sample in the batch is processed from the same state — same
        weights, same confidences — in parallel; the per-sample weight
        gradients and confidence changes are then *averaged* and applied
        once, so the whole batch counts as a single observation. This
        differs from :meth:`fit`, which scans samples sequentially and lets
        the confidences adapt from one sample to the next (the natural mode
        for a time series). Use this method when samples are exchangeable —
        e.g. many token positions learned together.

        The per-sample errors at the input layer are stored on
        ``self.input_errors``, shape ``(batch, n_input_features)`` — see
        :meth:`input_error` for their meaning and sign convention.

        Parameters
        ----------
        x :
            Predictors, shape ``(batch, n_input_features)``.
        y :
            Observations, shape ``(batch, n_output_features)``.
        optimizer :
            Optax optimiser for the weight step. ``None`` (default) freezes
            the weights.
        learning_kind :
            Weight-gradient mode, as in :meth:`fit`.
        update_confidences :
            If True (default), carry the batch-averaged change of the
            confidence state (value-level precisions and the volatility
            level) into ``self.state``. Set to False to keep confidences
            pinned, e.g. for exact comparisons against backpropagation.
        time_step :
            Inference time step, applied once per batch.
        predicted :
            Optional per-sample predicted states from :meth:`predict_states`;
            when given, the internal forward sweep is skipped and ``x`` is
            ignored.

        Returns
        -------
        DeepNetwork
            Self, with ``self.state`` (and, if learning, ``self.opt_state``)
            advanced by one batch.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before calling batch_update.")
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                "batch_update() operates on batches: x and y must be 2D "
                "(batch, n_features)."
            )

        if optimizer is not None and (
            self.opt_state is None or self._optimizer is not optimizer
        ):
            self.opt_state = optimizer.init(self.state.weights_tuple())
            self._optimizer = optimizer

        self.state, self.opt_state, self.input_errors = batch_step(
            self.state,
            self.opt_state,
            x,
            y,
            optimizer=optimizer,
            learning_kind=learning_kind,
            update_confidences=update_confidences,
            time_step=float(time_step),
            predicted=predicted,
        )
        return self

    def reset(self) -> "DeepNetwork":
        """Reset the network state."""
        self.state = self._init_state()
        self.opt_state = None
        self._optimizer = None
        return self

    def save(self, path: Union[str, os.PathLike]) -> "DeepNetwork":
        """Serialise ``self.state`` array leaves to ``path``.

        Only the network's array content is written (weights and layer states). Static
        fields (kinds, flags, coupling functions) and the optimiser state are *not*
        persisted. The user is expected to rebuild the same topology before calling
        :meth:`load`.
        """
        if self.state is None:
            raise ValueError("Add at least one layer before saving.")
        eqx.tree_serialise_leaves(str(path), self.state)
        return self

    def load(self, path: Union[str, os.PathLike]) -> "DeepNetwork":
        """Read array leaves from ``path`` back into ``self.state``.

        ``self.state`` must already exist with the same treedef (same layer topology).
        Typically by recreating the builder before calling ``load``.
        """
        if self.state is None:
            raise ValueError(
                "Recreate the builder topology (add_layer …) before load()."
            )
        self.state = eqx.tree_deserialise_leaves(str(path), self.state)
        # The optimiser state's shape depends on the loaded weights — clear it.
        self.opt_state = None
        self._optimizer = None
        return self

    def to_pandas(self) -> pd.DataFrame:
        """Flatten ``self.trajectories`` into a wide-format ``pd.DataFrame``.

        Returns one row per recorded time step and one column per
        ``(layer, node, field)`` triple, named ``L{layer}_N{node}_{field}``. Only the
        fields that were passed to ``fit(record=...)`` appear.
        """
        if not self.trajectories:
            raise ValueError(
                "No recorded trajectories. Pass ``record=(...)`` to ``fit``."
            )
        columns: dict[str, np.ndarray] = {}
        for field, per_layer in self.trajectories.items():
            for layer_idx, arr in enumerate(per_layer):
                arr_np = np.asarray(arr)  # (T, n_nodes)
                for node_idx in range(arr_np.shape[1]):
                    columns[f"L{layer_idx}_N{node_idx}_{field}"] = arr_np[:, node_idx]
        return pd.DataFrame(columns)

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
