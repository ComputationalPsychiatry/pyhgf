# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised deep predictive coding network.

This module provides a vectorised implementation of deep HGF networks that uses layer-
wise matrix operations instead of per-node updates.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import warnings
from typing import Any, Callable, Optional, Sequence, Union

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
from pyhgf.updates.vectorised.learning import (
    SynapticUncertaintySettings,
    resolve_synaptic_uncertainty_settings,
)
from pyhgf.utils.vectorised_belief_propagation import (
    _weight_quantities,
    batch_step,
    batched_prediction_pass,
    batched_prediction_states,
    input_prediction_error,
    learn_sweep,
    prediction_pass,
    prediction_sweep,
    run_continuous_scan,
    run_scan,
    update_sweep,
)
from pyhgf.utils.weight_initialisation import (
    _init_matrix,
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
# ``LayerParams`` fields carried by continuous layers only.
_CONTINUOUS_PARAM_FIELDS: frozenset[str] = frozenset({
    "tonic_volatility",
    "tonic_drift",
    "autoconnection_strength",
})

# Minimum identical-layer count for ``add_layer_stack`` to auto-collapse the
# block into a ``LayerStack`` (i.e. switch the propagation kernels to
# ``jax.lax.scan``). Below this, the savings on JIT trace cost don't
# outweigh the scan setup overhead.
_SCAN_AUTO_THRESHOLD: int = 5
_LAYER_OVERRIDE_FIELDS: frozenset[str] = _LAYER_STATE_FIELDS | _LAYER_PARAM_FIELDS


class DeadGradientWarning(UserWarning):
    """Some weight matrices receive essentially no learning signal.

    Raised by :meth:`DeepNetwork.check_gradient_health` when part of the network is
    training while another part receives (near-)zero gradients — typically the
    matrices far from the observations. A network in this state silently degrades
    into a random-feature readout: the live matrices fit against a frozen upper
    stack, accuracy can look respectable, and nothing else reports the failure.

    The two known causes are Bayesian absorption (each layer's inference consumes
    most of the error message before passing the residual up) and float32
    annihilation of the belief-side factor ``(μ − μ̂)·π`` at high precision.
    """


class DeepNetwork:
    """Deep predictive coding network with vectorised operations.

    This class implements a deep hierarchical Gaussian filter using layer-wise
    vectorised operations for efficient scaling to large networks.

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
    update_input_layer :
        Whether the belief sweeps reach the top (input) layer. Defaults to ``False``.
    mean_field_updates :
        If ``False`` (default), use the relaxed prediction and posterior updates. If
        ``True``, use the original mean-field updates, matching the ``Network``
        class.

    Examples
    --------
    >>> # Build a network with method chaining
    >>> net = (
    ...     VectorisedDeepNetwork()
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
        update_input_layer: bool = False,
        predict_precision: bool = True,
        feedforward_uncertainty: bool = False,
        tonic_volatility: bool = False,
        mean_field_updates: bool = False,
    ):
        r"""Initialise a VectorisedDeepNetwork.

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
        update_input_layer :
            Whether the belief sweeps reach the top (input) layer, which holds the
            predictors. Defaults to ``False``.
        feedforward_uncertainty :
            Whether value parents propagate their uncertainty to their children's
            predicted precision. With ``False`` (the default) they do not: the
            value-coupling variance is dropped, the marginal and the conditional
            predicted precision coincide, and the only uncertainty entering a layer is
            its own volatility parent's, which makes every layer behave as the top
            layer already does. With ``True`` a volatile layer's marginal predicted
            precision carries the value-coupling variance, so a parent that is unsure
            makes its children less precise.
        tonic_volatility :
            Whether volatile layers carry a value-level tonic volatility
            :math:`\omega`. With ``False`` (the default) the parameter is
            structurally absent: a layer has no intrinsic volatility at all and
            only inherits volatility from its (implied) volatility parent. With
            ``True`` every volatile layer carries it (default ``-4.0``,
            overridable per layer via ``add_layer(..., tonic_volatility=...)``);
            :math:`\omega` then adds to the volatility parent's contribution
            inside the log-volatility exponent, and a layer built with
            ``volatility_parent=False`` still diffuses at the fixed rate
            :math:`\exp(\omega)`. Setting the value to ``0.0`` on a layer with a
            volatility parent is exactly neutral.
        mean_field_updates :
            If ``False`` (default), use the relaxed prediction and posterior
            updates, which lift the mean-field assumption on value-coupling edges
            via Schur-complement and Laplace/MGF corrections. If ``True``, use the
            original mean-field updates, matching
            ``Network(mean_field_updates=True)``: the log-volatility exponent
            carries no MGF correction, and every value message is weighted by the
            child's canonical predicted precision instead of the smoothing
            factors. The mean-field scheme carries no value-coupling variance, so
            combining it with ``feedforward_uncertainty=True`` raises a
            ``ValueError``; ``learning_kind='synaptic_uncertainty'`` is likewise
            unavailable, since its curvature and evidence terms are derived from
            the relaxed precisions.
        """
        if mean_field_updates and feedforward_uncertainty:
            raise ValueError(
                "mean_field_updates=True is incompatible with "
                "feedforward_uncertainty=True: the mean-field prediction treats "
                "value parents as point estimates, so there is no value-coupling "
                "variance to propagate."
            )
        self.coupling_fn = coupling_fn
        self.volatility_updates = volatility_updates
        self.max_posterior_precision = float(max_posterior_precision)
        self.precision_clipping_value = float(precision_clipping_value)
        self.update_input_layer = bool(update_input_layer)
        self.predict_precision = bool(predict_precision)
        self.feedforward_uncertainty = bool(feedforward_uncertainty)
        self.tonic_volatility = bool(tonic_volatility)
        self.mean_field_updates = bool(mean_field_updates)
        self.layer_sizes: list[int] = []
        self.layer_kinds: list[str] = []
        # Per-layer overrides for fields of ``LayerState`` and ``LayerParams``.
        # Populated from the ``**kwargs`` of :meth:`add_layer`.
        self.layer_overrides: list[dict[str, float]] = []
        self.add_constant_inputs: list[bool] = []
        self.fully_connected: list[bool] = []
        self.coupling_fns: list[Callable] = []  # per-layer coupling functions
        self.volatility_parents: list[bool] = []
        # Continuous-layer topology: for each layer, the index of the layer it
        # is the value / volatility parent of (``None`` when it has no such
        # child), plus the base strengths and connectivity of its W and κ
        # matrices (see ``_fan_in_matrix``). Other kinds carry the resolved
        # defaults here and never read them.
        self.value_children: list[Optional[int]] = []
        self.volatility_children: list[Optional[int]] = []
        self.value_couplings: list[float] = []
        self.volatility_couplings: list[float] = []
        self.volatility_fully_connected: list[bool] = []
        # Indices of consecutive layers to collapse into a ``LayerStack`` at
        # ``_init_state`` time. Each entry is a ``(start, end_exclusive)`` half-
        # open interval over ``self.layer_sizes``, populated automatically by
        # ``add_layer_stack`` when ≥ ``_SCAN_AUTO_THRESHOLD`` identical layers
        # sit on a compatible (matching-width, non-binary) layer below.
        self.scan_blocks: list[tuple[int, int]] = []
        self.state: Optional[Network] = None
        self.opt_state: Optional[optax.OptState] = None
        self._optimiser: Optional[optax.GradientTransformation] = None
        self.trajectories: Optional[Network] = None
        self.predictions: Optional[jnp.ndarray] = None
        # Per-sample input-layer errors from the last batch_update call,
        # shape (batch, n_input_features).
        self.input_errors: Optional[jnp.ndarray] = None

    @property
    def _is_continuous(self) -> bool:
        """Whether this is a continuous network.

        Kinds cannot mix within a network, so layer 0 decides for all of them.
        """
        return bool(self.layer_kinds) and self.layer_kinds[0] == "continuous"

    @classmethod
    def from_configs(
        cls,
        configs: list[LayerConfig],
        coupling_fn: Optional[Callable | str] = None,
        volatility_updates: str = "unbounded",
        max_posterior_precision: float = 1e10,
        precision_clipping_value: float = 1e-6,
        update_input_layer: bool = False,
        **network_kwargs,
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
        update_input_layer : bool, default False
            Whether the belief sweeps reach the top (input) layer.
        **network_kwargs
            Any remaining network-level constructor argument, such as
            ``predict_precision`` or ``feedforward_uncertainty``. Forwarded
            unchanged to :class:`DeepNetwork`, so a setting that has to be in
            place before the state is built can be given here.

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
            update_input_layer=update_input_layer,
            **network_kwargs,
        )

        # Add layers in order
        for config in configs:
            # Resolve per-layer coupling function
            layer_coupling_fn = resolve_coupling_fn(config.coupling_fn)
            if layer_coupling_fn is None:
                layer_coupling_fn = net_coupling_fn

            # Prepare per-layer parameter overrides. Annotated as ``Any`` so
            # unpacking below type-checks against ``add_layer``'s typed
            # keyword parameters, not just the float-valued ``**kwargs``.
            overrides: dict[str, Any] = {}
            if config.tonic_volatility is not None:
                overrides["tonic_volatility"] = config.tonic_volatility
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
            Configuration dictionary. ``"layers"`` (a list of layer configs, each a
            dict or a :class:`LayerConfig`) is required. Every other key naming a
            constructor argument of :class:`DeepNetwork` is optional and is forwarded
            to it, which covers ``"coupling_fn"``, ``"volatility_updates"``,
            ``"max_posterior_precision"``, ``"precision_clipping_value"``,
            ``"update_input_layer"``, ``"predict_precision"``,
            ``"feedforward_uncertainty"`` and ``"tonic_volatility"``. Keys naming
            nothing on the constructor are
            ignored, so a config may carry annotations of its own.

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
        # Read the network-level keys off the constructor signature rather than a
        # list kept here, so a new constructor argument is accepted by ``from_dict``
        # without a second edit. A network-level setting that reaches the state only
        # at construction is unsettable afterwards, so a silently dropped key would
        # build a network that does not match its own config.
        network_fields = set(inspect.signature(cls.__init__).parameters) - {"self"}
        net_config = {k: v for k, v in config.items() if k in network_fields}

        return cls.from_configs(configs, **net_config)

    def add_layer(
        self,
        size: int,
        kind: str = "volatile",
        add_constant_input: Optional[bool] = None,
        fully_connected: bool = True,
        coupling_fn: Optional[Callable] = None,
        volatility_parent: bool = True,
        value_children: Optional[int] = None,
        volatility_children: Optional[int] = None,
        value_coupling: Optional[float] = None,
        volatility_coupling: Optional[float] = None,
        volatility_fully_connected: Optional[bool] = None,
        **kwargs,
    ) -> "DeepNetwork":
        r"""Add a layer of nodes.

        Parameters
        ----------
        predict_precision :
            Whether the prediction sweep advances the precisions. With ``False``
            each layer's predicted precisions are held at its prior precision, its
            effective precision is zero and its volatility level is left untouched,
            so prediction produces only the expected means.
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
            binary case. ``"continuous"`` uses regular continuous state nodes with the
            standard HGF drift semantics; continuous layers cannot mix with the other
            kinds, are filtered with :meth:`input_data` rather than :meth:`fit`, and
            connect through explicit value / volatility parent edges (see
            *value_children* / *volatility_children*).
        add_constant_input :
            If `True`, add a bias term to the layer's predictions. ``None`` (default)
            resolves to `True` for volatile/binary/categorical layers and `False` for
            continuous layers, whose per-node ``tonic_drift`` already plays the bias
            role (passing `True` with a continuous layer raises).
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
            the value level's only volatility source is its tonic volatility, when
            the network allocates one (``tonic_volatility=True``); without one it
            does not undergo a Gaussian random walk at all (its conditional
            predicted precision equals the prior precision). Volatile layers only —
            continuous layers have no internal volatility level and ignore this
            argument.
        value_children :
            Continuous layers only. Index of the existing layer this layer is the
            *value parent* of. When neither *value_children* nor
            *volatility_children* is given, the chain default applies: the new layer
            becomes the value parent of the previously added layer. Naming a parent
            explicitly replaces the chain default entirely, so a layer given only
            *volatility_children* is not a value parent of anything.
        volatility_children :
            Continuous layers only. Index of the existing layer this layer is the
            *volatility parent* of. The κ matrix connecting the two is fixed at
            construction (see *volatility_coupling*) and never learned.
        value_coupling :
            Continuous layers only — base value coupling strength (defaults to
            ``1.0``). Dense ``weights_mean`` entries are ``value_coupling /
            n_parents``, so the drift a child receives is ``value_coupling``
            times the *average* of :math:`g(\hat{\mu})` over the parent layer,
            invariant to the parent layer's width; one-to-one edges keep
            ``value_coupling`` on the diagonal (fan-in 1). Volatile and binary
            layers have no fixed value coupling — their weights are learned, or
            set via :meth:`weight_initialisation`.
        volatility_coupling :
            Continuous layers only — base volatility coupling strength
            :math:`\kappa_0` (defaults to ``1.0``). Dense κ entries are
            :math:`\kappa_0 / n_{\mathrm{parents}}` — the fan-in normalisation
            makes the phasic volatility a child receives :math:`\kappa_0` times
            the *average* of the parent layer's means, invariant to the parent
            layer's width (and its MGF correction the variance of that average).
            One-to-one κ has fan-in 1 and keeps :math:`\kappa_0` on the diagonal.
            Mirrors the drift-averaging the volatile backend applies on value
            edges.
        volatility_fully_connected :
            Continuous layers only. If True (the default), κ is dense: every node
            of this layer modulates the volatility of every node of the child
            layer. If False, κ is diagonal (one-to-one), which requires equal
            sizes and is the only connectivity supported by the ``"unbounded"``
            volatility update.
        **kwargs :
            Per-layer overrides for any field of :class:`pyhgf.typing.LayerState`
            (e.g. ``mean``, ``precision``, ``expected_mean``, ``expected_precision``,
            ``mean_vol``, ``precision_vol``, ...) or :class:`pyhgf.typing.LayerParams`
            (``tonic_volatility_vol``, plus ``tonic_volatility`` on networks built
            with ``tonic_volatility=True``).
            Each value is broadcast to the layer's ``size``. Unknown names raise
            ``ValueError``. Defaults match ``LayerState.create`` and
            ``LayerParams.create``.

        Returns
        -------
        VectorisedDeepNetwork
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
            "continuous-state": "continuous",
        }
        kind = _kind_aliases.get(kind, kind)

        valid_kinds = {"volatile", "binary", "categorical", "continuous"}
        if kind not in valid_kinds:
            raise ValueError(
                f"Invalid layer kind '{kind}'. Choose from {sorted(valid_kinds)}."
            )

        # Continuous layers form their own network type: the sweeps, the edge
        # semantics (drift vs. direct prediction), and the entry point
        # (``input_data`` vs. ``fit``) all differ, so mixing is rejected.
        if self.layer_kinds and self._is_continuous != (kind == "continuous"):
            raise ValueError(
                "Continuous layers cannot mix with volatile/binary/"
                "categorical layers in one network."
            )

        # The continuous-only arguments all default to ``None`` so an explicit
        # value on another kind is an error rather than a silent no-op.
        if kind != "continuous":
            misplaced = [
                name
                for name, value in (
                    ("value_children", value_children),
                    ("volatility_children", volatility_children),
                    ("value_coupling", value_coupling),
                    ("volatility_coupling", volatility_coupling),
                    ("volatility_fully_connected", volatility_fully_connected),
                )
                if value is not None
            ]
            if misplaced:
                raise ValueError(
                    f"{', '.join(misplaced)} are only valid for continuous layers."
                )

        if kind == "continuous":
            if add_constant_input:
                raise ValueError(
                    "Continuous layers cannot use add_constant_input=True: the "
                    "per-node tonic_drift already provides a constant offset."
                )
            add_constant_input = False
        elif add_constant_input is None:
            add_constant_input = True

        value_coupling = 1.0 if value_coupling is None else value_coupling
        volatility_coupling = (
            1.0 if volatility_coupling is None else volatility_coupling
        )
        volatility_fully_connected = (
            True if volatility_fully_connected is None else volatility_fully_connected
        )

        value_child, volatility_child = self._resolve_continuous_children(
            kind=kind,
            size=size,
            value_children=value_children,
            volatility_children=volatility_children,
            volatility_fully_connected=volatility_fully_connected,
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
        # Parameter fields are kind-specific: reject an override the layer's
        # kind would silently ignore. ``tonic_volatility`` is shared between
        # kinds: continuous layers always carry it, volatile layers only when
        # the network-level flag allocates it.
        wrong_params = []
        for k in kwargs:
            if k not in _LAYER_PARAM_FIELDS:
                continue
            if k == "tonic_volatility" and kind == "volatile":
                if not self.tonic_volatility:
                    raise ValueError(
                        "A tonic_volatility override on a volatile layer "
                        "requires the network to carry the parameter: build "
                        "the DeepNetwork with tonic_volatility=True."
                    )
                continue
            if (k in _CONTINUOUS_PARAM_FIELDS) != (kind == "continuous"):
                wrong_params.append(k)
        if wrong_params:
            raise ValueError(
                f"Parameter override(s) {wrong_params} do not apply to {kind!r} layers."
            )

        if not fully_connected:
            if add_constant_input:
                raise ValueError(
                    "One-to-one layers (fully_connected=False) cannot use "
                    "add_constant_input=True."
                )
            # The value edge connects this layer to its value child — the
            # previous layer for the chain kinds, the resolved child for
            # continuous layers.
            child_for_size = (
                value_child
                if kind == "continuous"
                else (len(self.layer_sizes) - 1 if self.layer_sizes else None)
            )
            if child_for_size is not None and self.layer_sizes[child_for_size] != size:
                raise ValueError(
                    f"One-to-one layers require the same size as the child "
                    f"layer ({self.layer_sizes[child_for_size]}), got {size}."
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
        self.value_children.append(value_child)
        self.volatility_children.append(volatility_child)
        self.value_couplings.append(float(value_coupling))
        self.volatility_couplings.append(float(volatility_coupling))
        self.volatility_fully_connected.append(bool(volatility_fully_connected))
        # Eagerly rebuild the Network so ``self.state`` is always queryable
        # after construction. Cheap for typical depths; only triggers fresh
        # array allocations, no JIT trace.
        self.state = self._init_state()
        # Optimiser state's shape depends on the weights tuple, so invalidate
        # any previously initialised ``opt_state``.
        self.opt_state = None
        self._optimiser = None
        return self

    def _resolve_continuous_children(
        self,
        *,
        kind: str,
        size: int,
        value_children: Optional[int],
        volatility_children: Optional[int],
        volatility_fully_connected: bool,
    ) -> tuple[Optional[int], Optional[int]]:
        """Validate and resolve the child indices of a new continuous layer.

        Returns ``(value_child, volatility_child)`` — the indices of the layers the new
        layer is the value / volatility parent of, applying the chain default (previous
        layer becomes the value child) when neither parent argument names a layer
        explicitly.
        """
        if kind != "continuous":
            return None, None

        n_existing = len(self.layer_sizes)
        if value_children is None and volatility_children is None:
            value_child = n_existing - 1 if n_existing else None
        else:
            value_child = value_children
        volatility_child = volatility_children

        for name, child in (
            ("value_children", value_child),
            ("volatility_children", volatility_child),
        ):
            if child is None:
                continue
            if not 0 <= int(child) < n_existing:
                raise IndexError(
                    f"{name}={child} does not name an existing layer "
                    f"(network has {n_existing} layers)."
                )

        if value_child is not None and value_child in self.value_children:
            raise ValueError(
                f"Layer {value_child} already has a value parent "
                f"(layer {self.value_children.index(value_child)})."
            )
        if (
            volatility_child is not None
            and volatility_child in self.volatility_children
        ):
            raise ValueError(
                f"Layer {volatility_child} already has a volatility parent "
                f"(layer {self.volatility_children.index(volatility_child)})."
            )

        if volatility_child is not None:
            child_size = self.layer_sizes[volatility_child]
            if not volatility_fully_connected and child_size != size:
                raise ValueError(
                    f"A one-to-one volatility coupling requires equal sizes "
                    f"(child layer {volatility_child} has {child_size} nodes, "
                    f"this layer {size})."
                )
            if self.volatility_updates == "unbounded":
                # The unbounded update is derived for a single volatility
                # child per node and ignores value children.
                if volatility_fully_connected and not (size == 1 and child_size == 1):
                    raise ValueError(
                        "The 'unbounded' volatility update requires a "
                        "one-to-one volatility coupling "
                        "(volatility_fully_connected=False). Use 'standard' "
                        "or 'eHGF' for fully connected volatility parents."
                    )
                if value_child is not None:
                    raise ValueError(
                        "Under the 'unbounded' volatility update a volatility "
                        "parent cannot also be a value parent. Use 'standard' "
                        "or 'eHGF' for mixed parents."
                    )

        return value_child, volatility_child

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
        mismatch with the layer below, or no layer below), the layers are simply added
        one by one the usual way. The layer below may be the observation layer of any
        kind (volatile, binary, or categorical) since the stack sweeps treat their
        bottom boundary with the same input-leaf conventions as the unrolled path.

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
        VectorisedDeepNetwork
            Self for method chaining.
        """
        auto_scan = (
            kind == "volatile"
            and len(layer_sizes) >= _SCAN_AUTO_THRESHOLD
            and len(set(layer_sizes)) == 1
            and len(self.layer_sizes) > 0
            and self.layer_sizes[-1] == layer_sizes[0]
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
        """Initialise network with uniform weights, as an Equinox ``Network`` PyTree.

        All inter-layer weights are set to ``1.0``. Use :meth:`weight_initialisation`
        after construction to apply Xavier, He, orthogonal, or sparse initialisation.
        """
        if self._is_continuous:
            return self._init_continuous_state()

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
            if self.tonic_volatility and self.layer_kinds[i] == "volatile":
                # Pre-#409 default, restored behind the network-level flag.
                param_kwargs["tonic_volatility"] = -4.0
            for k, v in overrides.items():
                if k in _LAYER_PARAM_FIELDS:
                    param_kwargs[k] = v
            params = LayerParams.create(n_nodes=size, **param_kwargs)

            # `weights_mean` lives on the parent (this Layer); layer 0 has none.
            if i > 0:
                prev_size = self.layer_sizes[i - 1]
                n_parent_cols = size + (1 if self.add_constant_inputs[i] else 0)
                if self.fully_connected[i]:
                    weights_mean = jnp.ones((prev_size, n_parent_cols))
                else:
                    weights_mean = jnp.eye(prev_size, n_parent_cols)
            else:
                weights_mean = None

            coupling_fn = self.coupling_fns[i]
            layers.append(
                Layer(
                    state=state,
                    params=params,
                    weights_mean=weights_mean,
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
            update_input_layer=self.update_input_layer,
            predict_precision=self.predict_precision,
            feedforward_uncertainty=self.feedforward_uncertainty,
            mean_field_updates=self.mean_field_updates,
        )

    def _fan_in_matrix(
        self,
        child: Optional[int],
        size: int,
        strength: float,
        dense: bool,
    ) -> Optional[jnp.ndarray]:
        """Build one fixed coupling matrix of a continuous layer, or ``None``.

        Used for both W (``value_coupling``) and κ (``volatility_coupling``). Dense
        entries are ``strength / size``, so what the child receives — the drift or the
        phasic volatility — is the base strength times the *average* of this layer,
        invariant to its width. One-to-one edges have fan-in 1 and keep the base
        strength on the diagonal.
        """
        if child is None:
            return None
        child_size = self.layer_sizes[child]
        if dense:
            return jnp.full((child_size, size), strength / size)
        return strength * jnp.eye(child_size, size)

    def _init_continuous_state(self) -> Network:
        """Build the ``Network`` PyTree of an all-continuous network.

        Both coupling matrices are fixed at construction and share the fan-in rule of
        :meth:`_fan_in_matrix`.
        """
        n = len(self.layer_sizes)
        has_vol_parent = [False] * n
        for j in range(n):
            child = self.volatility_children[j]
            if child is not None:
                has_vol_parent[child] = True

        layers: list[Layer] = []
        for i, size in enumerate(self.layer_sizes):
            overrides = self.layer_overrides[i]

            state = LayerState.create_continuous(
                size, has_volatility_parent=has_vol_parent[i]
            )
            # Only fields the continuous state allocates can be overridden; the
            # internal-volatility fields stay ``None``.
            state_overrides = {
                k: jnp.full(size, v)
                for k, v in overrides.items()
                if k in _LAYER_STATE_FIELDS and getattr(state, k) is not None
            }
            if state_overrides:
                state = dataclasses.replace(state, **state_overrides)

            param_kwargs = {}
            if i == 0:
                # Observation-leaf convention, mirroring the nodalised backend
                # (``Network.input_idxs``): a clamped leaf performs no
                # autoregression of its own and its volatility exponent is
                # zeroed, so the observation noise is 1/exp(0) plus whatever a
                # volatility parent contributes. Overridable per layer.
                param_kwargs.update({
                    "autoconnection_strength": 0.0,
                    "tonic_volatility": 0.0,
                })
            param_kwargs.update({
                k: v for k, v in overrides.items() if k in _CONTINUOUS_PARAM_FIELDS
            })
            params = LayerParams.create_continuous(n_nodes=size, **param_kwargs)

            value_child = self.value_children[i]
            weights_mean = self._fan_in_matrix(
                child=value_child,
                size=size,
                strength=self.value_couplings[i],
                dense=self.fully_connected[i],
            )
            volatility_child = self.volatility_children[i]
            volatility_weights = self._fan_in_matrix(
                child=volatility_child,
                size=size,
                strength=self.volatility_couplings[i],
                dense=self.volatility_fully_connected[i],
            )

            layers.append(
                Layer(
                    state=state,
                    params=params,
                    weights_mean=weights_mean,
                    coupling_fn=self.coupling_fns[i],
                    add_constant_input=False,
                    has_volatility_parent=has_vol_parent[i],
                    is_input_layer=(i == 0),
                    fully_connected=self.fully_connected[i],
                    kind="continuous",
                    value_child_idx=value_child,
                    volatility_child_idx=volatility_child,
                    volatility_weights=volatility_weights,
                )
            )

        return Network(
            layers=tuple(layers),
            volatility_updates=self.volatility_updates,
            max_posterior_precision=self.max_posterior_precision,
            precision_clipping_value=self.precision_clipping_value,
            update_input_layer=self.update_input_layer,
            predict_precision=self.predict_precision,
            feedforward_uncertainty=self.feedforward_uncertainty,
            mean_field_updates=self.mean_field_updates,
        )

    def install_weight_belief(
        self, layers: Optional[Sequence[int]] = None
    ) -> "DeepNetwork":
        """Give every weight a belief, leaving any already installed untouched.

        The mean of each weight's belief is the weight itself; this adds the
        second parameter, the accumulated precision above the prior, as an
        array of zeros beside it (see
        :attr:`pyhgf.typing.vectorised.Layer.weights_precision_delta`). Zero
        precision_delta is the prior belief, so a network trained from a fresh install
        starts exactly where plain descent at the prior variance starts.

        ``learning_kind="synaptic_uncertainty"`` calls this itself, so it is
        only needed to install a belief ahead of time, or on part of a network.

        Parameters
        ----------
        layers :
            Which elements receive one, as indices into ``state.layers`` where
            ``0`` is the bottom layer. ``None`` (default) covers every element
            holding a weight matrix. The bottom layer holds none, so naming it
            raises.

        Returns
        -------
        DeepNetwork
            ``self``, with the beliefs installed.

        Raises
        ------
        ValueError
            If the network has no layers yet, or a named element holds no
            weight matrix.
        """
        if self.state is None:
            raise ValueError(
                "Network has no layers yet. Call add_layer(...) before "
                "install_weight_belief()."
            )
        selected = None if layers is None else {int(i) for i in layers}
        if selected is not None:
            holding = {
                i
                for i, elem in enumerate(self.state.layers)
                if elem.weights_mean is not None
            }
            unheld = sorted(selected - holding)
            if unheld:
                raise ValueError(
                    f"layers {unheld} hold no weight matrix, so they cannot carry "
                    f"a weight belief. Layers holding one: {sorted(holding)}."
                )
        elements = []
        for index, elem in enumerate(self.state.layers):
            if (
                elem.weights_mean is None
                or elem.weights_precision_delta is not None
                or (selected is not None and index not in selected)
            ):
                elements.append(elem)
                continue
            elements.append(
                dataclasses.replace(
                    elem, weights_precision_delta=jnp.zeros_like(elem.weights_mean)
                )
            )
        self.state = dataclasses.replace(self.state, layers=tuple(elements))
        return self

    def _ensure_optimiser_state(
        self, optimiser: Optional[optax.GradientTransformation]
    ) -> None:
        """Install optimiser state, keeping it when the optimiser is equivalent.

        The state is rebuilt only when there is none yet, or when the new optimiser's
        state has a *different structure* from the carried one.
        """
        if optimiser is None:
            return
        # Every caller has already built the network; a state-less network
        # cannot reach a learning step.
        assert self.state is not None
        weights = self.state.weights_tuple()
        if self.opt_state is None:
            self.opt_state = optimiser.init(weights)
        elif self._optimiser is not optimiser:
            # ``eval_shape`` traces the initialiser without allocating, so the
            # comparison costs nothing on the common path.
            candidate = jax.eval_shape(optimiser.init, weights)
            if jax.tree_util.tree_structure(candidate) != jax.tree_util.tree_structure(
                self.opt_state
            ):
                self.opt_state = optimiser.init(weights)
        self._optimiser = optimiser

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
        self._require_deep_backend("weight_initialisation")

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
            if elem.weights_mean is None:
                continue
            if isinstance(elem, LayerStack):
                # Stack has weights_mean shape (N, n_child, n_parent[+1]). Use
                # the same seed for every slice — matches the unrolled init's
                # "same seed across all layers" semantics (necessary for
                # byte-parity with the unrolled path; the underlying "all
                # layers identical at init" pattern is a separate concern).
                n_slices, n_children, n_parents = elem.weights_mean.shape
                per_slice = _init_matrix(
                    init_fn,
                    n_children,
                    n_parents,
                    elem.add_constant_input,
                    seed,
                    kwargs,
                )
                new_weights = jnp.broadcast_to(
                    per_slice, (n_slices, n_children, n_parents)
                )
            else:
                n_children, n_parents = elem.weights_mean.shape
                new_weights = _init_matrix(
                    init_fn,
                    n_children,
                    n_parents,
                    elem.add_constant_input,
                    seed,
                    kwargs,
                )
            new_elements[i] = dataclasses.replace(elem, weights_mean=new_weights)

        self.state = dataclasses.replace(self.state, layers=tuple(new_elements))
        return self

    def _resolve_learning(
        self, learning_kind: str, learning_kwargs: Optional[dict]
    ) -> tuple[str, Optional[SynapticUncertaintySettings]]:
        """Resolve a learning kind into the gradient it descends and its settings.

        ``'synaptic_uncertainty'`` names both the weight-belief rule and the
        gradient it descends, so the kind comes back unchanged either way. What
        selecting it adds is its settings, a step size taken from the belief
        rather than an optimiser, and the belief installed on the
        weight-carrying elements, which this does. Every other kind is a plain
        gradient and carries no settings.

        Parameters
        ----------
        learning_kind :
            The kind requested by :meth:`fit` or :meth:`batch_update`.
        learning_kwargs :
            Settings for the belief rule, resolved by
            :func:`~pyhgf.updates.vectorised.learning.resolve_synaptic_uncertainty_settings`.
            Only meaningful for ``'synaptic_uncertainty'``.

        Returns
        -------
        tuple
            ``(gradient_kind, settings)``, the second being ``None`` for every
            kind but ``'synaptic_uncertainty'``.
        """
        if learning_kind != "synaptic_uncertainty":
            if learning_kwargs:
                raise ValueError(
                    "learning_kwargs is only used by "
                    f"learning_kind='synaptic_uncertainty', not by {learning_kind!r}."
                )
            return learning_kind, None

        if self.mean_field_updates:
            raise ValueError(
                "learning_kind='synaptic_uncertainty' is derived from the "
                "relaxed updates (its curvature and evidence terms read the "
                "smoothing-corrected precisions) and is not available with "
                "mean_field_updates=True."
            )

        # Resolved before the belief is installed, so invalid settings raise
        # without leaving the network half-configured.
        settings = resolve_synaptic_uncertainty_settings(learning_kwargs)
        self.install_weight_belief()
        return learning_kind, settings

    def _require_deep_backend(self, method: str) -> None:
        """Reject deep-network entry points on a continuous network."""
        if self._is_continuous:
            raise ValueError(
                f"{method}() is not available for continuous networks: they "
                "are filters, not trainable deep networks. Use input_data() "
                "to run observations through the network."
            )

    def input_data(
        self,
        input_data: Union[np.ndarray, jnp.ndarray],
        time_steps: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        record: Optional[tuple] = None,
    ) -> "DeepNetwork":
        r"""Filter a sequence of observations through a continuous network.

        The vectorised equivalent of
        :meth:`pyhgf.model.network.Network.input_data`: for every observation,
        the top-down prediction sweep advances every layer's expected mean and
        precisions, the observation is clamped on layer 0, and the bottom-up
        sweep propagates prediction errors through the value- and
        volatility-coupling edges. Continuous networks only.

        Parameters
        ----------
        input_data :
            Observations for layer 0, shape ``(T, n_obs)`` (or ``(T,)`` when
            layer 0 has a single node).
        time_steps :
            Per-step time steps :math:`\Delta t`, shape ``(T,)``. Defaults to
            ``1.0`` everywhere.
        record :
            Tuple of ``LayerState`` field names to record at every time step,
            as in :meth:`fit`; the result lands in ``self.trajectories``.
            ``None`` (default) records nothing and only the per-step
            predictions are kept in ``self.predictions``.

        Returns
        -------
        DeepNetwork
            Self, with ``self.state`` advanced by the whole sequence.
        """
        if not self._is_continuous:
            raise ValueError(
                "input_data() is only available for continuous networks; use "
                "fit() for volatile/binary/categorical networks."
            )
        assert self.state is not None

        record_tuple: tuple = tuple(record) if record else ()
        if record_tuple:
            # Only fields this network actually allocates can be recorded — a
            # continuous layer leaves the internal-volatility fields at ``None``,
            # so asking for them would silently record nothing.
            valid_fields = {
                name
                for name in LayerState.__dataclass_fields__
                if any(
                    getattr(elem.state, name) is not None for elem in self.state.layers
                )
            }
            invalid = [f for f in record_tuple if f not in valid_fields]
            if invalid:
                raise ValueError(
                    f"Unknown record field(s) {invalid}. Valid: {sorted(valid_fields)}."
                )

        y = jnp.asarray(input_data)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"input_data has {y.shape[1]} features but layer 0 has "
                f"{self.layer_sizes[0]} nodes."
            )
        if time_steps is None:
            time_steps = jnp.ones(y.shape[0])
        else:
            time_steps = jnp.asarray(time_steps)

        self.state, step_output = run_continuous_scan(
            self.state, y, time_steps, record_tuple
        )

        if record_tuple:
            self.trajectories, self.predictions = step_output
        else:
            self.trajectories = None
            self.predictions = step_output

        return self

    def fit(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        optimiser: Optional[optax.GradientTransformation] = None,
        learning_kind: str = "precision_weighted",
        learning_kwargs: Optional[dict] = None,
        record: Optional[tuple] = None,
        weight_update: bool = True,
        time_step: float = 1.0,
        update_precisions: bool = True,
        check_gradient_health: bool = True,
    ) -> "DeepNetwork":
        r"""Fit network to data.

        Parameters
        ----------
        x :
            Input data, shape (n_samples, n_input_features).
        y :
            Target data, shape (n_samples, n_output_features).
        optimiser :
            Any ``optax.GradientTransformation`` — e.g. ``optax.sgd(0.2)``,
            ``optax.adam(1e-3)``, or any chain. Migration from the legacy
            ``lr=`` API: ``lr=0.2`` → ``optimiser=optax.sgd(0.2)``,
            ``lr="adam"`` → ``optimiser=optax.adam(1e-3)``.
        learning_kind :
            Gradient computation mode passed to
            :func:`~pyhgf.updates.vectorised.learning.learning_weights_vectorised`:
            ``"standard"`` or ``"precision_weighted"`` (default), both reading
            the settled beliefs, strictly local per edge.
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
        check_gradient_health :
            If True (default), probe the per-matrix gradient magnitudes on the
            last sample after training and raise :class:`DeadGradientWarning`
            when part of the network receives no learning signal.
        update_precisions :
            If True (default), the carried confidence fields (value-level posterior
            precision and the volatility level) evolve from sample to sample. If False, they
            are reset to their pre-fit values after every sample.
        time_step :
            Uniform inference time step :math:`\\Delta t` applied at every
            ``propagation_step``. Replaces the legacy ``net.state.time_step``
            attribute (``NetworkState`` no longer carries it). Default ``1.0``.

        Returns
        -------
        DeepNetwork
            Self with updated state and optimiser state.
        """
        self._require_deep_backend("fit")
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

        gradient_kind, synaptic_uncertainty_settings = self._resolve_learning(
            learning_kind, learning_kwargs
        )

        if synaptic_uncertainty_settings is None and optimiser is None:
            raise ValueError(
                f"learning_kind={learning_kind!r} needs an optimiser. Pass one, "
                "or use learning_kind='synaptic_uncertainty', whose step size "
                "is the belief's own variance."
            )
        self._ensure_optimiser_state(optimiser)

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        (self.state, self.opt_state), step_output = run_scan(
            (self.state, self.opt_state),
            (x, y),
            optimiser,
            gradient_kind,
            weight_update,
            record_tuple,
            float(time_step),
            update_precisions,
            synaptic_uncertainty_settings,
        )

        if record_tuple:
            self.trajectories, self.predictions = step_output
        else:
            self.trajectories = None
            self.predictions = step_output

        if check_gradient_health and weight_update:
            self._warn_if_gradients_dead(x[-1], y[-1], gradient_kind)

        return self

    def check_gradient_health(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        learning_kind: str = "precision_weighted",
        relative_floor: float = 1e-9,
    ) -> dict:
        """Measure the learning signal each weight matrix actually receives.

        Runs one prediction + update sweep on ``(x, y)`` from the current state
        (without mutating it), extracts the per-matrix child-side gradient factors,
        and flags matrices whose signal is zero or negligible relative to the
        largest. ``fit`` calls this automatically after training and warns.

        Parameters
        ----------
        x, y :
            One sample, shapes ``(n_input_features,)`` and ``(n_output_features,)``.
        learning_kind :
            The weight-gradient mode to probe — probe the one being trained with.
        relative_floor :
            A matrix counts as dead when its factor magnitude is exactly zero or
            below this fraction of the largest matrix's magnitude.

        Returns
        -------
        dict
            ``"magnitudes"``: per-element max ``|u|`` (``None`` for the bottom
            element). ``"dead"``: indices of matrices receiving no usable signal,
            by either detector — magnitude (exactly zero, or negligible relative
            to the largest) or quantisation (the child's belief displacement sits
            within a few float32 ulps of its mean scale, so the belief-side factor
            is rounding noise; belief-side kinds only). ``"quantised"``: the
            subset flagged by the quantisation detector.
        """
        swept = update_sweep(
            prediction_sweep(self.state, jnp.asarray(x)), jnp.asarray(y)
        )
        factors = _weight_quantities(swept, learning_kind)
        magnitudes = [
            None if f is None else float(np.abs(np.asarray(f[0])).max())
            for f in factors
        ]
        live = max((m for m in magnitudes if m is not None), default=0.0)
        dead = {
            i
            for i, m in enumerate(magnitudes)
            if m is not None and (m == 0.0 or m < live * relative_floor)
        }

        # Second detector: a displacement within a few float32 ulps of the mean
        # scale makes ``(mean - expected_mean)`` pure rounding noise, so
        # ``pe * precision`` returns quantisation garbage of healthy-looking
        # magnitude. Measured directly: at precision 1e7 the factors read ~0.3
        # while carrying no information.
        quantised: list = []
        eps32 = float(np.finfo(np.float32).eps)
        for i in range(1, len(swept.layers)):
            state = swept.layers[i - 1].state  # the child driving matrix i
            pe = np.abs(np.asarray(state.value_prediction_error))
            scale = np.maximum(
                np.abs(np.asarray(state.mean)),
                np.abs(np.asarray(state.expected_mean)),
            )
            ulps = pe / np.maximum(scale * eps32, 1e-300)
            if float(np.median(ulps)) < 8.0:
                quantised.append(i)
        combined = sorted(dead | set(quantised))

        return {"magnitudes": magnitudes, "dead": combined, "quantised": quantised}

    def _warn_if_gradients_dead(self, x, y, learning_kind: str) -> None:
        """Run the gradient health check and make a dead upper stack loud."""
        health = self.check_gradient_health(x, y, learning_kind=learning_kind)
        healthy = [
            i
            for i, m in enumerate(health["magnitudes"])
            if m is not None and m > 0.0 and i not in health["dead"]
        ]
        # Warn only on *partial* death — the head-only-training signature. Uniform
        # smallness (plain convergence) stays silent.
        if health["dead"] and healthy:
            warnings.warn(
                f"Weight matrices {health['dead']} receive essentially no "
                f"learning signal under learning_kind={learning_kind!r} while "
                f"others do: only part of the network is training, and the rest "
                f"acts as a frozen random-feature map. Likely causes: Bayesian "
                f"absorption of the error message across layers, or float32 "
                f"annihilation of (mean - expected_mean) * precision at high "
                f"precision. Lower the pinned precisions or revisit the "
                f"precision regime. Silence this check with "
                f"check_gradient_health=False.",
                DeadGradientWarning,
                stacklevel=3,
            )

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
        self._require_deep_backend("predict")
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
        self._require_deep_backend("prediction")
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
        optimiser: Optional[optax.GradientTransformation] = None,
        learning_kind: str = "precision_weighted",
    ) -> "DeepNetwork":
        """Run the bottom-up prediction-error + posterior-update sweep.

        Clamps the observations ``y`` on the output (bottom) layer, computes the leaf
        prediction error, then performs the interleaved posterior-update + prediction-
        error for every interior layer, in the correct bottom-up order. Belief states
        (means and precisions) are always updated.

        If an ``optimiser`` is supplied, a weight-learning phase runs *after* the belief
        sweep: PE-driven gradients are formed and applied to every inter-layer weight
        matrix, with the optimiser state held on ``self.opt_state``. With
        ``optimiser=None`` (default) the weights are left unchanged, exactly as
        :meth:`fit` would leave them with ``weight_update=False``.

        Typically called right after :meth:`prediction`, which sets up the predicted
        beliefs this sweep corrects, so a full local learning step without backprop is::

            net.prediction(x).update(y, optimiser=optax.sgd(1e-2))

        Parameters
        ----------
        y :
            Observations clamped on the bottom layer, shape ``(n_output_features,)``.
            Single sample only.
        optimiser :
            Optional ``optax.GradientTransformation``. If given, run the weight-learning
            phase and update ``self.opt_state``. If ``None``, only beliefs are updated.
        learning_kind :
            Weight-gradient mode used when ``optimiser`` is supplied. One of
            ``"standard"`` or ``"precision_weighted"`` (default). Both are
            strictly local. Each weight update reads only its own
            connection's prediction error, activation, and precision.

        Returns
        -------
        DeepNetwork
            Self, with ``self.state`` (and, if learning, ``self.opt_state``) advanced.
        """
        self._require_deep_backend("update")
        if self.state is None:
            raise ValueError("Add at least one layer before calling update.")
        y = jnp.asarray(y)
        if y.ndim != 1:
            raise ValueError("update() operates on a single sample (1D y).")
        self.state = update_sweep(self.state, y)

        if optimiser is not None:
            self._ensure_optimiser_state(optimiser)
            self.state, self.opt_state = learn_sweep(
                self.state, self.opt_state, optimiser, learning_kind
            )
        return self

    def input_error(self) -> jnp.ndarray:
        """Prediction error at the input (top) layer.

        The error message the input receives from the layer below — the child
        layer's prediction error, weighted by its precision and
        routed back through the connecting weights. It is read off the child,
        so it does not depend on whether ``update_input_layer`` is set.
        Typical usage runs the two sweeps first::

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
        self._require_deep_backend("input_error")
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
        self._require_deep_backend("predict_states")
        if self.state is None:
            raise ValueError("Add at least one layer before calling predict_states.")
        states = batched_prediction_states(self.state, jnp.asarray(x))
        return states[0].expected_mean, states

    def batch_update(
        self,
        x: Union[np.ndarray, jnp.ndarray],
        y: Union[np.ndarray, jnp.ndarray],
        optimiser: Optional[optax.GradientTransformation] = None,
        learning_kind: str = "precision_weighted",
        learning_kwargs: Optional[dict] = None,
        update_precisions: bool = True,
        time_step: float = 1.0,
        predicted: Optional[tuple] = None,
        sample_weight: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    ) -> "DeepNetwork":
        """One batch-synchronous learning step over many samples at once.

        Every sample in the batch is processed from the same state — same
        weights, same precisions — in parallel; the per-sample weight
        gradients and precision changes are then *averaged* and applied
        once, so the whole batch counts as a single observation. This
        differs from :meth:`fit`, which scans samples sequentially and lets
        the precisions adapt from one sample to the next (the natural mode
        for a time series). Use this method when samples are exchangeable —
        e.g. many token positions learned together.

        The per-sample errors at the input layer are stored on
        ``self.input_errors``, shape ``(batch, n_input_features)``. See
        :meth:`input_error` for their meaning and sign convention.

        Parameters
        ----------
        x :
            Predictors, shape ``(batch, n_input_features)``.
        y :
            Observations, shape ``(batch, n_output_features)``.
        optimiser :
            Optax optimiser for the weight step. ``None`` (default) freezes
            the weights.
        learning_kind :
            Weight-gradient mode, as in :meth:`fit`.
        update_precisions :
            If True (default), carry the batch-averaged change of the
            precision state (value-level precisions and the volatility
            level) into ``self.state``. Set to False to keep the precisions
            pinned, e.g. for exact comparisons against backpropagation.
        time_step :
            Inference time step, applied once per batch.
        predicted :
            Optional per-sample predicted states from :meth:`predict_states`;
            when given, the internal forward sweep is skipped and ``x`` is
            ignored.
        sample_weight :
            Optional per-sample weights, shape ``(batch,)``. The batch average
            becomes a weighted average with denominator ``sample_weight.sum()``,
            so padded or masked rows do not dilute the update. Pass a 0/1 mask
            for a variable-length batch. ``None`` keeps the plain mean.

        Returns
        -------
        DeepNetwork
            Self, with ``self.state`` (and, if learning, ``self.opt_state``)
            advanced by one batch.
        """
        self._require_deep_backend("batch_update")
        if self.state is None:
            raise ValueError("Add at least one layer before calling batch_update.")
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                "batch_update() operates on batches: x and y must be 2D "
                "(batch, n_features)."
            )

        gradient_kind, synaptic_uncertainty_settings = self._resolve_learning(
            learning_kind, learning_kwargs
        )

        self._ensure_optimiser_state(optimiser)

        self.state, self.opt_state, self.input_errors = batch_step(
            self.state,
            self.opt_state,
            x,
            y,
            optimiser=optimiser,
            learning_kind=gradient_kind,
            update_precisions=update_precisions,
            time_step=float(time_step),
            predicted=predicted,
            sample_weight=None if sample_weight is None else jnp.asarray(sample_weight),
            synaptic_uncertainty_settings=synaptic_uncertainty_settings,
        )
        return self

    def reset(self) -> "DeepNetwork":
        """Reset the network state."""
        self.state = self._init_state()
        self.opt_state = None
        self._optimiser = None
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
        self._optimiser = None
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
        return f"VectorisedDeepNetwork(nodes={self.n_nodes}, layers={self.layer_sizes})"
