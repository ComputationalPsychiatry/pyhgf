# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Equinox PyTree types for the vectorised deep network."""

from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from jax import Array


class LayerState(eqx.Module):
    """Vectorised per-layer state, as an ``eqx.Module``.

    Each field is an array with one entry per node in the layer.

    Parameters
    ----------
    mean :
        The posterior mean of the value level.
    precision :
        The posterior precision of the value level.
    expected_mean :
        The predicted (expected) mean of the value level.
    expected_precision :
        The marginal predicted precision of the value level.
    conditional_expected_precision :
        The conditional predicted precision of the value level used by the
        structured-Gaussian (smoothing) update.
    effective_precision :
        The effective precision of the value-level prediction.
    value_prediction_error :
        The value prediction error of the value level.
    mean_vol :
        The posterior mean of the volatility level.
    precision_vol :
        The posterior precision of the volatility level.
    expected_mean_vol :
        The predicted (expected) mean of the volatility level.
    expected_precision_vol :
        The marginal predicted precision of the volatility level.
    effective_precision_vol :
        The effective precision of the volatility-level prediction.
    volatility_prediction_error :
        The volatility prediction error of the volatility level.
    """

    # Value level (external)
    mean: Array
    precision: Array
    expected_mean: Array
    expected_precision: Array
    conditional_expected_precision: Array
    effective_precision: Array
    value_prediction_error: Array
    # Volatility level (internal). ``None`` when the layer has no volatility
    # parent (see :meth:`create`).
    mean_vol: Optional[Array]
    precision_vol: Optional[Array]
    expected_mean_vol: Optional[Array]
    expected_precision_vol: Optional[Array]
    effective_precision_vol: Optional[Array]
    volatility_prediction_error: Optional[Array]

    @classmethod
    def create(cls, n_nodes: int, has_volatility_parent: bool = True) -> "LayerState":
        """Initialise a layer state with defaults.

        With ``has_volatility_parent=False`` the six volatility-level fields are
        set to ``None`` instead of being allocated. A frozen volatility level is
        never predicted or updated — every access to these fields sits behind a
        ``has_volatility_parent`` guard (see
        :func:`pyhgf.updates.vectorized.volatile.prediction` and
        :mod:`~pyhgf.updates.vectorized.volatile.prediction_error`) — so storing
        them would only carry dead arrays through the state. As ``None`` pytree
        nodes they hold no data and are skipped by every ``tree_map`` over the
        state (stacking, scanning, recording).
        """
        vol = (
            (lambda v: jnp.full(n_nodes, v))
            if has_volatility_parent
            else (lambda v: None)
        )
        return cls(
            mean=jnp.zeros(n_nodes),
            precision=jnp.ones(n_nodes),
            expected_mean=jnp.zeros(n_nodes),
            expected_precision=jnp.ones(n_nodes),
            conditional_expected_precision=jnp.ones(n_nodes),
            effective_precision=jnp.zeros(n_nodes),
            value_prediction_error=jnp.zeros(n_nodes),
            mean_vol=vol(0.0),
            precision_vol=vol(1.0),
            expected_mean_vol=vol(0.0),
            expected_precision_vol=vol(1.0),
            effective_precision_vol=vol(0.0),
            volatility_prediction_error=vol(0.0),
        )

    @classmethod
    def create_continuous(
        cls, n_nodes: int, has_volatility_parent: bool = False
    ) -> "LayerState":
        """Initialise the state of a layer of regular continuous nodes."""
        vope = jnp.zeros(n_nodes) if has_volatility_parent else None
        return cls(
            mean=jnp.zeros(n_nodes),
            precision=jnp.ones(n_nodes),
            expected_mean=jnp.zeros(n_nodes),
            expected_precision=jnp.ones(n_nodes),
            conditional_expected_precision=jnp.ones(n_nodes),
            effective_precision=jnp.zeros(n_nodes),
            value_prediction_error=jnp.zeros(n_nodes),
            mean_vol=None,
            precision_vol=None,
            expected_mean_vol=None,
            expected_precision_vol=None,
            effective_precision_vol=None,
            volatility_prediction_error=vope,
        )


# The six volatility-level fields of :class:`LayerState`, set to ``None`` on a
# layer without a volatility parent (see :meth:`LayerState.create`).
VOLATILITY_STATE_FIELDS: tuple = (
    "mean_vol",
    "precision_vol",
    "expected_mean_vol",
    "expected_precision_vol",
    "effective_precision_vol",
    "volatility_prediction_error",
)


class LayerParams(eqx.Module):
    r"""Per-layer static parameters.

    Each field is an array with one entry per node in the layer, or ``None`` when
    the field does not apply to the layer's kind: volatile layers carry
    ``tonic_volatility_vol`` (plus ``tonic_volatility`` when the value level's
    own tonic volatility is enabled — see ``DeepNetwork(tonic_volatility=True)``),
    continuous layers carry the other three.

    Parameters
    ----------
    tonic_volatility_vol :
        The tonic (baseline) volatility of the implied internal volatility level
        (volatile layers only).
    tonic_volatility :
        The tonic (baseline) log-volatility :math:`\omega` of the node's own
        Gaussian random walk. Continuous layers always carry it; volatile layers
        carry it only when enabled, and ``None`` means the value level has no
        intrinsic volatility at all.
    tonic_drift :
        The constant drift :math:`\rho` added to the predicted mean at every
        time step (continuous layers only).
    autoconnection_strength :
        The AR(1) coefficient :math:`\lambda \in [0, 1]` on the node's own mean
        in the prediction; ``1.0`` is a pure random walk (continuous layers
        only).
    """

    tonic_volatility_vol: Optional[Array] = None
    tonic_volatility: Optional[Array] = None
    tonic_drift: Optional[Array] = None
    autoconnection_strength: Optional[Array] = None

    @classmethod
    def create(
        cls,
        n_nodes: int,
        tonic_volatility_vol: float = -4.0,
        tonic_volatility: Optional[float] = None,
    ) -> "LayerParams":
        """Initialise volatile-layer params with defaults.

        ``tonic_volatility=None`` (the default) leaves the field structurally
        absent: the value level has no intrinsic volatility and diffuses only
        through its volatility parent.
        """
        return cls(
            tonic_volatility_vol=jnp.full(n_nodes, tonic_volatility_vol),
            tonic_volatility=(
                None
                if tonic_volatility is None
                else jnp.full(n_nodes, tonic_volatility)
            ),
        )

    @classmethod
    def create_continuous(
        cls,
        n_nodes: int,
        tonic_volatility: float = -4.0,
        tonic_drift: float = 0.0,
        autoconnection_strength: float = 1.0,
    ) -> "LayerParams":
        """Initialise continuous-layer params with the nodalised defaults."""
        return cls(
            tonic_volatility=jnp.full(n_nodes, tonic_volatility),
            tonic_drift=jnp.full(n_nodes, tonic_drift),
            autoconnection_strength=jnp.full(n_nodes, autoconnection_strength),
        )


class Layer(eqx.Module):
    r"""One layer of the vectorised deep network.

    ``weights_mean`` holds the *incoming* weights: the matrix connecting the layer
    *below* (child) into this layer (parent). The bottom layer (index 0) has
    ``weights_mean=None`` because no layer sits below it. Shape: ``(n_child, n_self[+1])``;
    the optional ``+1`` column carries the bias when ``add_constant_input=True``.

    Parameters
    ----------
    state :
        The per-layer state (see :py:class:`LayerState`).
    params :
        The per-layer static parameters (see :py:class:`LayerParams`).
    weights_mean :
        The incoming weights, i.e. the matrix connecting the layer below (child)
        into this layer, or `None` for the bottom layer. Also the mean of each
        weight's belief where one is installed.
    coupling_fn :
        The coupling function applied to the incoming weights.
    add_constant_input :
        Whether a constant (bias) input column is appended to the weights.
    has_volatility_parent :
        Whether the layer has a volatility parent.
    is_input_layer :
        Whether the layer is the input (bottom) layer of the network.
    fully_connected :
        Whether the incoming weights are fully connected.
    kind :
        The kind of layer, one of ``"volatile"``, ``"binary"``, ``"categorical"``,
        or ``"continuous"``.
    weights_precision_delta :
        Weight-belief precision, the second parameter of the belief each weight
        carries: ``weights_mean`` is the belief's mean and this its accumulated
        precision **above the prior**, same shape. The delta over the prior is
        stored rather than the precision itself because it starts at zero, so a
        per-step increment far below the prior precision accumulates exactly.
    value_child_idx :
        Continuous layers only — index (into ``Network.layers``) of the layer this
        layer is the *value parent* of, or ``None``. ``weights_mean`` then connects
        that child into this layer, shape ``(n_child, n_self)``, and enters the
        drift of the child's predicted mean. The chain convention of volatile
        networks (``weights_mean`` always connects the layer directly below) is a
        special case with ``value_child_idx = self_index - 1``.
    volatility_child_idx :
        Continuous layers only — index of the layer this layer is the *volatility
        parent* of, or ``None``. ``volatility_weights`` connects that child.
    volatility_weights :
        Volatility-coupling matrix :math:`\kappa`, shape ``(n_child, n_self)``,
        connecting the volatility child named by ``volatility_child_idx`` into
        this layer. Fixed at construction — never part of the learned weights
        (excluded from :meth:`Network.weights_tuple`).
    """

    state: LayerState
    params: LayerParams
    weights_mean: Optional[Array]
    coupling_fn: Callable = field(static=True)
    add_constant_input: bool = field(static=True)
    has_volatility_parent: bool = field(static=True)
    is_input_layer: bool = field(static=True)
    fully_connected: bool = field(static=True)
    kind: str = field(
        static=True
    )  # "volatile" | "binary" | "categorical" | "continuous"
    weights_precision_delta: Optional[Array] = None
    value_child_idx: Optional[int] = field(static=True, default=None)
    volatility_child_idx: Optional[int] = field(static=True, default=None)
    volatility_weights: Optional[Array] = None


class LayerStack(eqx.Module):
    """N identical layers stacked into one PyTree with a leading ``(N,)`` axis.

    ``state``/``params`` have leading axis ``N`` (each field shape goes from
    ``(n_nodes,)`` to ``(N, n_nodes)``). ``weights_mean`` goes from
    ``(n_child, n_self[+1])`` to ``(N, n_child, n_self[+1])``. Slice index 0 is the
    *bottommost* slice in the stack (closest to layer 0 of the network); slice ``N-1``
    is the topmost.

    Validation constraints, enforced at build time:

    * The layer immediately below the stack must have the same node count as the stack
    width (so ``weights_mean[0]`` shape matches).
    * ``weights_mean[k]`` for k > 0 is a square ``(W, W+bias)`` block connecting slice k
    (parent) to slice k-1 (child) within the stack.

    Parameters
    ----------
    state :
        The stacked per-layer state, each field with a leading ``(N,)`` axis.
    params :
        The stacked per-layer static parameters, each field with a leading
        ``(N,)`` axis.
    weights_mean :
        The stacked incoming weight matrices, shape ``(N, n_child, n_self[+1])``.
    coupling_fn :
        The coupling function shared by all stacked layers.
    add_constant_input :
        Whether a constant (bias) input column is appended to the weights.
    has_volatility_parent :
        Whether the layers have a volatility parent.
    fully_connected :
        Whether the incoming weights are fully connected.
    kind :
        The kind of layer, one of ``"volatile"``, ``"binary"``, or ``"categorical"``.
    n_layers :
        The number of stacked layers ``N``.
    """

    state: LayerState  # each field shape: (N, n_nodes)
    params: LayerParams  # each field shape: (N, n_nodes)
    weights_mean: Array  # shape: (N, n_child, n_self[+1])
    coupling_fn: Callable = field(static=True)
    add_constant_input: bool = field(static=True)
    has_volatility_parent: bool = field(static=True)
    fully_connected: bool = field(static=True)
    kind: str = field(static=True)
    n_layers: int = field(static=True)
    #: The stacked weight-belief precisions, shape ``(N, n_child, n_self[+1])``,
    #: or ``None`` when the stack carries no weight belief. Holds the precision_delta
    #: over the prior, exactly as :attr:`Layer.weights_precision_delta`.
    weights_precision_delta: Optional[Array] = None


def stack_layers(layers: list) -> LayerStack:
    """Combine N identical ``Layer`` instances into a single ``LayerStack``.

    All ``Layer``s must share static-field values (kind, coupling_fn,
    add_constant_input, has_volatility_parent, fully_connected) and have ``weights_mean``
    of identical shape. Static fields are taken from the first layer; arrays are stacked
    along a new axis 0.

    A ``LayerStack`` carries no DAG topology, so the continuous-layer fields
    (``value_child_idx``, ``volatility_child_idx``, ``volatility_weights``) have no
    counterpart here and continuous layers cannot be stacked.

    Parameters
    ----------
    layers :
        The list of identical ``Layer`` instances to stack.

    Returns
    -------
    layer_stack :
        The combined :py:class:`LayerStack`.
    """
    if not layers:
        raise ValueError("Cannot stack an empty list of Layers.")
    first = layers[0]
    for k, lay in enumerate(layers):
        if not isinstance(lay, Layer):
            raise TypeError(f"layers[{k}] is not a Layer: {type(lay).__name__}")
        for attr in (
            "add_constant_input",
            "has_volatility_parent",
            "fully_connected",
            "kind",
        ):
            if getattr(lay, attr) != getattr(first, attr):
                raise ValueError(
                    f"Cannot stack layers with differing static field {attr!r}: "
                    f"layers[0].{attr}={getattr(first, attr)!r}, "
                    f"layers[{k}].{attr}={getattr(lay, attr)!r}."
                )
        if lay.coupling_fn is not first.coupling_fn:
            raise ValueError(
                f"Cannot stack layers with differing coupling_fn identities. "
                f"Hoist the function to module scope so all layers share it."
            )
        if lay.weights_mean is None:
            raise ValueError(
                f"layers[{k}] has weights_mean=None (bottom layer of the network "
                f"can't be inside a LayerStack)."
            )
        if lay.weights_mean.shape != first.weights_mean.shape:
            raise ValueError(
                f"layers[{k}].weights_mean.shape={lay.weights_mean.shape} differs "
                f"from layers[0].weights_mean.shape={first.weights_mean.shape}."
            )

    stacked_state = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *(lay.state for lay in layers)
    )
    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *(lay.params for lay in layers)
    )
    stacked_weights = jnp.stack([lay.weights_mean for lay in layers])
    # A belief is stacked only when every slice carries one, since the stacked
    # field is one array across the whole stack.
    carries_belief = [lay.weights_precision_delta is not None for lay in layers]
    if any(carries_belief) and not all(carries_belief):
        raise ValueError(
            "Cannot stack layers where only some carry a weight belief: "
            f"{sum(carries_belief)} of {len(layers)} have "
            "weights_precision_delta. Install it on every layer or none."
        )
    stacked_precision = (
        jnp.stack([lay.weights_precision_delta for lay in layers])
        if all(carries_belief)
        else None
    )

    return LayerStack(
        state=stacked_state,
        params=stacked_params,
        weights_mean=stacked_weights,
        coupling_fn=first.coupling_fn,
        add_constant_input=first.add_constant_input,
        has_volatility_parent=first.has_volatility_parent,
        fully_connected=first.fully_connected,
        kind=first.kind,
        n_layers=len(layers),
        weights_precision_delta=stacked_precision,
    )


class Network(eqx.Module):
    """Complete vectorised network state.

    ``time_step`` is *not* stored on the network — it is passed as a per-step input to
    ``propagation_step``, matching the nodalised backend's
    ``input_data(time_steps=...)`` API.

    Optimiser state lives in a separate ``optax`` opt-state carried alongside
    ``Network`` in the scan carry; it is not part of the network PyTree.

    ``layers`` is a mixed tuple of ``Layer`` and ``LayerStack`` elements.

    Parameters
    ----------
    layers :
        A mixed tuple of ``Layer`` and ``LayerStack`` elements, ordered from the
        bottom (input) layer to the top.
    volatility_updates :
        The volatility update scheme, e.g. ``"unbounded"``.
    max_posterior_precision :
        The maximum posterior precision used to clip the precision updates.
    update_input_layer :
        Whether the sweeps reach the top (input) layer — see
        :class:`pyhgf.model.DeepNetwork`.
    predict_precision :
        Whether the prediction sweep advances the precisions — see
        :func:`pyhgf.updates.vectorized.volatile.prediction.vectorized_layer_prediction`.
    feedforward_uncertainty :
        Whether value parents propagate their uncertainty to their children's
        predicted precision — see :class:`pyhgf.model.DeepNetwork`.
    mean_field_updates :
        If ``False`` (default), use the relaxed prediction and posterior updates.
        If ``True``, use the original mean-field updates — see
        :class:`pyhgf.model.DeepNetwork`.
    """

    layers: tuple
    volatility_updates: str = field(static=True)
    max_posterior_precision: float = field(static=True)
    precision_clipping_value: float = field(static=True, default=1e-6)
    update_input_layer: bool = field(static=True, default=False)
    predict_precision: bool = field(static=True, default=True)
    feedforward_uncertainty: bool = field(static=True, default=False)
    mean_field_updates: bool = field(static=True, default=False)

    @property
    def n_layers(self) -> int:
        """Number of *elements* (``Layer`` or ``LayerStack``) in the network.

        A ``LayerStack`` counts as one element; use ``n_total_slices`` for the number of
        unrolled layers.
        """
        return len(self.layers)

    @property
    def n_total_slices(self) -> int:
        """Total unrolled layer count, expanding every ``LayerStack``."""
        return sum(
            (e.n_layers if isinstance(e, LayerStack) else 1) for e in self.layers
        )

    def get_layer_sizes(self) -> list[int]:
        """Per-element node count (one entry per ``Layer`` / ``LayerStack``)."""
        out = []
        for elem in self.layers:
            if isinstance(elem, LayerStack):
                out.append(elem.state.mean.shape[1])  # (N, n_nodes) -> n_nodes
            else:
                out.append(elem.state.mean.shape[0])
        return out

    def weights_tuple(self) -> tuple:
        """Per-element ``weights_mean`` tuple, matched 1:1 to ``self.layers``."""
        return tuple(elem.weights_mean for elem in self.layers)

    # ------------------------------------------------------------------
    # Legacy-shape views used by existing tests and the Rust-parity
    # cross-check. These are not used in the hot path — the kernels read
    # ``layer.state`` / ``layer.weights_mean`` directly. For ``LayerStack``
    # elements these views flatten the stack into its constituent slices
    # so consumers see the unrolled shape.
    # ------------------------------------------------------------------
    @property
    def weights(self) -> tuple:
        """Tuple of weight matrices (legacy view).

        Stacks are flattened.         Each entry is a ``(n_child, n_self[+1])`` array.
        The ``None`` slot on layer 0 is         dropped, and any ``LayerStack`` is
        expanded slice-by-slice.
        """
        out = []
        for elem in self.layers:
            if isinstance(elem, LayerStack):
                for k in range(elem.n_layers):
                    out.append(elem.weights_mean[k])
            elif elem.weights_mean is not None:
                out.append(elem.weights_mean)
        return tuple(out)

    @property
    def params(self) -> tuple:
        """Per-layer ``LayerParams`` tuple."""
        out = []
        for elem in self.layers:
            if isinstance(elem, LayerStack):
                for k in range(elem.n_layers):
                    out.append(jax.tree_util.tree_map(lambda x, k=k: x[k], elem.params))
            else:
                out.append(elem.params)
        return tuple(out)


# Convenience constant: every ``LayerState`` field, ordered as declared. Pass
# to ``DeepNetwork.fit(record=RECORD_ALL)`` for the legacy "record everything"
# behaviour without enumerating the field list at the call site.
RECORD_ALL: tuple = tuple(LayerState.__dataclass_fields__.keys())
