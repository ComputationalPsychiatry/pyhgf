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
        vol = (lambda v: jnp.full(n_nodes, v)) if has_volatility_parent else (lambda v: None)
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
    """Per-layer static parameters.

    Each field is an array with one entry per node in the layer.

    Parameters
    ----------
    tonic_volatility :
        The tonic (baseline) volatility of the value level.
    tonic_volatility_vol :
        The tonic (baseline) volatility of the volatility level.
    volatility_coupling :
        The volatility-coupling strength between the value and volatility levels.
    autoconnection_strength_vol :
        The autoconnection (self-coupling) strength of the volatility level.
    """

    tonic_volatility: Array
    tonic_volatility_vol: Array
    volatility_coupling: Array
    autoconnection_strength_vol: Array

    @classmethod
    def create(
        cls,
        n_nodes: int,
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -4.0,
        volatility_coupling: float = 1.0,
        autoconnection_strength_vol: float = 1.0,
    ) -> "LayerParams":
        """Initialise layer params with defaults."""
        return cls(
            tonic_volatility=jnp.full(n_nodes, tonic_volatility),
            tonic_volatility_vol=jnp.full(n_nodes, tonic_volatility_vol),
            volatility_coupling=jnp.full(n_nodes, volatility_coupling),
            autoconnection_strength_vol=jnp.full(n_nodes, autoconnection_strength_vol),
        )


class Layer(eqx.Module):
    """One layer of the vectorised deep network.

    ``weights_in`` is the matrix connecting the layer *below* (child) into this layer
    (parent). The bottom layer (index 0) has ``weights_in=None`` because no layer sits
    below it. Shape: ``(n_child, n_self[+1])``; the optional ``+1`` column carries the
    bias when ``add_constant_input=True``.

    Parameters
    ----------
    state :
        The per-layer state (see :py:class:`LayerState`).
    params :
        The per-layer static parameters (see :py:class:`LayerParams`).
    weights_in :
        The matrix connecting the layer below (child) into this layer, or `None`
        for the bottom layer.
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
        The kind of layer, either ``"volatile"`` or ``"binary"``.
    """

    state: LayerState
    params: LayerParams
    weights_in: Optional[Array]
    coupling_fn: Callable = field(static=True)
    add_constant_input: bool = field(static=True)
    has_volatility_parent: bool = field(static=True)
    is_input_layer: bool = field(static=True)
    fully_connected: bool = field(static=True)
    kind: str = field(static=True)  # "volatile" | "binary"


class LayerStack(eqx.Module):
    """N identical layers stacked into one PyTree with a leading ``(N,)`` axis.

    ``state``/``params`` have leading axis ``N`` (each field shape goes from
    ``(n_nodes,)`` to ``(N, n_nodes)``). ``weights_in`` goes from
    ``(n_child, n_self[+1])`` to ``(N, n_child, n_self[+1])``. Slice index 0 is the
    *bottommost* slice in the stack (closest to layer 0 of the network); slice ``N-1``
    is the topmost.

    Validation constraints, enforced at build time:

    * The layer immediately below the stack must have the same node count as the stack
    width (so ``weights_in[0]`` shape matches).
    * ``weights_in[k]`` for k > 0 is a square ``(W, W+bias)`` block connecting slice k
    (parent) to slice k-1 (child) within the stack.

    Parameters
    ----------
    state :
        The stacked per-layer state, each field with a leading ``(N,)`` axis.
    params :
        The stacked per-layer static parameters, each field with a leading
        ``(N,)`` axis.
    weights_in :
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
        The kind of layer, either ``"volatile"`` or ``"binary"``.
    n_layers :
        The number of stacked layers ``N``.
    """

    state: LayerState  # each field shape: (N, n_nodes)
    params: LayerParams  # each field shape: (N, n_nodes)
    weights_in: Array  # shape: (N, n_child, n_self[+1])
    coupling_fn: Callable = field(static=True)
    add_constant_input: bool = field(static=True)
    has_volatility_parent: bool = field(static=True)
    fully_connected: bool = field(static=True)
    kind: str = field(static=True)
    n_layers: int = field(static=True)


def stack_layers(layers: list) -> LayerStack:
    """Combine N identical ``Layer`` instances into a single ``LayerStack``.

    All ``Layer``s must share static-field values (kind, coupling_fn,
    add_constant_input, has_volatility_parent, fully_connected) and have ``weights_in``
    of identical shape. Static fields are taken from the first layer; arrays are stacked
    along a new axis 0.

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
        if lay.weights_in is None:
            raise ValueError(
                f"layers[{k}] has weights_in=None (bottom layer of the network "
                f"can't be inside a LayerStack)."
            )
        if lay.weights_in.shape != first.weights_in.shape:
            raise ValueError(
                f"layers[{k}].weights_in.shape={lay.weights_in.shape} differs "
                f"from layers[0].weights_in.shape={first.weights_in.shape}."
            )

    stacked_state = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *(lay.state for lay in layers)
    )
    stacked_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *(lay.params for lay in layers)
    )
    stacked_weights = jnp.stack([lay.weights_in for lay in layers])

    return LayerStack(
        state=stacked_state,
        params=stacked_params,
        weights_in=stacked_weights,
        coupling_fn=first.coupling_fn,
        add_constant_input=first.add_constant_input,
        has_volatility_parent=first.has_volatility_parent,
        fully_connected=first.fully_connected,
        kind=first.kind,
        n_layers=len(layers),
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
    """

    layers: tuple
    volatility_updates: str = field(static=True)
    max_posterior_precision: float = field(static=True)
    precision_clipping_value: float = field(static=True, default=1e-6)

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
        """Per-element ``weights_in`` tuple, matched 1:1 to ``self.layers``."""
        return tuple(elem.weights_in for elem in self.layers)

    # ------------------------------------------------------------------
    # Legacy-shape views used by existing tests and the Rust-parity
    # cross-check. These are not used in the hot path — the kernels read
    # ``layer.state`` / ``layer.weights_in`` directly. For ``LayerStack``
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
                    out.append(elem.weights_in[k])
            elif elem.weights_in is not None:
                out.append(elem.weights_in)
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
