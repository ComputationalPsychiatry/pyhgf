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
    """Vectorised per-layer state, as an ``eqx.Module``."""

    # Value level (external)
    mean: Array
    precision: Array
    expected_mean: Array
    expected_precision: Array
    conditional_expected_precision: Array
    effective_precision: Array
    value_prediction_error: Array
    # Volatility level (internal)
    mean_vol: Array
    precision_vol: Array
    expected_mean_vol: Array
    expected_precision_vol: Array
    effective_precision_vol: Array
    volatility_prediction_error: Array

    @classmethod
    def create(cls, n_nodes: int) -> "LayerState":
        """Initialise a layer state with defaults."""
        return cls(
            mean=jnp.zeros(n_nodes),
            precision=jnp.ones(n_nodes),
            expected_mean=jnp.zeros(n_nodes),
            expected_precision=jnp.ones(n_nodes),
            conditional_expected_precision=jnp.ones(n_nodes),
            effective_precision=jnp.zeros(n_nodes),
            value_prediction_error=jnp.zeros(n_nodes),
            mean_vol=jnp.zeros(n_nodes),
            precision_vol=jnp.ones(n_nodes),
            expected_mean_vol=jnp.zeros(n_nodes),
            expected_precision_vol=jnp.ones(n_nodes),
            effective_precision_vol=jnp.zeros(n_nodes),
            volatility_prediction_error=jnp.zeros(n_nodes),
        )


class LayerParams(eqx.Module):
    """Per-layer static parameters."""

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
    """

    layers: tuple
    update_type: str = field(static=True)
    max_posterior_precision: float = field(static=True)

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
