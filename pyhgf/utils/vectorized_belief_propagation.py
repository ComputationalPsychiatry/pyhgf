# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized belief propagation step for deep predictive coding networks."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pyhgf.typing.vectorised import Layer, LayerStack, Network
from pyhgf.updates.vectorized.binary import (
    vectorized_binary_prediction,
    vectorized_binary_prediction_error,
)
from pyhgf.updates.vectorized.learning import vectorized_weight_gradient
from pyhgf.updates.vectorized.volatile import (
    vectorized_layer_posterior_update,
    vectorized_layer_prediction,
    vectorized_layer_prediction_error,
)

# ---------------------------------------------------------------------------
# Element-shape helpers
# ---------------------------------------------------------------------------


def _bottom_slice(stack: LayerStack):
    """Return ``(state, params, weights_in)`` of the *bottommost* stack slice."""
    state = jax.tree_util.tree_map(lambda x: x[0], stack.state)
    params = jax.tree_util.tree_map(lambda x: x[0], stack.params)
    return state, params, stack.weights_in[0]


def _top_slice(stack: LayerStack):
    """Return ``(state, params, weights_in)`` of the *topmost* stack slice."""
    state = jax.tree_util.tree_map(lambda x: x[-1], stack.state)
    params = jax.tree_util.tree_map(lambda x: x[-1], stack.params)
    return state, params, stack.weights_in[-1]


def _parent_view(elem):
    """Treat a ``Layer`` or ``LayerStack`` uniformly when acting as a parent.

    Returns ``(state, weights_in, coupling_fn, add_constant_input)``. The four pieces
    ``propagation_step`` needs to predict a child below.

    For a ``LayerStack``, the parent is the *bottommost* slice (the slice closest to the
    child below the stack).
    """
    if isinstance(elem, LayerStack):
        state, _, weights = _bottom_slice(elem)
        return state, weights, elem.coupling_fn, elem.add_constant_input
    return elem.state, elem.weights_in, elem.coupling_fn, elem.add_constant_input


def _child_view(elem):
    """Treat a ``Layer`` or ``LayerStack`` uniformly when acting as a child.

    Returns ``(state, kind, is_input_layer)``. What's needed when something above is
    doing a posterior update or computing PE-driven weight grads using this element's
    state as the child.

    For a ``LayerStack``, the child role is filled by the *topmost* slice (the slice
    closest to the parent above the stack).
    """
    if isinstance(elem, LayerStack):
        state, _, _ = _top_slice(elem)
        return state, elem.kind, False  # interior; never an input layer
    return elem.state, elem.kind, elem.is_input_layer


# ---------------------------------------------------------------------------
# Top-down prediction
# ---------------------------------------------------------------------------


def _predict_layer_from_parent(
    child: Layer,
    parent_state,
    parent_weights,
    parent_coupling_fn,
    parent_has_constant: bool,
    *,
    time_step: float,
    precision_clipping_value: float,
):
    """Predict a single ``Layer`` child from a parent view."""
    if child.kind == "binary":
        new_state = vectorized_binary_prediction(
            child_state=child.state,
            parent_state=parent_state,
            weights=parent_weights,
            coupling_fn=parent_coupling_fn,
            parent_has_constant=parent_has_constant,
            precision_clipping_value=precision_clipping_value,
        )
    else:
        new_state = vectorized_layer_prediction(
            child_state=child.state,
            parent_state=parent_state,
            weights=parent_weights,
            params=child.params,
            time_step=time_step,
            coupling_fn=parent_coupling_fn,
            parent_has_constant=parent_has_constant,
            has_volatility_parent=child.has_volatility_parent,
            is_input_layer=child.is_input_layer,
        )
    return dataclasses.replace(child, state=new_state)


def _predict_stack_from_parent(
    stack: LayerStack,
    parent_state,
    parent_weights,
    parent_coupling_fn,
    parent_has_constant: bool,
    *,
    time_step: float,
):
    """Top-down sweep over a ``LayerStack``.

    Step 1 (boundary): predict the topmost slice from the external parent. Using the
    parent's coupling/weights/bias (which may differ from the stack's. For ``(L, S)``
    they're the layer's; for ``(S, S)`` they're the parent stack's bottommost slice).

    Step 2 (scan): predict slice ``k`` from slice ``k+1`` for ``k = N-2 ... 0``
    using ``stack.weights_in[k+1]`` and the stack's own coupling_fn / bias.
    Scan in reverse so the carry threads top-to-bottom through the stack.
    """
    top_slice_state, top_slice_params, _ = _top_slice(stack)
    new_top_state = vectorized_layer_prediction(
        child_state=top_slice_state,
        parent_state=parent_state,
        weights=parent_weights,
        params=top_slice_params,
        time_step=time_step,
        coupling_fn=parent_coupling_fn,
        parent_has_constant=parent_has_constant,
        has_volatility_parent=stack.has_volatility_parent,
        is_input_layer=False,
    )

    n = stack.n_layers
    if n == 1:
        # Degenerate: stack with a single slice. Just write the new state.
        new_state = jax.tree_util.tree_map(
            lambda x, v: x.at[0].set(v), stack.state, new_top_state
        )
        return dataclasses.replace(stack, state=new_state)

    # xs: per-iteration data for predicting slices N-2 ... 0 from the slice above.
    # At step k, body(parent_state, xs[k]) → predict slice k. The "parent's
    # weights" used to predict slice k come from slice k+1 — i.e. stack.weights_in[k+1].
    xs_child_state = jax.tree_util.tree_map(lambda x: x[: n - 1], stack.state)
    xs_child_params = jax.tree_util.tree_map(lambda x: x[: n - 1], stack.params)
    xs_parent_weights = stack.weights_in[1:]  # shape (n-1, ...)

    def body(parent_state_carry, k_data):
        child_state, child_params, parent_weights_k = k_data
        new_child_state = vectorized_layer_prediction(
            child_state=child_state,
            parent_state=parent_state_carry,
            weights=parent_weights_k,
            params=child_params,
            time_step=time_step,
            coupling_fn=stack.coupling_fn,
            parent_has_constant=stack.add_constant_input,
            has_volatility_parent=stack.has_volatility_parent,
            is_input_layer=False,
        )
        return new_child_state, new_child_state

    _, new_states_below = jax.lax.scan(
        body,
        init=new_top_state,
        xs=(xs_child_state, xs_child_params, xs_parent_weights),
        reverse=True,
    )

    # new_states_below has shape (n-1, ...) for slices 0..n-2;
    # new_top_state is for slice n-1. Concatenate along axis 0.
    new_full_state = jax.tree_util.tree_map(
        lambda below, top: jnp.concatenate([below, top[None, ...]], axis=0),
        new_states_below,
        new_top_state,
    )
    return dataclasses.replace(stack, state=new_full_state)


def _topdown_predict(
    parent_elem, child_elem, *, time_step: float, precision_clipping_value: float = 1e-6
):
    """Predict ``child_elem`` from ``parent_elem``.

    Both can be Layer or LayerStack.
    """
    parent_state, parent_weights, parent_coupling_fn, parent_has_const = _parent_view(
        parent_elem
    )
    if isinstance(child_elem, LayerStack):
        # LayerStacks are continuous/volatile only — the binary clip never applies.
        return _predict_stack_from_parent(
            child_elem,
            parent_state,
            parent_weights,
            parent_coupling_fn,
            parent_has_const,
            time_step=time_step,
        )
    return _predict_layer_from_parent(
        child_elem,
        parent_state,
        parent_weights,
        parent_coupling_fn,
        parent_has_const,
        time_step=time_step,
        precision_clipping_value=precision_clipping_value,
    )


# ---------------------------------------------------------------------------
# Leaf PE (bottom element of the network)
# ---------------------------------------------------------------------------


def _leaf_pe(layer: Layer, *, volatility_updates: str, max_posterior_precision: float):
    """Compute the PE of the bottom layer (a ``Layer``; leaves can't be stacks)."""
    if layer.kind == "binary":
        new_state = vectorized_binary_prediction_error(layer=layer.state)
    else:
        new_state = vectorized_layer_prediction_error(
            layer=layer.state,
            params=layer.params,
            volatility_updates=volatility_updates,
            has_volatility_parent=layer.has_volatility_parent,
            max_posterior_precision=max_posterior_precision,
        )
    return dataclasses.replace(layer, state=new_state)


# ---------------------------------------------------------------------------
# Bottom-up posterior + PE
# ---------------------------------------------------------------------------


def _posterior_pe_layer(
    parent: Layer,
    child_state,
    child_is_input_layer: bool,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
):
    """Single-layer posterior update + PE."""
    new_state = vectorized_layer_posterior_update(
        layer=parent.state,
        child=child_state,
        weights=parent.weights_in,
        coupling_fn=parent.coupling_fn,
        parent_has_constant=parent.add_constant_input,
        max_posterior_precision=max_posterior_precision,
        child_is_input_layer=child_is_input_layer,
    )
    if parent.kind == "binary":
        new_state = vectorized_binary_prediction_error(layer=new_state)
    else:
        new_state = vectorized_layer_prediction_error(
            layer=new_state,
            params=parent.params,
            volatility_updates=volatility_updates,
            has_volatility_parent=parent.has_volatility_parent,
            max_posterior_precision=max_posterior_precision,
        )
    return dataclasses.replace(parent, state=new_state)


def _posterior_pe_stack(
    stack: LayerStack,
    child_state_init,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
):
    """Bottom-up sweep over a ``LayerStack`` (posterior update + PE per slice).

    Scan forward from slice 0 (bottommost) to slice N-1 (topmost). The carry is the
    just-PE'd child state below the current slice; on the first iteration it's the
    external ``child_state_init``.

    ``child_is_input_layer=False`` throughout — Phase 8 v1 requires the layer below a
    stack to be non-leaf, so the boundary is interior.
    """

    def body(child_carry_state, slice_data):
        slice_state, slice_params, slice_weights = slice_data
        new_state = vectorized_layer_posterior_update(
            layer=slice_state,
            child=child_carry_state,
            weights=slice_weights,
            coupling_fn=stack.coupling_fn,
            parent_has_constant=stack.add_constant_input,
            max_posterior_precision=max_posterior_precision,
            child_is_input_layer=False,
        )
        new_state = vectorized_layer_prediction_error(
            layer=new_state,
            params=slice_params,
            volatility_updates=volatility_updates,
            has_volatility_parent=stack.has_volatility_parent,
            max_posterior_precision=max_posterior_precision,
        )
        return new_state, new_state

    _, new_full_state = jax.lax.scan(
        body,
        init=child_state_init,
        xs=(stack.state, stack.params, stack.weights_in),
    )
    return dataclasses.replace(stack, state=new_full_state)


def _bottomup_posterior_pe(
    parent_elem,
    child_elem,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
):
    """Posterior update + PE for ``parent_elem`` using ``child_elem`` below."""
    child_state, _, child_is_input_layer = _child_view(child_elem)
    if isinstance(parent_elem, LayerStack):
        return _posterior_pe_stack(
            parent_elem,
            child_state,
            volatility_updates=volatility_updates,
            max_posterior_precision=max_posterior_precision,
        )
    return _posterior_pe_layer(
        parent_elem,
        child_state,
        child_is_input_layer,
        volatility_updates=volatility_updates,
        max_posterior_precision=max_posterior_precision,
    )


# ---------------------------------------------------------------------------
# Weight gradients
# ---------------------------------------------------------------------------


def _grad_layer(parent: Layer, child_elem, learning_kind: str):
    """Per-Layer weight gradient (same shape as ``parent.weights_in``)."""
    child_state, child_kind, _ = _child_view(child_elem)
    return vectorized_weight_gradient(
        parent_state=parent.state,
        child_state=child_state,
        coupling_fn=parent.coupling_fn,
        kind=learning_kind,
        parent_has_constant=parent.add_constant_input,
        child_is_binary=(child_kind == "binary"),
    )


def _grad_stack(stack: LayerStack, child_elem, learning_kind: str):
    """Per-slice weight gradients for a ``LayerStack``.

    The child of slice 0 is the layer below the stack (``child_elem``); the child of
    slice k>0 is slice k-1 within the stack. Pre-pend the external child's state to the
    stack's state along axis 0 to form an ``(N+1, ...)`` array, then ``vmap`` the per-
    slice grad over the N parent slices and the N child slots.
    """
    child_state, _, _ = _child_view(child_elem)

    combined_state = jax.tree_util.tree_map(
        lambda c, s: jnp.concatenate([c[None, ...], s], axis=0),
        child_state,
        stack.state,
    )
    # parent_states[k] = stack.state[k]; child_states[k] = combined[k]
    parent_states = stack.state
    child_states = jax.tree_util.tree_map(lambda x: x[:-1], combined_state)

    def per_slice(parent_state, child_state_for_slice):
        return vectorized_weight_gradient(
            parent_state=parent_state,
            child_state=child_state_for_slice,
            coupling_fn=stack.coupling_fn,
            kind=learning_kind,
            parent_has_constant=stack.add_constant_input,
            child_is_binary=False,
        )

    return jax.vmap(per_slice)(parent_states, child_states)


def _weight_grad(parent_elem, child_elem, learning_kind: str):
    """Weight gradient for ``parent_elem.weights_in``."""
    if isinstance(parent_elem, LayerStack):
        return _grad_stack(parent_elem, child_elem, learning_kind)
    return _grad_layer(parent_elem, child_elem, learning_kind)


# ---------------------------------------------------------------------------
# Element-level state writeback (for clamping x/y at the boundaries)
# ---------------------------------------------------------------------------


def _set_top_predictors(elem, x):
    """Clamp ``expected_mean`` and ``mean`` of the top element to ``x``.

    The top element must be a ``Layer`` (input layer).
    """
    if isinstance(elem, LayerStack):
        raise NotImplementedError("Top of network must be a Layer, not a LayerStack.")
    new_state = dataclasses.replace(elem.state, expected_mean=x, mean=x)
    return dataclasses.replace(elem, state=new_state)


def _set_bottom_observations(elem, y):
    """Clamp ``mean`` of the bottom element to ``y``.

    Must be a ``Layer`` (leaf).
    """
    if isinstance(elem, LayerStack):
        raise NotImplementedError(
            "Bottom of network must be a Layer, not a LayerStack."
        )
    new_state = dataclasses.replace(elem.state, mean=y)
    return dataclasses.replace(elem, state=new_state)


def _writeback_weights(elem, new_w):
    """Replace ``weights_in`` on a Layer or LayerStack with ``new_w``."""
    return dataclasses.replace(elem, weights_in=new_w)


# ---------------------------------------------------------------------------
# Top-level propagation step
# ---------------------------------------------------------------------------


def propagation_step(
    network: Network,
    opt_state: optax.OptState,
    inputs: tuple,
    *,
    optimizer: optax.GradientTransformation,
    time_step: float = 1.0,
    learning_kind: str = "precision_weighted",
    weight_update: bool = True,
) -> tuple[tuple[Network, optax.OptState], jnp.ndarray]:
    """Single propagation step through the network.

    Belief-propagation sweep (top-down prediction → leaf PE → interleaved
    posterior/PE bottom-up) followed by an optional weight-learning phase. Each step
    dispatches per element on ``Layer`` vs ``LayerStack``:

    * ``Layer`` → standard per-layer kernel call (unrolled).
    * ``LayerStack`` → ``jax.lax.scan`` over the stack's slices.

    Top and bottom elements must be ``Layer``s. A ``LayerStack``'s child below (and
    parent above) can themselves be ``Layer`` or ``LayerStack``; the stack-stack case
    requires the boundary widths to match.
    """
    x, y = inputs
    elements = list(network.layers)
    n_elements = len(elements)
    max_posterior_precision = network.max_posterior_precision
    volatility_updates = network.volatility_updates
    precision_clipping_value = network.precision_clipping_value

    # 1. Set predictors on the top element.
    elements[-1] = _set_top_predictors(elements[-1], x)

    # 2. Clamp observations on the bottom element.
    elements[0] = _set_bottom_observations(elements[0], y)

    # 3. Top-down prediction: predict each element from the one above.
    for i in range(n_elements - 1, 0, -1):
        elements[i - 1] = _topdown_predict(
            elements[i],
            elements[i - 1],
            time_step=time_step,
            precision_clipping_value=precision_clipping_value,
        )

    # 4a. PE on the bottom (leaf) element.
    elements[0] = _leaf_pe(
        elements[0],
        volatility_updates=volatility_updates,
        max_posterior_precision=max_posterior_precision,
    )

    # 4b. Interleaved bottom-up: posterior + PE on every interior element.
    for i in range(1, n_elements - 1):
        elements[i] = _bottomup_posterior_pe(
            elements[i],
            elements[i - 1],
            volatility_updates=volatility_updates,
            max_posterior_precision=max_posterior_precision,
        )

    # 5. Weight learning — same optax flow as before, but element-shaped grads.
    if weight_update:
        weights = tuple(elem.weights_in for elem in elements)
        grads_list: list = [None]  # bottom element has no weights_in
        for i in range(1, n_elements):
            grads_list.append(_weight_grad(elements[i], elements[i - 1], learning_kind))
        grads = tuple(grads_list)

        updates, new_opt_state = optimizer.update(grads, opt_state, weights)
        new_weights = optax.apply_updates(weights, updates)

        for i, new_w in enumerate(new_weights):
            if new_w is not None:
                elements[i] = _writeback_weights(elements[i], new_w)
    else:
        new_opt_state = opt_state

    new_network = dataclasses.replace(network, layers=tuple(elements))
    output_pred = new_network.layers[0].state.expected_mean
    return (new_network, new_opt_state), output_pred


# ---------------------------------------------------------------------------
# Scan driver + prediction-only sweep (unchanged contract)
# ---------------------------------------------------------------------------


@eqx.filter_jit
def run_scan(
    init_carry: tuple,
    inputs: tuple,
    optimizer: optax.GradientTransformation,
    learning_kind: str,
    weight_update: bool,
    record: tuple,
    time_step: float = 1.0,
) -> tuple:
    r"""Run ``jax.lax.scan`` over the belief-propagation step.

    Decorated with ``eqx.filter_jit``: arrays in ``init_carry`` / ``inputs``
    are dynamic; ``optimizer`` / ``learning_kind`` / ``weight_update`` /
    ``record`` / ``time_step`` are static and form the JIT cache key.

    Parameters
    ----------
    record :
        Tuple of ``LayerState`` field names to record at every time step (e.g.
        ``("expected_mean", "precision")``). An empty tuple disables recording. The scan
        output is just the per-step ``output_pred``. With a non-empty tuple, the
        per-step output is ``(traj_step, output_pred)`` where ``traj_step`` is
        ``dict[field_name, tuple[Array, ...]]`` (one per-element array per field, with
        ``LayerStack`` elements contributing arrays of shape ``(N, n_nodes)``). After
        ``scan`` stacks across time, each leaf carries a leading ``(T,)`` axis.
    time_step :
        Uniform inference time step :math:`\\Delta t` passed to every
        ``propagation_step`` call. Defaults to ``1.0``.

    Returns
    -------
    ``((final_network, final_opt_state), step_output)`` where ``step_output`` is either
    the stacked predictions alone (``record == ()``) or a
    ``(stacked_traj, stacked_predictions)`` tuple.
    """

    def _scan_body(carry, xs):
        network, opt_state = carry
        (new_network, new_opt_state), pred = propagation_step(
            network,
            opt_state,
            xs,
            optimizer=optimizer,
            time_step=time_step,
            learning_kind=learning_kind,
            weight_update=weight_update,
        )
        if record:
            traj_step = {
                field: tuple(
                    getattr(_state_for_record(elem), field)
                    for elem in new_network.layers
                )
                for field in record
            }
            return (new_network, new_opt_state), (traj_step, pred)
        return (new_network, new_opt_state), pred

    return jax.lax.scan(_scan_body, init_carry, inputs)


def _state_for_record(elem):
    """Return the ``LayerState`` to read trajectory fields from.

    For a ``Layer`` this is ``elem.state`` (shape ``(n_nodes,)`` per field). For a
    ``LayerStack`` it's the stacked state (shape ``(N, n_nodes)`` per field) — the user
    gets the whole stack's trajectory in one block.
    """
    return elem.state


@eqx.filter_jit
def prediction_pass(network: Network, x: jnp.ndarray) -> jnp.ndarray:
    """Forward-only sweep through the network (no PE / posterior / learning).

    Sets predictors on the top element and runs the top-down prediction pass; returns
    the bottom element's ``expected_mean``. Used by
    :meth:`pyhgf.model.DeepNetwork.predict`.
    """
    elements = list(network.layers)
    n_elements = len(elements)

    elements[-1] = _set_top_predictors(elements[-1], x)

    for i in range(n_elements - 1, 0, -1):
        elements[i - 1] = _topdown_predict(
            elements[i],
            elements[i - 1],
            time_step=1.0,
            precision_clipping_value=network.precision_clipping_value,
        )

    return elements[0].state.expected_mean
