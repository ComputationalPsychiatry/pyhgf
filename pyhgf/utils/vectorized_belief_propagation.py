# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized belief propagation step for deep predictive coding networks."""

from __future__ import annotations

import dataclasses
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pyhgf.typing.vectorised import (
    VOLATILITY_STATE_FIELDS as _VOL_STATE_FIELDS,
)
from pyhgf.typing.vectorised import (
    Layer,
    LayerStack,
    Network,
)
from pyhgf.updates.vectorized.binary import (
    vectorized_binary_prediction,
    vectorized_binary_prediction_error,
)
from pyhgf.updates.vectorized.categorical import (
    vectorized_categorical_prediction,
    vectorized_categorical_prediction_error,
)
from pyhgf.updates.vectorized.learning import (
    vectorized_weight_gradient,
    vectorized_weight_gradient_factors,
)
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
    elif child.kind == "categorical":
        new_state = vectorized_categorical_prediction(
            child_state=child.state,
            parent_state=parent_state,
            weights=parent_weights,
            coupling_fn=parent_coupling_fn,
            parent_has_constant=parent_has_constant,
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
    elif layer.kind == "categorical":
        new_state = vectorized_categorical_prediction_error(layer=layer.state)
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


def _match_child_vol_structure(child_state, has_volatility_parent):
    """Align a child state's volatility fields to a consumer's volatility structure.

    A layer without a volatility parent stores its six volatility fields as
    ``None`` rather than arrays. Where such a child meets a ``LayerStack`` with a
    different volatility structure — a ``scan`` carry seeded by the child, or a
    concatenation of the child onto the stack — the two pytrees must match.

    Reconciling them here is value-neutral: cross-layer coupling is value-only,
    so a parent update never reads its child's volatility level (that level is
    internal to each layer). Materialising zero volatility fields when the
    consumer has them, or dropping to ``None`` when it does not, only fixes the
    structure; no volatility quantity of the child is ever consumed.
    """
    if has_volatility_parent:
        n = child_state.mean.shape[-1]
        repl = {
            f: (
                jnp.zeros(n)
                if getattr(child_state, f) is None
                else getattr(child_state, f)
            )
            for f in _VOL_STATE_FIELDS
        }
    else:
        repl = {f: None for f in _VOL_STATE_FIELDS}
    return dataclasses.replace(child_state, **repl)


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
    # The scan carry becomes a stack slice each step, so seed it with the
    # child's state coerced to the stack's volatility structure.
    child_state_init = _match_child_vol_structure(
        child_state_init, stack.has_volatility_parent
    )

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


def _layer_weight_op(kernel, parent: Layer, child_elem, learning_kind: str):
    """Apply a per-layer weight kernel to a ``Layer`` parent and its child."""
    child_state, child_kind, _ = _child_view(child_elem)
    return kernel(
        parent_state=parent.state,
        child_state=child_state,
        coupling_fn=parent.coupling_fn,
        kind=learning_kind,
        parent_has_constant=parent.add_constant_input,
        child_is_binary=(child_kind == "binary"),
    )


def _stack_weight_op(kernel, stack: LayerStack, child_elem, learning_kind: str):
    """Apply a per-layer weight kernel to every slice of a ``LayerStack``.

    The child of slice 0 is the layer below the stack (``child_elem``); the child of
    slice k>0 is slice k-1 within the stack. Pre-pend the external child's state to the
    stack's state along axis 0 to form an ``(N+1, ...)`` array, then ``vmap`` the kernel
    over the N parent slices (``stack.state``) and the N child slots
    (``combined[:-1]``).
    """
    child_state, _, _ = _child_view(child_elem)
    child_state = _match_child_vol_structure(child_state, stack.has_volatility_parent)

    combined_state = jax.tree_util.tree_map(
        lambda c, s: jnp.concatenate([c[None, ...], s], axis=0),
        child_state,
        stack.state,
    )
    child_states = jax.tree_util.tree_map(lambda x: x[:-1], combined_state)

    def per_slice(parent_state, child_state_for_slice):
        return kernel(
            parent_state=parent_state,
            child_state=child_state_for_slice,
            coupling_fn=stack.coupling_fn,
            kind=learning_kind,
            parent_has_constant=stack.add_constant_input,
            child_is_binary=False,
        )

    return jax.vmap(per_slice)(stack.state, child_states)


def _weight_op(kernel, parent_elem, child_elem, learning_kind: str):
    """Dispatch a per-layer weight kernel on ``Layer`` vs ``LayerStack``."""
    if isinstance(parent_elem, LayerStack):
        return _stack_weight_op(kernel, parent_elem, child_elem, learning_kind)
    return _layer_weight_op(kernel, parent_elem, child_elem, learning_kind)


def _weight_grad(parent_elem, child_elem, learning_kind: str):
    """Weight gradient (same shape as ``parent_elem.weights_in``)."""
    return _weight_op(
        vectorized_weight_gradient, parent_elem, child_elem, learning_kind
    )


def _weight_factor(parent_elem, child_elem, learning_kind: str):
    """Rank-one gradient factors ``(u, v)`` for ``parent_elem.weights_in``."""
    return _weight_op(
        vectorized_weight_gradient_factors, parent_elem, child_elem, learning_kind
    )


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

    Parameters
    ----------
    network :
        The current vectorised network state.
    opt_state :
        The current optax optimiser state.
    inputs :
        A tuple ``(x, y)`` with the predictors set on the top element and the
        observations clamped on the bottom element.
    optimizer :
        The optax optimiser used for the weight-learning phase.
    time_step :
        The time elapsed since the previous step.
    learning_kind :
        The weight-gradient mode passed to
        :py:func:`pyhgf.updates.vectorized.learning.vectorized_weight_gradient`.
    weight_update :
        Whether to apply the weight-learning phase after belief propagation.

    Returns
    -------
    carry :
        A tuple ``((network, opt_state), surprise)`` where `network` and
        `opt_state` are updated and `surprise` is the step's surprise.
    """
    x, y = inputs

    # Belief propagation: top-down prediction (clamping x on top) then the
    # bottom-up PE + posterior sweep (clamping y at the bottom).
    swept = _update_sweep(_prediction_sweep(network, x, time_step=time_step), y)

    # Optional weight-learning phase.
    if weight_update:
        new_network, new_opt_state = _learn_sweep(
            swept, opt_state, optimizer, learning_kind
        )
    else:
        new_network, new_opt_state = swept, opt_state

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
    init_carry :
        The initial scan carry, a tuple ``(network, opt_state)``.
    inputs :
        The per-step inputs scanned over, a tuple of predictor/observation arrays
        with a leading time axis.
    optimizer :
        The optax optimiser used for the weight-learning phase.
    learning_kind :
        The weight-gradient mode passed to
        :py:func:`pyhgf.updates.vectorized.learning.vectorized_weight_gradient`.
    weight_update :
        Whether to apply the weight-learning phase at every step.
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


def _prediction_sweep(
    network: Network, x: jnp.ndarray, *, time_step: float = 1.0
) -> Network:
    """Top-down prediction sweep, returning the updated network.

    Clamps the predictors on the top element and predicts every element from the one
    above. No prediction errors, posterior updates, or weight learning are performed.
    """
    elements = list(network.layers)
    n_elements = len(elements)

    elements[-1] = _set_top_predictors(elements[-1], x)

    for i in range(n_elements - 1, 0, -1):
        elements[i - 1] = _topdown_predict(
            elements[i],
            elements[i - 1],
            time_step=time_step,
            precision_clipping_value=network.precision_clipping_value,
        )

    return dataclasses.replace(network, layers=tuple(elements))


def _update_sweep(network: Network, y: jnp.ndarray) -> Network:
    """Bottom-up prediction-error + posterior-update sweep, returning the network.

    Clamps the observations on the bottom element, computes the leaf prediction error,
    then performs the interleaved posterior-update + PE for every interior element, in
    bottom-up order. Belief states are updated; inter-layer weights are not.
    """
    elements = list(network.layers)
    n_elements = len(elements)

    # Clamp observations and compute the leaf prediction error.
    elements[0] = _set_bottom_observations(elements[0], y)
    elements[0] = _leaf_pe(
        elements[0],
        volatility_updates=network.volatility_updates,
        max_posterior_precision=network.max_posterior_precision,
    )

    # Interleaved bottom-up posterior + PE on every interior element.
    for i in range(1, n_elements - 1):
        elements[i] = _bottomup_posterior_pe(
            elements[i],
            elements[i - 1],
            volatility_updates=network.volatility_updates,
            max_posterior_precision=network.max_posterior_precision,
        )

    return dataclasses.replace(network, layers=tuple(elements))


@eqx.filter_jit
def prediction_sweep(network: Network, x: jnp.ndarray) -> Network:
    """JIT-compiled top-down prediction sweep.

    See :func:`_prediction_sweep`.
    """
    return _prediction_sweep(network, x)


@eqx.filter_jit
def update_sweep(network: Network, y: jnp.ndarray) -> Network:
    """JIT-compiled bottom-up PE + posterior sweep.

    See :func:`_update_sweep`.
    """
    return _update_sweep(network, y)


def _input_prediction_error(network: Network) -> jnp.ndarray:
    r"""Prediction error routed to the network's input (top) layer.

    The bottom-up sweep (:func:`_update_sweep`) never touches the top layer:
    its values are clamped to the predictors ``x``, so it receives no
    posterior update and no prediction error. This function computes the
    error message the top layer *would* receive from the layer below it —
    the same confidence-weighted quantity that drives the posterior mean
    shift of every interior layer:

    .. math::

        \varepsilon_x = g'(\hat{\mu}_x) \odot W^\top (g_a \, \delta_a),

    where :math:`\delta_a` is the child layer's value prediction error,
    :math:`g_a` its smoothing gain (the same gain used by
    :func:`pyhgf.updates.vectorized.volatile.posterior.vectorized_posterior_update_mean_value_level`),
    :math:`W` the weight matrix connecting the child into the top layer
    (bias column excluded), and :math:`g'` the derivative of the top layer's
    coupling function at the clamped predictors.

    With unit precisions and an identity coupling this reduces to
    :math:`W^\top \delta_a` — the error multiplied back through the weights.
    Because prediction errors follow the ``observed - predicted`` convention,
    the result is the *negative* of the gradient of a squared-error loss with
    respect to the predictors.

    Must be called after the update sweep, so the child layer carries its
    posterior prediction error.

    Parameters
    ----------
    network :
        The network state, after :func:`_update_sweep`.

    Returns
    -------
    jnp.ndarray
        The prediction error at the top layer, shape ``(n_input_features,)``.
    """
    top = network.layers[-1]
    if isinstance(top, LayerStack):
        raise NotImplementedError("Top of network must be a Layer, not a LayerStack.")
    if top.weights_in is None:
        raise ValueError(
            "The network has a single layer: there is no layer below the "
            "input layer to route an error from."
        )
    child_state, _, _ = _child_view(network.layers[-2])

    weights = top.weights_in
    if top.add_constant_input:
        # The bias column connects the constant node, not a real input.
        weights = weights[:, :-1]

    # Smoothing gain of the child layer — identical to the gain used by the
    # interior posterior mean update, so the top layer sees exactly the
    # message any interior layer would see.
    pi_y = child_state.precision - child_state.expected_precision
    gain = (
        child_state.conditional_expected_precision
        * child_state.precision
        / (child_state.conditional_expected_precision + pi_y)
    )

    coupling_prime = jax.vmap(jax.grad(top.coupling_fn))(top.state.expected_mean)
    return (
        jnp.matmul(weights.T, gain * child_state.value_prediction_error)
        * coupling_prime
    )


@eqx.filter_jit
def input_prediction_error(network: Network) -> jnp.ndarray:
    """JIT-compiled prediction error at the input (top) layer.

    See :func:`_input_prediction_error`.
    """
    return _input_prediction_error(network)


def _weight_gradients(network: Network, learning_kind: str) -> tuple:
    """Per-element weight gradients, without applying them.

    Must run *after* :func:`_update_sweep`, so the per-layer states already carry their
    prediction errors / posteriors. Returns one gradient per element, matched 1:1 to
    ``network.layers`` (``None`` for the bottom element, which has no incoming weights).
    Gradients are in descent form, ready for optax.
    """
    elements = network.layers
    grads_list: list = [None]  # bottom element has no weights_in
    for i in range(1, len(elements)):
        grads_list.append(_weight_grad(elements[i], elements[i - 1], learning_kind))
    return tuple(grads_list)


def _apply_weight_updates(
    network: Network,
    grads: tuple,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[Network, optax.OptState]:
    """One optimiser step on every ``weights_in``, from precomputed gradients."""
    elements = list(network.layers)
    weights = tuple(elem.weights_in for elem in elements)

    updates, new_opt_state = optimizer.update(grads, opt_state, weights)
    new_weights = optax.apply_updates(weights, updates)
    for i, new_w in enumerate(new_weights):
        if new_w is not None:
            elements[i] = _writeback_weights(elements[i], new_w)

    return dataclasses.replace(network, layers=tuple(elements)), new_opt_state


def _learn_sweep(
    network: Network,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    learning_kind: str = "precision_weighted",
) -> tuple[Network, optax.OptState]:
    """Weight-learning phase: PE-driven gradients + an optimiser step.

    Mirrors the weight-update block of :func:`propagation_step`. Must run *after*
    :func:`_update_sweep`, so the per-layer states already carry their prediction errors
    / posteriors. Updates ``weights_in`` on every element that has them.
    """
    grads = _weight_gradients(network, learning_kind)
    return _apply_weight_updates(network, grads, opt_state, optimizer)


@eqx.filter_jit
def learn_sweep(
    network: Network,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    learning_kind: str,
) -> tuple[Network, optax.OptState]:
    """JIT-compiled weight-learning phase.

    See :func:`_learn_sweep`.
    """
    return _learn_sweep(network, opt_state, optimizer, learning_kind)


# ---------------------------------------------------------------------------
# Pure per-sample step + batch-synchronous learning
# ---------------------------------------------------------------------------

# The state fields that carry information from one sample to the next. Every
# other field is rewritten by the sweeps: expected means and precisions come
# from the prediction sweep, posterior means are rebuilt as expected mean +
# correction. What persists is the confidence side — the value-level
# posterior precision (each prediction reads the previous one) and the
# volatility level's belief.
_CARRIED_FIELDS: tuple = ("precision", "mean_vol", "precision_vol")


def _confidence_increments(before: Network, after: Network) -> tuple:
    """Per-element change of the carried confidence fields, ``after - before``.

    Returns one ``dict`` per element, keyed by field name. For a ``Layer``
    each entry has shape ``(n_nodes,)``; for a ``LayerStack``,
    ``(n_slices, n_nodes)``.
    """
    # ``mean_vol``/``precision_vol`` are ``None`` on layers without a volatility
    # parent — there is no volatility confidence to carry, so skip them.
    return tuple(
        {
            field: getattr(elem_after.state, field) - getattr(elem_before.state, field)
            for field in _CARRIED_FIELDS
            if getattr(elem_before.state, field) is not None
        }
        for elem_before, elem_after in zip(before.layers, after.layers)
    )


def apply_confidence_increments(network: Network, increments: tuple) -> Network:
    """Add confidence increments (see :func:`_confidence_increments`) to a network.

    Used by :func:`batch_step` to carry the batch-averaged confidence change into the
    state used by the next batch.
    """
    new_elements = []
    for elem, inc in zip(network.layers, increments):
        new_state = dataclasses.replace(
            elem.state,
            **{
                field: getattr(elem.state, field) + inc[field]
                for field in _CARRIED_FIELDS
                if field in inc
            },
        )
        new_elements.append(dataclasses.replace(elem, state=new_state))
    return dataclasses.replace(network, layers=tuple(new_elements))


def sample_step(
    network: Network,
    x: jnp.ndarray,
    y: jnp.ndarray,
    learning_kind: str = "precision_weighted",
    time_step: float = 1.0,
) -> tuple[jnp.ndarray, tuple, tuple]:
    """One full local learning step for one sample, as a pure function.

    Runs the prediction sweep (clamp ``x`` on top, predict downward) and the
    update sweep (clamp ``y`` at the bottom, compute errors and correct
    beliefs upward), then reads out everything a caller needs without
    mutating anything:

    Parameters
    ----------
    network :
        The state template. Not modified; every call starting from the same
        template sees the same weights and the same confidences, which is
        what makes this function safe to ``jax.vmap`` over a batch of
        samples.
    x :
        Predictors clamped on the top layer, shape ``(n_input_features,)``.
    y :
        Observations clamped on the bottom layer, shape
        ``(n_output_features,)``.
    learning_kind :
        Weight-gradient mode, as in
        :func:`pyhgf.updates.vectorized.learning.vectorized_weight_gradient`.
    time_step :
        Inference time step for the prediction sweep.

    Returns
    -------
    input_error :
        The prediction error at the input (top) layer — see
        :func:`input_prediction_error`.
    grads :
        Per-element weight gradients (descent form, ``None`` for the bottom
        element). Average these across a batch and apply once.
    increments :
        Per-element change of the carried confidence fields (value-level
        posterior precision and the volatility level), relative to the
        template. Average these across a batch and apply once with
        :func:`apply_confidence_increments`.
    """
    updated = _update_sweep(_prediction_sweep(network, x, time_step=time_step), y)
    return (
        _input_prediction_error(updated),
        _weight_gradients(updated, learning_kind),
        _confidence_increments(network, updated),
    )


def _batch_step(
    network: Network,
    opt_state: Optional[optax.OptState],
    x: jnp.ndarray,
    y: jnp.ndarray,
    optimizer: Optional[optax.GradientTransformation] = None,
    learning_kind: str = "precision_weighted",
    update_confidences: bool = True,
    time_step: float = 1.0,
    predicted: Optional[tuple] = None,
) -> tuple[Network, Optional[optax.OptState], jnp.ndarray]:
    """One batch-synchronous learning step over many samples at once.

    Every sample in the batch is processed from the *same* state template —
    same weights, same confidences — through the same sweeps as
    :func:`sample_step`, under ``jax.vmap``, so samples are exchangeable and
    nothing depends on their order. The per-sample results are then averaged
    and applied once, so the batch counts as a single observation:

    * weight gradients → one optimiser step (skipped when ``optimizer`` is
      ``None``);
    * confidence increments → added to the carried fields (skipped when
      ``update_confidences`` is ``False``, e.g. to keep confidences pinned
      during backprop-parity comparisons).

    Averaging (rather than summing) makes the result invariant to repeating
    the batch: the same samples twice produce the same step.

    Parameters
    ----------
    network :
        The state template shared by every sample in the batch.
    opt_state :
        The optimiser state, or ``None`` when ``optimizer`` is ``None``.
    x :
        Predictors, shape ``(batch, n_input_features)``.
    y :
        Observations, shape ``(batch, n_output_features)``.
    optimizer :
        Optax optimiser for the weight step. ``None`` freezes the weights.
    learning_kind :
        Weight-gradient mode.
    update_confidences :
        Whether to carry the batch-averaged confidence increments into the
        returned network.
    time_step :
        Inference time step, applied once per batch.
    predicted :
        Optional per-sample predicted states from
        :func:`batched_prediction_states` (one batched ``LayerState`` per
        element). When given, the internal prediction sweep is skipped and
        the update starts from these states — the forward pass a caller has
        already run is not repeated. ``x`` is ignored in that case.

    Returns
    -------
    network :
        The template advanced by one batch: new weights and, if requested,
        new confidences. Everything else is untouched (it is rewritten by
        the sweeps on the next call anyway).
    opt_state :
        The advanced optimiser state (``None`` if no optimiser was given).
    input_errors :
        Per-sample prediction errors at the input layer, shape
        ``(batch, n_input_features)`` — the messages a caller passes to
        whatever sits behind this network.
    """

    # Each sample contributes only its two gradient factors (small vectors);
    # the batch-mean gradient is then one contraction per weight matrix. This
    # avoids materialising one weight-matrix-sized gradient per sample under
    # vmap — the same arithmetic, a batch factor less memory traffic. Every
    # gradient kind is separable, so this is the only path.
    def finish_sample(swept: Network, yi):
        updated = _update_sweep(swept, yi)
        factors = None
        if optimizer is not None:
            factors = tuple(
                [None]
                + [
                    _weight_factor(
                        updated.layers[i], updated.layers[i - 1], learning_kind
                    )
                    for i in range(1, len(updated.layers))
                ]
            )
        return (
            _input_prediction_error(updated),
            _confidence_increments(network, updated),
            factors,
        )

    if predicted is None:

        def per_sample(xi, yi):
            return finish_sample(
                _prediction_sweep(network, xi, time_step=time_step), yi
            )

        input_errors, increments, factors = jax.vmap(per_sample)(x, y)
    else:
        # Rebuild each sample's network around the shared (unbatched) weights
        # and static fields; only the layer states carry a batch axis.
        def per_sample_predicted(states_i, yi):
            swept = dataclasses.replace(
                network,
                layers=tuple(
                    dataclasses.replace(elem, state=state_i)
                    for elem, state_i in zip(network.layers, states_i)
                ),
            )
            return finish_sample(swept, yi)

        input_errors, increments, factors = jax.vmap(per_sample_predicted)(predicted, y)

    new_network = network
    if optimizer is not None:
        mean_grads = tuple(_contract_factors(f) for f in factors)
        new_network, opt_state = _apply_weight_updates(
            new_network, mean_grads, opt_state, optimizer
        )

    if update_confidences:
        mean_increments = jax.tree_util.tree_map(lambda i: i.mean(axis=0), increments)
        new_network = apply_confidence_increments(new_network, mean_increments)

    return new_network, opt_state, input_errors


# The compiled entry point. The implementation stays callable unjitted so a
# larger compiled program (a fused pipeline step) can inline it.
batch_step = eqx.filter_jit(_batch_step)


def _contract_factors(factors) -> Optional[jnp.ndarray]:
    """Batch-mean gradient from stacked per-sample factors.

    ``factors`` is ``None`` for the bottom element, or a ``(u, v)`` pair with
    a leading batch axis: ``(batch, n_children)`` and ``(batch, n_parents)``
    for a ``Layer``; ``(batch, n_slices, ...)`` for a ``LayerStack``. The
    mean over samples of ``u ⊗ v`` is computed as a single contraction.
    """
    if factors is None:
        return None
    u, v = factors
    if u.ndim == 2:
        return jnp.einsum("bi,bj->ij", u, v) / u.shape[0]
    return jnp.einsum("bni,bnj->nij", u, v) / u.shape[0]


@eqx.filter_jit
def prediction_pass(network: Network, x: jnp.ndarray) -> jnp.ndarray:
    """Forward-only sweep through the network (no PE / posterior / learning).

    Sets predictors on the top element and runs the top-down prediction pass; returns
    the bottom element's ``expected_mean``. Used by
    :meth:`pyhgf.model.DeepNetwork.predict`.

    Parameters
    ----------
    network :
        The current vectorised network state.
    x :
        The predictors set on the top element.

    Returns
    -------
    expected_mean :
        The bottom element's ``expected_mean`` after the forward sweep.
    """
    return _prediction_sweep(network, x).layers[0].state.expected_mean


@eqx.filter_jit
def batched_prediction_pass(network: Network, x: jnp.ndarray) -> jnp.ndarray:
    """Forward-only sweep for a batch of samples, compiled once and reused.

    The batched equivalent of :func:`prediction_pass`: every row of ``x`` is
    an independent sample swept from the same network state. Used by
    :meth:`pyhgf.model.DeepNetwork.predict` so repeated batched calls hit
    the compilation cache instead of rebuilding the batching wrapper.

    Parameters
    ----------
    network :
        The current vectorised network state.
    x :
        Predictors, shape ``(batch, n_input_features)``.

    Returns
    -------
    expected_mean :
        The bottom element's ``expected_mean`` per sample, shape
        ``(batch, n_output_features)``.
    """
    return jax.vmap(
        lambda xi: _prediction_sweep(network, xi).layers[0].state.expected_mean
    )(x)


@eqx.filter_jit
def batched_prediction_states(network: Network, x: jnp.ndarray) -> tuple:
    """Batched forward sweep returning the per-sample swept states.

    Like :func:`batched_prediction_pass`, but keeps what the sweep computed:
    one batched ``LayerState`` per element (each field with a leading batch
    axis). Passing these to :func:`batch_step` as ``predicted`` lets the
    learning step start directly from them instead of repeating the forward
    sweep — the weights and static fields are not duplicated per sample,
    only the layer states are.

    The states are the *only* output: the per-sample predictions are read
    from the bottom element's ``expected_mean`` after the call. Returning
    that array alongside the states from the same compiled function produces
    incorrect values under the vmap-of-jit composition on CPU, so callers
    must read it from the returned states.

    Parameters
    ----------
    network :
        The current vectorised network state.
    x :
        Predictors, shape ``(batch, n_input_features)``.

    Returns
    -------
    states :
        One batched ``LayerState`` per element, ordered as
        ``network.layers``.
    """

    def one(xi):
        return tuple(elem.state for elem in _prediction_sweep(network, xi).layers)

    return jax.vmap(one)(x)
