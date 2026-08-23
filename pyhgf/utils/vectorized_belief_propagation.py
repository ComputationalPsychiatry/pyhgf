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
from pyhgf.updates.vectorized.continuous import (
    ValueChild,
    VolatilityChild,
    vectorized_continuous_posterior_update,
    vectorized_continuous_prediction,
    vectorized_continuous_prediction_error,
)
from pyhgf.updates.vectorized.learning import (
    SynapticUncertaintySettings,
    clamped_layer_evidence,
    evidence_pullback,
    learning_weights_vectorized,
    vectorized_synaptic_uncertainty_update,
)
from pyhgf.updates.vectorized.volatile import (
    vectorized_layer_posterior_update,
    vectorized_layer_prediction,
    vectorized_layer_prediction_error,
    vectorized_posterior_update_precision_value_level,
    vectorized_root_prediction,
)

# ---------------------------------------------------------------------------
# Element-shape helpers
# ---------------------------------------------------------------------------


def _stack_slice(stack: LayerStack, index: int):
    """Return ``(state, params, weights_mean)`` of the stack slice at ``index``.

    Index ``0`` is the bottommost slice, ``-1`` the topmost.
    """
    state = jax.tree_util.tree_map(lambda x: x[index], stack.state)
    params = jax.tree_util.tree_map(lambda x: x[index], stack.params)
    return state, params, stack.weights_mean[index]


def _parent_view(elem):
    """Treat a ``Layer`` or ``LayerStack`` uniformly when acting as a parent.

    Returns ``(state, weights_mean, coupling_fn, add_constant_input)``. The four pieces
    ``propagation_step`` needs to predict a child below.

    For a ``LayerStack``, the parent is the *bottommost* slice (the slice closest to the
    child below the stack).
    """
    if isinstance(elem, LayerStack):
        state, _, weights = _stack_slice(elem, 0)
        return state, weights, elem.coupling_fn, elem.add_constant_input
    return elem.state, elem.weights_mean, elem.coupling_fn, elem.add_constant_input


def _child_view(elem):
    """Treat a ``Layer`` or ``LayerStack`` uniformly when acting as a child.

    Returns ``(state, kind, is_input_layer)``. What's needed when something above is
    doing a posterior update or computing prediction-error-driven weight gradients using
    this element's state as the child.

    For a ``LayerStack``, the child role is filled by the *topmost* slice (the slice
    closest to the parent above the stack).
    """
    if isinstance(elem, LayerStack):
        state, _, _ = _stack_slice(elem, -1)
        return state, elem.kind, False  # interior; never the clamped leaf
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
    predict_precision: bool = True,
    feedforward_uncertainty: bool = False,
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
            predict_precision=predict_precision,
            feedforward_uncertainty=feedforward_uncertainty,
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
    predict_precision: bool = True,
    feedforward_uncertainty: bool = False,
):
    """Top-down sweep over a ``LayerStack``.

    Boundary step: predict the topmost slice from the external parent, using the
    parent's coupling function, weights, and bias — a ``Layer`` parent's own, or,
    for a ``LayerStack`` parent, those of its bottommost slice.

    Scan step: predict slice ``k`` from slice ``k+1`` for ``k = N-2 ... 0`` using
    ``stack.weights_mean[k+1]`` and the stack's own coupling function and bias.
    The scan runs in reverse so the carry threads top-to-bottom through the stack.
    """
    top_slice_state, top_slice_params, _ = _stack_slice(stack, -1)
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
        predict_precision=predict_precision,
        feedforward_uncertainty=feedforward_uncertainty,
    )

    # xs: per-iteration data for predicting slices N-2 ... 0 from the slice above.
    # At step k, body(parent_state, xs[k]) → predict slice k. The "parent's
    # weights" used to predict slice k come from slice k+1 — i.e. stack.weights_mean[k+1].
    # A single-slice stack yields zero-length xs: the scan runs no steps and the
    # concatenation below just wraps the top state.
    n = stack.n_layers
    xs_child_state = jax.tree_util.tree_map(lambda x: x[: n - 1], stack.state)
    xs_child_params = jax.tree_util.tree_map(lambda x: x[: n - 1], stack.params)
    xs_parent_weights = stack.weights_mean[1:]  # shape (n-1, ...)

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
            predict_precision=predict_precision,
            feedforward_uncertainty=feedforward_uncertainty,
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
    parent_elem,
    child_elem,
    *,
    time_step: float,
    precision_clipping_value: float,
    predict_precision: bool = True,
    feedforward_uncertainty: bool = False,
):
    """Predict ``child_elem`` from ``parent_elem``.

    Either element can be a ``Layer`` or a ``LayerStack``.
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
            predict_precision=predict_precision,
            feedforward_uncertainty=feedforward_uncertainty,
        )
    return _predict_layer_from_parent(
        child_elem,
        parent_state,
        parent_weights,
        parent_coupling_fn,
        parent_has_const,
        time_step=time_step,
        precision_clipping_value=precision_clipping_value,
        predict_precision=predict_precision,
        feedforward_uncertainty=feedforward_uncertainty,
    )


# ---------------------------------------------------------------------------
# Leaf prediction error (bottom element of the network)
# ---------------------------------------------------------------------------


def _leaf_pe(
    layer: Layer,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
    time_step: float = 1.0,
):
    """Compute the prediction error of the bottom layer (never a stack)."""
    if layer.kind == "binary":
        new_state = vectorized_binary_prediction_error(layer=layer.state)
    elif layer.kind == "categorical":
        new_state = vectorized_categorical_prediction_error(layer=layer.state)
    else:
        new_state = vectorized_layer_prediction_error(
            layer=layer.state,
            volatility_updates=volatility_updates,
            time_step=time_step,
            has_volatility_parent=layer.has_volatility_parent,
            max_posterior_precision=max_posterior_precision,
        )
    return dataclasses.replace(layer, state=new_state)


# ---------------------------------------------------------------------------
# Bottom-up posterior update + prediction error
# ---------------------------------------------------------------------------


def _posterior_pe_layer(
    parent: Layer,
    child_state,
    child_is_input_layer: bool,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
    time_step: float = 1.0,
):
    """Single-layer posterior update + prediction error."""
    new_state = vectorized_layer_posterior_update(
        layer=parent.state,
        child=child_state,
        weights=parent.weights_mean,
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
            volatility_updates=volatility_updates,
            time_step=time_step,
            has_volatility_parent=parent.has_volatility_parent,
            max_posterior_precision=max_posterior_precision,
        )
    return dataclasses.replace(parent, state=new_state)


def _top_precision_only(
    parent: Layer,
    child_state,
    child_is_input_layer: bool,
    *,
    max_posterior_precision: float,
):
    r"""Update the top layer's precision from the layer below, leaving its mean clamped.

    The top layer holds the predictors, and its mean is read back by the weight update
    (:func:`~pyhgf.updates.vectorized.learning.learning_weights_vectorized`
    forms the parent-side factor from ``coupling_fn(parent.mean)``). Those weights must
    be learned against the predictors that were actually supplied, so the mean stays
    pinned to ``x`` and only its precision moves.

    Two things follow from the clamp, and both are deliberate:

    * The value prediction error is identically zero. A layer whose mean never leaves
      its prediction has no residual, so the field is written as zero rather than left
      holding a stale value.
    * The volatility level is *not* updated. Its prediction error would reduce to
      :math:`\hat{\pi} / \pi - 1`, which the clamp makes non-positive at every step,
      so the layer would conclude "no volatility" and keep concluding it. That drives
      :math:`\Omega \to 0`, and without diffusion :math:`\hat{\pi} \to \pi`, so
      each step's evidence would add to the last and the precision would grow without
      bound. Leaving the volatility level at its tonic value instead keeps
      :math:`\Omega` constant, and the precision settles at
      :math:`1/\Omega + \text{evidence}` — still tracking how well the layer below
      accounts for the predictors, but bounded.
    """
    # Only called for a top element that has a layer below it, so it carries an
    # incoming matrix.
    assert parent.weights_mean is not None
    weights = parent.weights_mean
    if parent.add_constant_input:
        weights = weights[:, :-1]

    precision = jnp.clip(
        vectorized_posterior_update_precision_value_level(
            layer=parent.state,
            child=child_state,
            weights=weights,
            coupling_fn=parent.coupling_fn,
            child_is_input_layer=child_is_input_layer,
        ),
        parent.state.expected_precision,
        max_posterior_precision,
    )
    new_state = dataclasses.replace(
        parent.state,
        precision=precision,
        value_prediction_error=jnp.zeros_like(precision),
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
    child_is_input_layer: bool,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
    time_step: float = 1.0,
):
    r"""Bottom-up sweep over a ``LayerStack``.

    Posterior update and prediction error for every slice, from slice 0 (bottommost)
    to slice N-1 (topmost). The carry is the child state below the current slice,
    already carrying its prediction error.

    Slice 0 is the boundary and runs outside the scan: its child is the external
    element below the stack, which may be the clamped observation leaf (volatile,
    binary, or categorical), so it receives the real *child_is_input_layer* flag.
    A leaf never moves its posterior precision, so its evidence
    :math:`\pi_y = \pi_a - \tilde{\pi}_a` is identically zero and the interior
    (harmonic) form of the smoothing correction would silently zero out the whole
    message; the flag switches to the canonical factor instead. This mirrors
    :func:`_predict_stack_from_parent`, which peels the *topmost* slice to meet
    the external parent. Every scanned slice has a stack slice as its child,
    interior by construction, so the scan runs with ``child_is_input_layer=False``.
    """
    # The scan carry becomes a stack slice each step, so seed it with the
    # child's state coerced to the stack's volatility structure.
    child_state_init = _match_child_vol_structure(
        child_state_init, stack.has_volatility_parent
    )

    def slice_posterior_pe(slice_state, slice_weights, child_state, is_leaf_child):
        new_state = vectorized_layer_posterior_update(
            layer=slice_state,
            child=child_state,
            weights=slice_weights,
            coupling_fn=stack.coupling_fn,
            parent_has_constant=stack.add_constant_input,
            max_posterior_precision=max_posterior_precision,
            child_is_input_layer=is_leaf_child,
        )
        return vectorized_layer_prediction_error(
            layer=new_state,
            volatility_updates=volatility_updates,
            time_step=time_step,
            has_volatility_parent=stack.has_volatility_parent,
            max_posterior_precision=max_posterior_precision,
        )

    # Boundary: slice 0 from the external child.
    slice0_state, _, slice0_weights = _stack_slice(stack, 0)
    new_slice0 = slice_posterior_pe(
        slice0_state, slice0_weights, child_state_init, child_is_input_layer
    )

    def body(child_carry_state, slice_data):
        slice_state, slice_weights = slice_data
        new_state = slice_posterior_pe(
            slice_state, slice_weights, child_carry_state, False
        )
        return new_state, new_state

    # Slices 1 .. N-1, each from the freshly updated slice below. A single-slice
    # stack yields zero-length xs: the scan runs no steps and the concatenation
    # below just wraps slice 0.
    _, new_states_above = jax.lax.scan(
        body,
        init=new_slice0,
        xs=(
            jax.tree_util.tree_map(lambda x: x[1:], stack.state),
            stack.weights_mean[1:],
        ),
    )
    new_full_state = jax.tree_util.tree_map(
        lambda first, above: jnp.concatenate([first[None, ...], above], axis=0),
        new_slice0,
        new_states_above,
    )
    return dataclasses.replace(stack, state=new_full_state)


def _bottomup_posterior_pe(
    parent_elem,
    child_elem,
    *,
    volatility_updates: str,
    max_posterior_precision: float,
    time_step: float = 1.0,
):
    """Posterior update + prediction error for ``parent_elem`` from ``child_elem``."""
    child_state, _, child_is_input_layer = _child_view(child_elem)
    if isinstance(parent_elem, LayerStack):
        return _posterior_pe_stack(
            parent_elem,
            child_state,
            child_is_input_layer,
            volatility_updates=volatility_updates,
            max_posterior_precision=max_posterior_precision,
            time_step=time_step,
        )
    return _posterior_pe_layer(
        parent_elem,
        child_state,
        child_is_input_layer,
        volatility_updates=volatility_updates,
        max_posterior_precision=max_posterior_precision,
        time_step=time_step,
    )


# ---------------------------------------------------------------------------
# Weight gradients
# ---------------------------------------------------------------------------


def _layer_weight_op(
    parent: Layer, child_elem, learning_kind: str, child_evidence=None
):
    """Learning factors for a ``Layer`` parent and its child.

    *child_evidence* is passed only by the evidence walk; ``None`` leaves
    ``learning_weights_vectorized`` to recover the evidence from the cache.
    """
    child_state, child_kind, _ = _child_view(child_elem)
    return learning_weights_vectorized(
        parent_state=parent.state,
        child_state=child_state,
        coupling_fn=parent.coupling_fn,
        kind=learning_kind,
        parent_has_constant=parent.add_constant_input,
        child_kind=child_kind,
        child_evidence=child_evidence,
    )


def _stack_weight_op(stack: LayerStack, child_elem, learning_kind: str, evidence=None):
    """Learning factors for every slice of a ``LayerStack``, and the evidence above it.

    The child of slice 0 is the layer below the stack (``child_elem``); the child of
    slice k>0 is slice k-1 within the stack. Slice 0 is the boundary and is computed
    on its own: its child keeps its actual kind, since it may be the clamped binary
    or categorical observation layer, whose gradient and importance factors differ
    from a continuous child's. The interior slices ``1 .. N-1`` all have a stack
    slice as their child and are ``vmap``-ed together.

    The evidence walk cannot be vectorised the same way, because slice k's evidence is
    a function of slice k-1's: it is a recursion, not a map. It is carried by a
    ``scan`` that emits, per slice, the evidence *arriving* at that slice from below,
    which is exactly what the slice's own increment needs; the scan's final carry is
    the evidence leaving the top of the stack, for the element above it.

    *evidence* is ``None`` for the learning kinds that never read the importance
    factor; the walk is then skipped and the evidence above the stack is ``None`` too.

    Returns
    -------
    tuple
        The stacked factor triple, and the evidence at the top of the stack.
    """
    child_state, child_kind, _ = _child_view(child_elem)

    def carry(evidence_below, slice_data):
        slice_state, slice_weights = slice_data
        above = evidence_pullback(
            parent_state=slice_state,
            child_evidence=evidence_below,
            weights=slice_weights,
            coupling_fn=stack.coupling_fn,
            parent_has_constant=stack.add_constant_input,
        )
        return above, evidence_below

    if evidence is None:
        evidence_out = None
        per_slice_evidence = jnp.zeros(stack.state.mean.shape)
    else:
        evidence_out, per_slice_evidence = jax.lax.scan(
            carry, evidence, (stack.state, stack.weights_mean)
        )

    walking = evidence is not None

    def slice_factors(parent_state, child_state_for_slice, child_evidence, kind):
        return learning_weights_vectorized(
            parent_state=parent_state,
            child_state=child_state_for_slice,
            coupling_fn=stack.coupling_fn,
            kind=learning_kind,
            parent_has_constant=stack.add_constant_input,
            child_kind=kind,
            child_evidence=child_evidence if walking else None,
        )

    # Boundary: slice 0 from the external child, with the child's own kind.
    first = slice_factors(
        jax.tree_util.tree_map(lambda x: x[0], stack.state),
        child_state,
        per_slice_evidence[0],
        child_kind,
    )
    # Interior slices 1 .. N-1, whose children are the stack's own slices. A
    # single-slice stack maps over a zero-length axis and the concatenation
    # below just wraps the boundary factors.
    rest = jax.vmap(lambda p, c, e: slice_factors(p, c, e, "continuous"))(
        jax.tree_util.tree_map(lambda x: x[1:], stack.state),
        jax.tree_util.tree_map(lambda x: x[:-1], stack.state),
        per_slice_evidence[1:],
    )
    factors = jax.tree_util.tree_map(
        lambda f, r: jnp.concatenate([f[None, ...], r], axis=0), first, rest
    )
    return (factors, evidence_out) if walking else factors


def _weight_op(parent_elem, child_elem, learning_kind: str):
    """Dispatch the learning factors on ``Layer`` vs ``LayerStack``."""
    if isinstance(parent_elem, LayerStack):
        return _stack_weight_op(parent_elem, child_elem, learning_kind)
    return _layer_weight_op(parent_elem, child_elem, learning_kind)


# ---------------------------------------------------------------------------
# Element-level state writeback (for clamping x/y at the boundaries)
# ---------------------------------------------------------------------------


def _set_top_predictors(elem, x):
    """Clamp ``expected_mean`` and ``mean`` of the top element to the predictors ``x``.

    The top element must be a ``Layer``.
    """
    if isinstance(elem, LayerStack):
        raise NotImplementedError("Top of network must be a Layer, not a LayerStack.")
    new_state = dataclasses.replace(elem.state, expected_mean=x, mean=x)
    return dataclasses.replace(elem, state=new_state)


def _sweeps_reach_top(network: Network) -> bool:
    """Whether both sweeps should treat the top element as a member of the hierarchy.

    Gating the two sweeps on one predicate keeps them in step: a top element that
    receives a posterior update must also receive the precision prediction that
    carries it into the next step, or its precision would accumulate with nothing
    advancing it.

    Requires ``update_input_layer``, something below the top to send it a message,
    and a volatile top element — a binary layer's predicted precision is a function
    of its predicted mean, which is clamped to the predictors here, so there is no
    precision of its own for the sweeps to move.
    """
    return (
        network.update_input_layer
        and len(network.layers) > 1
        and network.layers[-1].kind == "volatile"
    )


def _predict_top_precisions(elem, *, time_step: float, predict_precision: bool = True):
    """Predict the top element's precisions, which no parent above it can supply.

    The element's ``expected_mean`` stays clamped to the predictors; what this adds is
    the predicted precision of both levels, so the precision the bottom-up sweep wrote
    into ``precision`` on the previous step is carried forward (damped by the volatility
    level) instead of being ignored.

    ``predict_precision`` is threaded through: the network-level switch has to reach the
    top element as well, or it would keep diffusing while every layer below it froze.
    """
    new_state = vectorized_root_prediction(
        layer_state=elem.state,
        params=elem.params,
        time_step=time_step,
        has_volatility_parent=elem.has_volatility_parent,
        predict_precision=predict_precision,
    )
    return dataclasses.replace(elem, state=new_state)


def _set_bottom_observations(elem, y):
    """Clamp ``mean`` of the bottom element to the observations ``y``.

    The bottom element must be a ``Layer``.
    """
    if isinstance(elem, LayerStack):
        raise NotImplementedError(
            "Bottom of network must be a Layer, not a LayerStack."
        )
    new_state = dataclasses.replace(elem.state, mean=y)
    return dataclasses.replace(elem, state=new_state)


# ---------------------------------------------------------------------------
# Top-level propagation step
# ---------------------------------------------------------------------------


def propagation_step(
    network: Network,
    opt_state: optax.OptState,
    inputs: tuple,
    *,
    optimizer: Optional[optax.GradientTransformation],
    time_step: float = 1.0,
    learning_kind: str = "precision_weighted",
    weight_update: bool = True,
    synaptic_uncertainty_settings: Optional[SynapticUncertaintySettings] = None,
) -> tuple[tuple[Network, optax.OptState], jnp.ndarray]:
    """Single propagation step through the network.

    Belief-propagation sweep — top-down prediction, leaf prediction error, then the
    interleaved posterior update + prediction error bottom-up — followed by an
    optional weight-learning phase. Each step dispatches per element:

    * ``Layer``: standard per-layer kernel call (unrolled).
    * ``LayerStack``: ``jax.lax.scan`` over the stack's slices.

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
        :py:func:`pyhgf.updates.vectorized.learning.learning_weights_vectorized`.
    weight_update :
        Whether to apply the weight-learning phase after belief propagation.

    Returns
    -------
    carry :
        A tuple ``((network, opt_state), output_pred)`` where ``network`` and
        ``opt_state`` are updated and ``output_pred`` is the bottom element's
        ``expected_mean`` — the prediction of the observations for this step.
    """
    x, y = inputs

    # Belief propagation: top-down prediction (clamping x on top) then the
    # bottom-up prediction-error + posterior sweep (clamping y at the bottom).
    swept = _update_sweep(
        _prediction_sweep(network, x, time_step=time_step), y, time_step=time_step
    )

    # Optional weight-learning phase.
    if weight_update:
        new_network, new_opt_state = _learn_sweep(
            swept, opt_state, optimizer, learning_kind, synaptic_uncertainty_settings
        )
    else:
        new_network, new_opt_state = swept, opt_state

    output_pred = new_network.layers[0].state.expected_mean
    return (new_network, new_opt_state), output_pred


# ---------------------------------------------------------------------------
# Scan driver + prediction-only sweep
# ---------------------------------------------------------------------------


@eqx.filter_jit
def run_scan(
    init_carry: tuple,
    inputs: tuple,
    optimizer: Optional[optax.GradientTransformation],
    learning_kind: str,
    weight_update: bool,
    record: tuple,
    time_step: float = 1.0,
    update_precisions: bool = True,
    synaptic_uncertainty_settings: Optional[SynapticUncertaintySettings] = None,
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
        :py:func:`pyhgf.updates.vectorized.learning.learning_weights_vectorized`.
    weight_update :
        Whether to apply the weight-learning phase at every step.
    record :
        Tuple of ``LayerState`` field names to record at every time step (e.g.
        ``("expected_mean", "precision")``). An empty tuple disables recording and the
        scan output is the per-step ``output_pred`` alone. With a non-empty tuple, the
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
    template = init_carry[0]

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
            synaptic_uncertainty_settings=synaptic_uncertainty_settings,
        )
        if not update_precisions:
            # Static-cascade mode: precisions are parameters, not filter state. Note
            # that recorded carried fields then show the template values, since the
            # restore runs before recording.
            new_network = _restore_precisions(new_network, template)
        if record:
            traj_step = {
                field: tuple(getattr(elem.state, field) for elem in new_network.layers)
                for field in record
            }
            return (new_network, new_opt_state), (traj_step, pred)
        return (new_network, new_opt_state), pred

    return jax.lax.scan(_scan_body, init_carry, inputs)


def _prediction_sweep(
    network: Network, x: jnp.ndarray, *, time_step: float = 1.0
) -> Network:
    """Top-down prediction sweep, returning the updated network.

    Clamps the predictors on the top element and predicts every element from the one
    above. No prediction errors, posterior updates, or weight learning are performed.

    With ``network.update_input_layer``, the top element also gets its own precision
    prediction (:func:`_predict_top_precisions`) — the only part of it a parent could
    have supplied, had there been one.
    """
    elements = list(network.layers)
    n_elements = len(elements)

    elements[-1] = _set_top_predictors(elements[-1], x)
    if _sweeps_reach_top(network):
        elements[-1] = _predict_top_precisions(
            elements[-1],
            time_step=time_step,
            predict_precision=network.predict_precision,
        )

    for i in range(n_elements - 1, 0, -1):
        elements[i - 1] = _topdown_predict(
            elements[i],
            elements[i - 1],
            time_step=time_step,
            precision_clipping_value=network.precision_clipping_value,
            predict_precision=network.predict_precision,
            feedforward_uncertainty=network.feedforward_uncertainty,
        )

    return dataclasses.replace(network, layers=tuple(elements))


def _update_sweep(
    network: Network, y: jnp.ndarray, *, time_step: float = 1.0
) -> Network:
    """Bottom-up prediction-error + posterior-update sweep, returning the network.

    Clamps the observations on the bottom element, computes the leaf prediction error,
    then performs the interleaved posterior update + prediction error for every interior
    element, in bottom-up order. Belief states are updated; inter-layer weights are not.
    The inference time step scales the volatility-level posterior updates with the same
    time step the prediction sweep uses.

    With ``network.update_input_layer``, the sweep also reaches the top element, but
    on different terms from the interior: its mean stays clamped to the predictors and
    only its precision moves (see :func:`_top_precision_only`). The top element holds
    observed inputs that the weight update reads back, so its value is not the
    network's to revise, only its precision is.
    """
    elements = list(network.layers)
    n_elements = len(elements)

    # Clamp observations and compute the leaf prediction error.
    elements[0] = _set_bottom_observations(elements[0], y)
    elements[0] = _leaf_pe(
        elements[0],
        volatility_updates=network.volatility_updates,
        max_posterior_precision=network.max_posterior_precision,
        time_step=time_step,
    )

    # Interleaved bottom-up posterior update + prediction error on every
    # interior element.
    for i in range(1, n_elements - 1):
        elements[i] = _bottomup_posterior_pe(
            elements[i],
            elements[i - 1],
            volatility_updates=network.volatility_updates,
            max_posterior_precision=network.max_posterior_precision,
            time_step=time_step,
        )

    # The top element, when asked for: precision only, mean left on the predictors.
    if _sweeps_reach_top(network):
        child_state, _, child_is_input_layer = _child_view(elements[-2])
        elements[-1] = _top_precision_only(
            elements[-1],
            child_state,
            child_is_input_layer,
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
def update_sweep(network: Network, y: jnp.ndarray, time_step: float = 1.0) -> Network:
    """JIT-compiled bottom-up prediction-error + posterior sweep.

    See :func:`_update_sweep`.
    """
    return _update_sweep(network, y, time_step=time_step)


def _input_prediction_error(network: Network) -> jnp.ndarray:
    r"""Prediction error routed to the network's input (top) layer.

    This is the error message the top layer receives from the layer below it —
    the same gain-weighted prediction error that drives the posterior mean
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

    The quantity is the same whether or not the network updates its top layer:
    it is read off the child, not the top. With ``update_input_layer=True`` the
    top layer's own ``value_prediction_error`` is this message divided by the
    top layer's posterior precision — the shift the belief actually made, rather
    than the raw message that drove it.

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
    if top.weights_mean is None:
        raise ValueError(
            "The network has a single layer: there is no layer below the "
            "input layer to route an error from."
        )
    child_state, _, _ = _child_view(network.layers[-2])

    weights = top.weights_mean
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


def _weight_quantities(network: Network, learning_kind: str) -> tuple:
    r"""Per-element weight-learning factors, without applying them.

    Must run *after* :func:`_update_sweep`, so the per-layer states already carry their
    prediction errors / posteriors. Returns one entry per element, matched 1:1 to
    ``network.layers`` (``None`` for the bottom element, which has no incoming
    weights); each entry is the factor tuple of
    :func:`pyhgf.updates.vectorized.learning.learning_weights_vectorized`.
    Assemble them with :func:`_gradient_matrix` and :func:`_importance_pair`.

    Under ``learning_kind="synaptic_uncertainty"`` the importance increment's
    child-side factor is the evidence precision, and this walks it up the stack in its
    own quantity: seeded at the clamped layer by
    :func:`~pyhgf.updates.vectorized.learning.clamped_layer_evidence` and raised one
    element at a time by
    :func:`~pyhgf.updates.vectorized.learning.evidence_pullback`. This loop runs bottom
    to top already, which is the order the recursion needs, so the walk costs one
    matrix product per element and no extra sweep.

    The evidence is *carried* rather than recovered from the filter's cache as
    :math:`\pi_a - \tilde\pi_a`. The two agree in exact arithmetic only where the
    filter's own chain is seeded with the clamped layer's likelihood curvature, which
    it is not: a clamped categorical layer enters that chain at unit precision, because
    that convention is what makes the message it routes exact cross-entropy
    backpropagation. One variable cannot serve both roles, so the curvature gets its
    own. Carrying it also removes a subtraction of two large nearly-equal precisions,
    which loses every significant digit once a layer's precision has accumulated.

    The other learning kinds never read the importance factor, so they skip the walk.
    """
    elements = network.layers
    if learning_kind != "synaptic_uncertainty":
        return (None,) + tuple(
            _weight_op(elements[i], elements[i - 1], learning_kind)
            for i in range(1, len(elements))
        )

    child_state, child_kind, _ = _child_view(elements[0])
    evidence = clamped_layer_evidence(child_state, child_kind)

    factors: list = [None]
    for i in range(1, len(elements)):
        parent = elements[i]
        if isinstance(parent, LayerStack):
            stack_factors, evidence = _stack_weight_op(
                parent, elements[i - 1], learning_kind, evidence
            )
            factors.append(stack_factors)
            continue
        factors.append(
            _layer_weight_op(parent, elements[i - 1], learning_kind, evidence)
        )
        if i + 1 < len(elements):
            evidence = evidence_pullback(
                parent_state=parent.state,
                child_evidence=evidence,
                weights=parent.weights_mean,
                coupling_fn=parent.coupling_fn,
                parent_has_constant=parent.add_constant_input,
            )
    return tuple(factors)


def _gradient_matrix(factors) -> Optional[jnp.ndarray]:
    """Assemble the descent gradient from one element's factors."""
    if factors is None:
        return None
    u, h = factors[0], factors[1]
    return u[..., :, None] * h[..., None, :]


def _importance_pair(factors) -> Optional[tuple]:
    """Build the importance factor pair ``(p, h**2)`` from one element's factors.

    The parent side is squared here rather than at the source, since both quantities
    read the same activation. Squaring can overflow a finite activation, so the non-
    finite guard is applied after it, matching the child side.
    """
    if factors is None:
        return None
    squared = factors[1] ** 2
    return factors[2], jnp.where(jnp.isfinite(squared), squared, 0.0)


def _apply_weight_updates(
    network: Network,
    grads: tuple,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> tuple[Network, optax.OptState]:
    """One optimiser step on every ``weights_mean``, from precomputed gradients."""
    elements = list(network.layers)
    weights = tuple(elem.weights_mean for elem in elements)

    updates, new_opt_state = optimizer.update(grads, opt_state, weights)
    new_weights = optax.apply_updates(weights, updates)
    for i, new_w in enumerate(new_weights):
        if new_w is not None:
            elements[i] = dataclasses.replace(elements[i], weights_mean=new_w)

    return dataclasses.replace(network, layers=tuple(elements)), new_opt_state


def _apply_synaptic_uncertainty_updates(
    network: Network,
    grads: tuple,
    importance: tuple,
    settings: SynapticUncertaintySettings,
) -> Network:
    """Advance every weight belief one step, mean and precision together.

    The rule needs no optimiser: the step size is the belief's own variance
    (see
    :func:`pyhgf.updates.vectorized.learning.resolve_synaptic_uncertainty_settings`),
    so the update is applied here and both the mean (``weights_mean``) and the
    accumulated precision (``weights_precision_delta``) are written back to
    the element that carries them.

    Parameters
    ----------
    network :
        The network whose beliefs are advanced.
    grads :
        One descent gradient per element, ``None`` for the bottom element.
    importance :
        One importance entry per element, aligned with ``grads``.
    settings :
        The resolved settings of the rule.

    Returns
    -------
    Network
        The network with updated weights and precisions.

    Raises
    ------
    ValueError
        If an element holding weights carries no belief, which means the
        install step was skipped.
    """
    elements = list(network.layers)
    for i, elem in enumerate(elements):
        if elem.weights_mean is None or grads[i] is None:
            continue
        if elem.weights_precision_delta is None:
            raise ValueError(
                f"layers[{i}] holds weights but no weight belief. Install one "
                "with add_layer(weight_belief=True), or let "
                "learning_kind='synaptic_uncertainty' install it."
            )
        new_weights, new_delta = vectorized_synaptic_uncertainty_update(
            elem.weights_mean,
            elem.weights_precision_delta,
            grads[i],
            importance[i],
            settings,
        )
        elements[i] = dataclasses.replace(
            elem, weights_mean=new_weights, weights_precision_delta=new_delta
        )
    return dataclasses.replace(network, layers=tuple(elements))


def _learn_sweep(
    network: Network,
    opt_state: optax.OptState,
    optimizer: Optional[optax.GradientTransformation],
    learning_kind: str = "precision_weighted",
    synaptic_uncertainty_settings: Optional[SynapticUncertaintySettings] = None,
) -> tuple[Network, optax.OptState]:
    """Weight-learning phase: prediction-error-driven gradients, then one step.

    Mirrors the weight-update block of :func:`propagation_step`. Must run *after*
    :func:`_update_sweep`, so the per-layer states already carry their prediction errors
    / posteriors. Updates ``weights_mean`` on every element that has them.

    With ``synaptic_uncertainty_settings`` the weight-belief rule runs instead
    of the optimiser: the gradients are formed from ``learning_kind`` as usual,
    the importance increments alongside them, and both the means and the
    precisions advance. The optimiser state is returned unchanged, since the
    rule carries none.
    """
    factors = _weight_quantities(network, learning_kind)
    grads = tuple(_gradient_matrix(f) for f in factors)
    if synaptic_uncertainty_settings is not None:
        importance = tuple(_importance_pair(f) for f in factors)
        return _apply_synaptic_uncertainty_updates(
            network, grads, importance, synaptic_uncertainty_settings
        ), opt_state
    return _apply_weight_updates(network, grads, opt_state, optimizer)


@eqx.filter_jit
def learn_sweep(
    network: Network,
    opt_state: optax.OptState,
    optimizer: Optional[optax.GradientTransformation],
    learning_kind: str,
    synaptic_uncertainty_settings: Optional[SynapticUncertaintySettings] = None,
) -> tuple[Network, optax.OptState]:
    """JIT-compiled weight-learning phase.

    See :func:`_learn_sweep`.
    """
    return _learn_sweep(
        network, opt_state, optimizer, learning_kind, synaptic_uncertainty_settings
    )


# ---------------------------------------------------------------------------
# Pure per-sample step + batch-synchronous learning
# ---------------------------------------------------------------------------

# The state fields that carry information from one sample to the next. Every
# other field is rewritten by the sweeps: expected means and precisions come
# from the prediction sweep, posterior means are rebuilt as expected mean +
# correction. What persists is the value-level posterior precision (each
# prediction reads the previous one) and the volatility level's belief.
_CARRIED_FIELDS: tuple = ("precision", "mean_vol", "precision_vol")


def _precision_increments(before: Network, after: Network) -> tuple:
    """Per-element change of the carried precision fields, ``after - before``.

    Returns one ``dict`` per element, keyed by field name. For a ``Layer``
    each entry has shape ``(n_nodes,)``; for a ``LayerStack``,
    ``(n_slices, n_nodes)``.
    """
    # ``mean_vol``/``precision_vol`` are ``None`` on layers without a volatility
    # parent — there is no volatility-level belief to carry, so skip them.
    return tuple(
        {
            field: getattr(elem_after.state, field) - getattr(elem_before.state, field)
            for field in _CARRIED_FIELDS
            if getattr(elem_before.state, field) is not None
        }
        for elem_before, elem_after in zip(before.layers, after.layers)
    )


def apply_precision_increments(network: Network, increments: tuple) -> Network:
    """Add precision increments (see :func:`_precision_increments`) to a network.

    Used by :func:`batch_step` to carry the batch-averaged precision change into the
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


def _restore_precisions(network: Network, template: Network) -> Network:
    """Reset the carried precision fields to a template's values.

    The inverse of carrying: with this applied after every propagation step, each
    sample's sweeps still compute full per-sample posteriors but the filtered
    precisions never become the next sample's starting point.
    """
    new_elements = []
    for elem, ref in zip(network.layers, template.layers):
        repl = {
            field: getattr(ref.state, field)
            for field in _CARRIED_FIELDS
            if getattr(ref.state, field) is not None
        }
        new_elements.append(
            dataclasses.replace(elem, state=dataclasses.replace(elem.state, **repl))
        )
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
    mutating anything.

    Parameters
    ----------
    network :
        The state template. Not modified; every call starting from the same
        template sees the same weights and the same precisions, which is
        what makes this function safe to ``jax.vmap`` over a batch of
        samples.
    x :
        Predictors clamped on the top layer, shape ``(n_input_features,)``.
    y :
        Observations clamped on the bottom layer, shape
        ``(n_output_features,)``.
    learning_kind :
        Weight-gradient mode, as in
        :func:`pyhgf.updates.vectorized.learning.learning_weights_vectorized`.
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
        Per-element change of the carried precision fields (value-level
        posterior precision and the volatility level), relative to the
        template. Average these across a batch and apply once with
        :func:`apply_precision_increments`.
    """
    updated = _update_sweep(
        _prediction_sweep(network, x, time_step=time_step), y, time_step=time_step
    )
    return (
        _input_prediction_error(updated),
        tuple(_gradient_matrix(f) for f in _weight_quantities(updated, learning_kind)),
        _precision_increments(network, updated),
    )


def _batch_step(
    network: Network,
    opt_state: Optional[optax.OptState],
    x: jnp.ndarray,
    y: jnp.ndarray,
    optimizer: Optional[optax.GradientTransformation] = None,
    learning_kind: str = "precision_weighted",
    update_precisions: bool = True,
    time_step: float = 1.0,
    predicted: Optional[tuple] = None,
    sample_weight: Optional[jnp.ndarray] = None,
    synaptic_uncertainty_settings: Optional[SynapticUncertaintySettings] = None,
    weight_reuse: float = 1.0,
) -> tuple[Network, Optional[optax.OptState], jnp.ndarray]:
    """One batch-synchronous learning step over many samples at once.

    Every sample in the batch is processed from the *same* state template —
    same weights, same precisions — through the same sweeps as
    :func:`sample_step`, under ``jax.vmap``, so samples are exchangeable and
    nothing depends on their order. The per-sample results are then averaged
    and applied once, so the batch counts as a single observation:

    * the mean weight gradient drives one optimiser step (skipped when
      ``optimizer`` is ``None``);
    * the mean precision increments are added to the carried fields (skipped
      when ``update_precisions`` is ``False``, e.g. to keep the carried
      precisions pinned when comparing against backpropagation).

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
    update_precisions :
        Whether to carry the batch-averaged precision increments into the
        returned network.
    time_step :
        Inference time step, applied once per batch.
    sample_weight :
        Optional per-sample weights, shape ``(batch,)``. The batch mean becomes a
        weighted mean whose denominator is ``sample_weight.sum()`` rather than the
        row count, so rows that carry no information do not dilute the update.

        This exists because "average over the batch" is ambiguous once a caller pads.
        A padded row contributes a zero gradient either way, but with a plain mean it
        still counts in the denominator, so the effective step shrinks by the padding
        fraction. Anything that hands this function a variable-length batch — a token
        sequence, a masked objective, a ragged observation — is affected, and the
        symptom is a *systematic* gradient scale error rather than noise. Pass the mask
        as weights to make the reduction mean-over-contributing-rows instead.

        ``None`` (default) keeps the plain mean, so existing behaviour is unchanged.
    predicted :
        Optional per-sample predicted states from
        :func:`batched_prediction_states` (one batched ``LayerState`` per
        element). When given, the internal prediction sweep is skipped and
        the update starts from these states — the forward pass a caller has
        already run is not repeated. ``x`` is ignored in that case.
    synaptic_uncertainty_settings :
        When given, the weight-belief rule runs in place of ``optimizer``:
        each element's mean and accumulated precision advance together and
        the optimiser state is left untouched (see
        :func:`pyhgf.updates.vectorized.learning.resolve_synaptic_uncertainty_settings`
        ).
        ``learning_kind`` still selects the gradient the rule descends.
    weight_reuse :
        How many times each weight matrix is applied per sample, default ``1.0``
        (once, the ordinary case).

        This exists because "average over the batch" is also ambiguous when one
        weight matrix is *reused* several times per sample. A weight shared across
        ``k`` positions of a sample sees ``k`` rows per sample, so the plain mean
        divides by ``k`` more than that weight's true per-sample quantities, which
        sum over its ``k`` uses and average only over samples. Pass ``k`` to recover
        those sums. The caller owns the count, since only it knows how the rows were
        built (see :func:`pyhgf.model.conv.conv_block`, which passes the patch
        count).

        Both halves of the step are rescaled, so the weight-belief rule of
        ``synaptic_uncertainty`` stays internally consistent: the gradient because
        the chain rule sums a shared weight's uses, and the importance because the
        curvature those uses impose accumulates the same way. Rescaling only the
        gradient would move the mean ``k`` times faster while the belief tightened
        at the one-use rate, leaving a step ``k`` times too large once accumulated
        curvature dominates the prior.

        The importance half carries a modelling assumption the gradient half does
        not. Summing the gradient over uses is the chain rule; summing curvature
        over them treats the ``k`` uses as independent observations, which
        overlapping convolution patches are not. Where that matters, the same
        correction can be had with a smaller ``k``.

    Returns
    -------
    network :
        The template advanced by one batch: new weights and, if requested,
        new precisions. Everything else is untouched (it is rewritten by
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
        updated = _update_sweep(swept, yi, time_step=time_step)
        factors = None
        if optimizer is not None or synaptic_uncertainty_settings is not None:
            factors = _weight_quantities(updated, learning_kind)
        return (
            _input_prediction_error(updated),
            _precision_increments(network, updated),
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
    if optimizer is not None or synaptic_uncertainty_settings is not None:
        mean_grads = tuple(
            None if f is None else _contract_factors((f[0], f[1]), sample_weight)
            for f in factors
        )
        if weight_reuse != 1.0:
            mean_grads = tuple(
                None if g is None else g * weight_reuse for g in mean_grads
            )
        if synaptic_uncertainty_settings is not None:
            importance = tuple(
                _reduce_importance(_importance_pair(f), sample_weight) for f in factors
            )
            if weight_reuse != 1.0:
                # The increment is the outer product H[a, i] = p[a] * q[i], so
                # scaling one factor scales it.
                importance = tuple(
                    None if imp is None else (imp[0] * weight_reuse, imp[1])
                    for imp in importance
                )
            new_network = _apply_synaptic_uncertainty_updates(
                new_network, mean_grads, importance, synaptic_uncertainty_settings
            )
        else:
            new_network, opt_state = _apply_weight_updates(
                new_network, mean_grads, opt_state, optimizer
            )

    if update_precisions:
        if sample_weight is None:
            reduce = lambda i: i.mean(axis=0)  # noqa: E731
        else:
            denominator = jnp.maximum(sample_weight.sum(), 1.0)

            def reduce(i):
                return jnp.tensordot(sample_weight, i, axes=(0, 0)) / denominator

        mean_increments = jax.tree_util.tree_map(reduce, increments)
        new_network = apply_precision_increments(new_network, mean_increments)

    return new_network, opt_state, input_errors


# Compiled entry point. The unjitted ``_batch_step`` is importable so a larger
# compiled program (e.g. a fused pipeline step) can inline it.
batch_step = eqx.filter_jit(_batch_step)


def _reduce_importance(imp_factors, sample_weight=None) -> Optional[tuple]:
    """Batch-mean importance factors from stacked per-sample factors.

    ``imp_factors`` is ``None`` for the bottom element, or a ``(p, q)`` pair with a
    leading batch axis (see :func:`_importance_pair`). Each side is averaged over the
    batch separately. For a continuous child the child-side factor is identical across
    the batch (every sample sweeps from the same state template, and the conditional
    predicted precision is built from carried precisions and volatility states, not from
    the sample's values), and a binary child contributes ones, so the outer product of
    the two means equals the mean of the per-sample outer products.

    ``sample_weight`` follows the semantics of :func:`_contract_factors`: a weighted
    mean whose denominator is the weight the batch actually carries, so padded rows do
    not dilute the increment.
    """
    if imp_factors is None:
        return None
    p, q = imp_factors
    if sample_weight is None:
        return p.mean(axis=0), q.mean(axis=0)
    denominator = jnp.maximum(sample_weight.sum(), 1.0)
    return (
        jnp.tensordot(sample_weight, p, axes=(0, 0)) / denominator,
        jnp.tensordot(sample_weight, q, axes=(0, 0)) / denominator,
    )


def _contract_factors(factors, sample_weight=None) -> Optional[jnp.ndarray]:
    """Batch-mean gradient from stacked per-sample factors.

    ``factors`` is ``None`` for the bottom element, or a ``(u, v)`` pair with
    a leading batch axis: ``(batch, n_children)`` and ``(batch, n_parents)``
    for a ``Layer``; ``(batch, n_slices, ...)`` for a ``LayerStack``. The
    mean over samples of ``u ⊗ v`` is computed as a single contraction.
    """
    if factors is None:
        return None
    u, v = factors
    if sample_weight is None:
        if u.ndim == 2:
            return jnp.einsum("bi,bj->ij", u, v) / u.shape[0]
        return jnp.einsum("bni,bnj->nij", u, v) / u.shape[0]
    # Weighted mean: the denominator is the weight the batch actually carries,
    # not the number of rows it happens to be padded to.
    denominator = jnp.maximum(sample_weight.sum(), 1.0)
    if u.ndim == 2:
        return jnp.einsum("b,bi,bj->ij", sample_weight, u, v) / denominator
    return jnp.einsum("b,bni,bnj->nij", sample_weight, u, v) / denominator


@eqx.filter_jit
def prediction_pass(network: Network, x: jnp.ndarray) -> jnp.ndarray:
    """Forward-only sweep through the network.

    Sets the predictors on the top element and runs the top-down prediction sweep —
    no prediction errors, posterior updates, or weight learning — returning the
    bottom element's ``expected_mean``. Used by
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


# ---------------------------------------------------------------------------
# DAG networks
# ---------------------------------------------------------------------------
#
# Continuous networks are DAGs rather than chains: each layer can have one
# value-parent layer and one volatility-parent layer, recorded on the *parent*
# as ``value_child_idx`` / ``volatility_child_idx``. The builder guarantees
# that every parent has a higher index than its children, so the prediction
# sweep runs top-down in descending index order and the update sweep runs
# bottom-up in ascending order.


def _assert_continuous_network(network: Network) -> None:
    """Check that every element is a continuous ``Layer`` (no stacks, no mixing)."""
    for i, elem in enumerate(network.layers):
        if isinstance(elem, LayerStack):
            raise NotImplementedError(
                "Continuous networks do not support LayerStack elements yet."
            )
        if elem.kind != "continuous":
            raise ValueError(
                f"Layer {i} has kind {elem.kind!r}: continuous sweeps require "
                "an all-continuous network."
            )


def _continuous_parent_maps(
    network: Network,
) -> tuple[list, list]:
    """Invert the child indices into per-layer parent indices.

    Returns ``(value_parent_of, volatility_parent_of)``, each one entry per layer
    holding the parent's index or ``None``.
    """
    n = len(network.layers)
    value_parent_of: list = [None] * n
    volatility_parent_of: list = [None] * n
    for j, elem in enumerate(network.layers):
        if elem.value_child_idx is not None:
            value_parent_of[elem.value_child_idx] = j
        if elem.volatility_child_idx is not None:
            volatility_parent_of[elem.volatility_child_idx] = j
    return value_parent_of, volatility_parent_of


def _continuous_prediction_sweep(
    network: Network,
    value_parent_of: list,
    volatility_parent_of: list,
    *,
    time_step: float = 1.0,
) -> Network:
    """Top-down prediction sweep over a continuous network.

    Predicts every layer from its value and volatility parents, in descending
    index order so parents are always predicted before their children. Nothing
    is clamped: observations enter in :func:`_continuous_update_sweep`. The
    parent maps come from :func:`_continuous_parent_maps`.
    """
    elements = list(network.layers)
    for i in range(len(elements) - 1, -1, -1):
        elem = elements[i]
        vp = value_parent_of[i]
        vlp = volatility_parent_of[i]
        new_state = vectorized_continuous_prediction(
            child_state=elem.state,
            params=elem.params,
            time_step=time_step,
            value_parent_state=None if vp is None else elements[vp].state,
            weights=None if vp is None else elements[vp].weights_mean,
            coupling_fn=None if vp is None else elements[vp].coupling_fn,
            volatility_parent_state=(None if vlp is None else elements[vlp].state),
            volatility_weights=(
                None if vlp is None else elements[vlp].volatility_weights
            ),
            is_static_leaf=elem.is_input_layer and vlp is None,
        )
        elements[i] = dataclasses.replace(elem, state=new_state)

    return dataclasses.replace(network, layers=tuple(elements))


def _continuous_update_sweep(
    network: Network,
    y: jnp.ndarray,
    value_parent_of: list,
    *,
    time_step: float = 1.0,
) -> Network:
    """Bottom-up prediction-error and posterior-update sweep.

    Clamps the observations on layer 0, computes its prediction errors, then
    walks the layers in ascending order: each layer's posterior integrates the
    prediction errors of its value and volatility children (already computed,
    since children carry lower indices), and its own prediction errors are then
    written for the parents above. ``value_parent_of`` comes from
    :func:`_continuous_parent_maps`; the volatility side is read off each
    layer's ``has_volatility_parent``.
    """
    elements = list(network.layers)

    # Clamp the observations and compute the leaf prediction errors.
    leaf_state = dataclasses.replace(elements[0].state, mean=y)
    leaf_state = vectorized_continuous_prediction_error(
        leaf_state, has_volatility_parent=elements[0].has_volatility_parent
    )
    elements[0] = dataclasses.replace(elements[0], state=leaf_state)

    for i in range(1, len(elements)):
        elem = elements[i]

        value_child = None
        if elem.value_child_idx is not None:
            child_elem = elements[elem.value_child_idx]
            value_child = ValueChild(
                state=child_elem.state,
                weights=elem.weights_mean,
                coupling_fn=elem.coupling_fn,
                # Only layer 0 is clamped, and the update loop below skips it,
                # so its posterior precision is the one that never moves.
                precision_is_clamped=child_elem.is_input_layer,
            )

        volatility_child = None
        if elem.volatility_child_idx is not None:
            child_elem = elements[elem.volatility_child_idx]
            volatility_child = VolatilityChild(
                state=child_elem.state,
                kappa=elem.volatility_weights,
                params=child_elem.params,
            )

        if value_child is None and volatility_child is None:
            # A layer nobody names as parent receives no message; nothing to do.
            continue

        new_state = vectorized_continuous_posterior_update(
            elem.state,
            value_child=value_child,
            volatility_child=volatility_child,
            volatility_updates=network.volatility_updates,
            time_step=time_step,
            max_posterior_precision=network.max_posterior_precision,
        )

        # Prediction errors are only needed when a parent above will read them.
        if value_parent_of[i] is not None or elem.has_volatility_parent:
            new_state = vectorized_continuous_prediction_error(
                new_state, has_volatility_parent=elem.has_volatility_parent
            )

        elements[i] = dataclasses.replace(elem, state=new_state)

    return dataclasses.replace(network, layers=tuple(elements))


@eqx.filter_jit
def run_continuous_scan(
    network: Network,
    ys: jnp.ndarray,
    time_steps: jnp.ndarray,
    record: tuple = (),
) -> tuple:
    """Filter a sequence of observations through a continuous network.

    Runs ``jax.lax.scan`` over (prediction sweep, update sweep) pairs. There
    is no weight learning: the coupling matrices are parameters of the filter.

    Parameters
    ----------
    network :
        The initial continuous network state.
    ys :
        Observations clamped on layer 0 at each step, shape ``(T, n_obs)``.
    time_steps :
        Per-step time steps, shape ``(T,)``.
    record :
        Tuple of ``LayerState`` field names to record at every step. With an
        empty tuple (default) the per-step output is layer 0's ``expected_mean``
        alone; otherwise it is ``(traj_step, prediction)``.

    Returns
    -------
    ``(final_network, step_output)`` with per-step outputs stacked along a
    leading ``(T,)`` axis.
    """
    # The topology is static, so validate and invert it once rather than once
    # per sweep inside the scan body.
    _assert_continuous_network(network)
    value_parent_of, volatility_parent_of = _continuous_parent_maps(network)

    def body(net, xs):
        y, dt = xs
        predicted = _continuous_prediction_sweep(
            net, value_parent_of, volatility_parent_of, time_step=dt
        )
        updated = _continuous_update_sweep(predicted, y, value_parent_of, time_step=dt)
        prediction = updated.layers[0].state.expected_mean
        if record:
            traj_step = {
                field: tuple(getattr(elem.state, field) for elem in updated.layers)
                for field in record
            }
            return updated, (traj_step, prediction)
        return updated, prediction

    return jax.lax.scan(body, network, (ys, time_steps))
