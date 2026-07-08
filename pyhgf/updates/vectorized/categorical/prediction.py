# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorized prediction for categorical state node layers."""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerState


def vectorized_categorical_prediction(
    child_state: LayerState,
    parent_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
) -> LayerState:
    r"""Predict a categorical state node layer: one belief per class.

    The layer's nodes jointly represent a single categorical choice. The incoming linear
    prediction plays the role of the logits, and the expected mean is their softmax
    across the layer:

    .. math::

        \hat{\mu} = \mathrm{softmax}(W \, g(\hat{\mu}_{parent}))

    All precision fields are set to one. This convention makes the layer compose with
    the existing kernels without special cases: with a one-hot observation clamped, the
    value prediction error is the raw residual ``one_hot - softmax(logits)`` (the
    cross-entropy gradient in logit space) the smoothing gain is exactly one (posterior
    precision equals the expected precision), so the message routed to the parent and
    the weight gradients coincide with the cross-entropy backpropagation quantities.

    The softmax couples the nodes within the single categorical layer: each class's
    expected mean depends on every class's logit through the shared normalization, so
    the nodes are not independently local. The layer acts as one joint unit. That unit
    is still local with respect to the rest of the network: it reads only its parent's
    state and routes its residual back to that parent. No global normalization or
    non-local dependency exists beyond the within-layer softmax.

    Parameters
    ----------
    child_state :
        Current state of the categorical child layer (being predicted).
    parent_state :
        Current state of the parent layer (predictor).
    weights :
        Weight matrix connecting child to parent, shape``(n_children, n_parents)`` or
        ``(n_children, n_parents + 1)`` when the parent layer includes a constant input
        node.
    coupling_fn :
        Coupling function applied to parent means.
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0) appended to its
        activations; the corresponding column of *weights* carries the bias connections
        and is treated as linearly coupled (:math:`g(1) = 1`).

    Returns
    -------
    LayerState
        Updated child layer state with the categorical expected values.
    """
    # Apply coupling to the parent activations only; the constant bias node is
    # always wired in linearly (g(1) = 1) regardless of coupling_fn.
    coupled_parents = coupling_fn(parent_state.expected_mean)
    if parent_has_constant:
        coupled_parents = jnp.concatenate([coupled_parents, jnp.ones(1)])
    logits = jnp.matmul(weights, coupled_parents)

    expected_mean = jax.nn.softmax(logits)
    ones = jnp.ones_like(expected_mean)

    return dataclasses.replace(
        child_state,
        expected_mean=expected_mean,
        expected_precision=ones,
        conditional_expected_precision=ones,
        effective_precision=jnp.zeros_like(expected_mean),
    )
