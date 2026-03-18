# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized weight learning for deep predictive coding networks."""

from typing import Callable

import jax.numpy as jnp

from pyhgf.model.vectorized_types import LayerState


def vectorized_weight_update(
    parent_state: LayerState,
    child_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    lr: float,
) -> jnp.ndarray:
    """Update weights using prospective configuration.

    This computes the weight update based on the prediction error and
    the coupled parent activation: delta = pe * coupling_fn(parent_mean).

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations).
    weights :
        Current weight matrix, shape (n_children, n_parents).
    coupling_fn :
        Coupling function applied to parent means.
    lr :
        Learning rate for weight updates.

    Returns
    -------
    jnp.ndarray
        Updated weight matrix.
    """
    # Prediction error at child layer
    # child_state.mean shape: (n_children,)
    # child_state.expected_mean shape: (n_children,)
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation
    # parent_state.mean shape: (n_parents,)
    coupled_parent = coupling_fn(parent_state.mean)

    # Weight delta: pe * coupling_fn(parent_mean)
    # Broadcast: (n_children, 1) * (1, n_parents) -> (n_children, n_parents)
    coupling_delta = pe[:, None] * coupled_parent[None, :]

    # Handle NaN and inf
    coupling_delta = jnp.where(jnp.isnan(coupling_delta), 0.0, coupling_delta)
    coupling_delta = jnp.where(jnp.isinf(coupling_delta), 0.0, coupling_delta)

    # Update with learning rate (no weighting by number of parents)
    new_weights = weights + coupling_delta * lr

    # Handle inf values in result
    new_weights = jnp.where(jnp.isinf(new_weights), weights, new_weights)

    return new_weights


def vectorized_weight_update_dynamic(
    parent_state: LayerState,
    child_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
) -> jnp.ndarray:
    """Update weights using dynamic (precision-weighted) learning rate.

    This uses the precision ratio between child and parent as a
    natural learning rate.

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations).
    weights :
        Current weight matrix, shape (n_children, n_parents).
    coupling_fn :
        Coupling function applied to parent means.

    Returns
    -------
    jnp.ndarray
        Updated weight matrix.
    """
    # Prediction error at child layer
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation
    coupled_parent = coupling_fn(parent_state.mean)

    # Weight delta: pe * coupling_fn(parent_mean)
    coupling_delta = pe[:, None] * coupled_parent[None, :]
    coupling_delta = jnp.where(jnp.isnan(coupling_delta), 0.0, coupling_delta)
    coupling_delta = jnp.where(jnp.isinf(coupling_delta), 0.0, coupling_delta)

    # Dynamic learning rate: precision weighting
    # (n_children, 1) / (1, n_parents) -> (n_children, n_parents)
    precision_weighting = (
        child_state.precision[:, None] / parent_state.precision[None, :]
    )

    # No weighting by number of parents
    new_weights = weights + coupling_delta * precision_weighting

    new_weights = jnp.where(jnp.isinf(new_weights), weights, new_weights)

    return new_weights
