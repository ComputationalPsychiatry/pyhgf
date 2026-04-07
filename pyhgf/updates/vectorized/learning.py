# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized weight learning for deep predictive coding networks."""

from typing import Callable, Optional

import jax.numpy as jnp

from pyhgf.typing import LayerState


def vectorized_weight_update(
    parent_state: LayerState,
    child_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    lr: Optional[float] = None,
) -> jnp.ndarray:
    r"""Unified weight update for vectorized layers.

    Branches on the ``lr`` parameter:

    - **Fixed** (``lr`` is a float):
      :math:`\Delta w = \text{lr} \cdot \text{PE} \cdot \pi_\text{child}
      \cdot g(\text{parent})`
    - **Dynamic** (``lr is None``): Kalman-gain rule.
      :math:`K = \pi_\text{parent} / (\pi_\text{parent} + \pi_\text{child})`
      :math:`\Delta w = K \cdot \text{PE} \cdot g(\text{parent})`

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
        Fixed learning rate.  When ``None`` (default) the dynamic
        precision-weighted Kalman-gain rule is used instead.

    Returns
    -------
    jnp.ndarray
        Updated weight matrix.
    """
    # Prediction error at child layer
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation
    coupled_parent = coupling_fn(parent_state.mean)

    # Base outer product: PE ⊗ g(parent)
    # Broadcast: (n_children, 1) * (1, n_parents) -> (n_children, n_parents)
    coupling_delta = pe[:, None] * coupled_parent[None, :]

    if lr is not None:
        # Fixed LR: scale by lr · π_child
        coupling_delta = coupling_delta * child_state.precision[:, None] * lr
    else:
        # Dynamic: scale by Kalman gain K = π_parent / (π_parent + π_child)
        kalman_gain = parent_state.precision[None, :] / (
            parent_state.precision[None, :] + child_state.precision[:, None]
        )
        coupling_delta = coupling_delta * kalman_gain

    # Guard against NaN / inf
    coupling_delta = jnp.where(
        jnp.isnan(coupling_delta) | jnp.isinf(coupling_delta), 0.0, coupling_delta
    )

    new_weights = weights + coupling_delta
    new_weights = jnp.where(jnp.isinf(new_weights), weights, new_weights)

    return new_weights
