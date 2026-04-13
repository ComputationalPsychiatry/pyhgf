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
    parent_has_constant: bool = False,
    adam_m: Optional[jnp.ndarray] = None,
    adam_v: Optional[jnp.ndarray] = None,
    adam_t: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    r"""Unified weight update for vectorized layers.

    Branches on the ``lr`` parameter:

    - **Fixed** (``lr`` is a float):
      :math:`\Delta w = \text{lr} \cdot \text{PE} \cdot \pi_\text{child}
      \cdot g(\text{parent})`
    - **Dynamic** (``lr is None``): Kalman-gain rule.
      :math:`K = \pi_\text{parent} / (\pi_\text{parent} + \pi_\text{child})`
      :math:`\Delta w = K \cdot \text{PE} \cdot g(\text{parent})`

    When ``adam_m`` and ``adam_v`` are provided (not *None*), the fixed-LR gradient is
    filtered through Adam before being applied.

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations).
    weights :
        Current weight matrix, shape ``(n_children, n_parents)`` or
        ``(n_children, n_parents + 1)`` when the parent layer includes
        a constant input node.
    coupling_fn :
        Coupling function applied to parent means.
    lr :
        Fixed learning rate.  When ``None`` (default) the dynamic
        precision-weighted Kalman-gain rule is used instead.
    parent_has_constant :
        If True, the parent layer has a constant input node.  The parent
        mean is augmented with 1.0 and the precision with the parent
        precision mean so the bias column of *weights* is updated.
    adam_m :
        First moment estimate for Adam, same shape as *weights*.
        Pass ``None`` to disable Adam.
    adam_v :
        Second moment estimate for Adam, same shape as *weights*.
    adam_t :
        Global Adam timestep (already incremented for this step).
    adam_beta1 :
        Exponential decay rate for the first moment.
    adam_beta2 :
        Exponential decay rate for the second moment.
    adam_epsilon :
        Small constant for numerical stability.

    Returns
    -------
    new_weights :
        Updated weight matrix.
    new_adam_m :
        Updated first moment (or *None* when Adam is not used).
    new_adam_v :
        Updated second moment (or *None* when Adam is not used).
    """
    # Prediction error at child layer
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation
    parent_mean = parent_state.mean
    parent_precision = parent_state.precision
    if parent_has_constant:
        parent_mean = jnp.concatenate([parent_mean, jnp.ones(1)])
        parent_precision = jnp.concatenate([
            parent_precision,
            jnp.array([jnp.mean(parent_precision)]),
        ])
    coupled_parent = coupling_fn(parent_mean)

    # Base outer product: PE ⊗ g(parent)
    coupling_delta = pe[:, None] * coupled_parent[None, :]

    if lr is not None:
        # Raw gradient (before LR scaling)
        gradient = coupling_delta * child_state.precision[:, None]

        if adam_m is not None and adam_v is not None:
            # Adam-filtered update
            new_m = adam_beta1 * adam_m + (1.0 - adam_beta1) * gradient
            new_v = adam_beta2 * adam_v + (1.0 - adam_beta2) * gradient**2
            m_hat = new_m / (1.0 - adam_beta1**adam_t)
            v_hat = new_v / (1.0 - adam_beta2**adam_t)
            coupling_delta = lr * m_hat / (jnp.sqrt(v_hat) + adam_epsilon)
        else:
            coupling_delta = gradient * lr
            new_m = None
            new_v = None
    else:
        # Dynamic: Kalman gain (Adam not applicable)
        kalman_gain = parent_precision[None, :] / (
            parent_precision[None, :] + child_state.precision[:, None]
        )
        coupling_delta = coupling_delta * kalman_gain
        new_m = None
        new_v = None

    # Guard against NaN / inf
    coupling_delta = jnp.where(
        jnp.isnan(coupling_delta) | jnp.isinf(coupling_delta), 0.0, coupling_delta
    )

    new_weights = weights + coupling_delta
    new_weights = jnp.where(jnp.isinf(new_weights), weights, new_weights)

    return new_weights, new_m, new_v
