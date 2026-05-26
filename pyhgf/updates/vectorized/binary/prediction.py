# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction for binary state node layers."""

from typing import Callable

import jax.numpy as jnp
from jax.nn import sigmoid

from pyhgf.typing import LayerState


def vectorized_binary_prediction(
    child_state: LayerState,
    parent_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
) -> LayerState:
    r"""Predict expected mean and precision for a binary state node layer.

    The expected mean is the sigmoid of the weighted, coupled parent predictions:

    .. math::

        \hat{\mu}_b = \sigma\!\left( W \, g(\hat{\mu}_a) \right),

    and the expected precision is the Bernoulli variance:

    .. math::

        \tilde{\pi}_b = \hat{\mu}_b \, (1 - \hat{\mu}_b).

    .. note::

        The ``expected_precision`` field actually stores the **variance** (i.e. the
        uncertainty) of the Bernoulli distribution. Reusing this name avoids the
        need for a separate posterior update step for binary nodes: the continuous
        parent's posterior update can consume the binary prediction error directly.
        Binary leaves carry no AR-volatility random walk, so the conditional
        predicted precision :math:`\hat{\pi}_b` (``conditional_expected_precision``)
        is set equal to :math:`\tilde{\pi}_b` — matching the paper's Limit 3 leaf
        convention used by the parent's posterior-step (smoothing) update.

    Parameters
    ----------
    child_state :
        Current state of the binary child layer (being predicted).
    parent_state :
        Current state of the parent layer (predictor).
    weights :
        Weight matrix connecting child to parent, shape ``(n_children, n_parents)``
        or ``(n_children, n_parents + 1)`` when the parent layer includes a
        constant input node.
    coupling_fn :
        Coupling function applied to parent means (default: :func:`jax.numpy.tanh`).
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0)
        appended to its activations; the corresponding column of *weights* carries
        the bias connections and is treated as linearly coupled (:math:`g(1) = 1`).

    Returns
    -------
    LayerState
        Updated child layer state with binary expected values.
    """
    # Apply coupling to the parent activations only; the constant bias node is
    # always wired in linearly (g(1) = 1) regardless of coupling_fn.
    coupled_parents = coupling_fn(parent_state.expected_mean)
    if parent_has_constant:
        coupled_parents = jnp.concatenate([coupled_parents, jnp.ones(1)])
    logit = jnp.matmul(weights, coupled_parents)

    # Sigmoid transform to get binary expected mean
    expected_mean = sigmoid(logit)
    expected_mean = jnp.clip(expected_mean, 1e-6, 1 - 1e-6)

    # Binary variance = μ̂(1 − μ̂), stored as "expected_precision"
    expected_precision = expected_mean * (1 - expected_mean)

    return child_state._replace(
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        # Binary leaves carry no AR-volatility random walk, so the conditional
        # precision coincides with the marginal — matches the Limit 3 leaf
        # convention used by the parent's posterior update.
        conditional_expected_precision=expected_precision,
    )
