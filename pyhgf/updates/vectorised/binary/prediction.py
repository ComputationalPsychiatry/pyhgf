# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised prediction for binary state node layers."""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax.nn import sigmoid

from pyhgf.coupling import Coupling
from pyhgf.typing.vectorised import LayerState


def vectorised_binary_prediction(
    child_state: LayerState,
    parent_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
    precision_clipping_value: float = 1e-6,
    coupling_moments: str = "taylor",
) -> LayerState:
    r"""Predict expected mean and precision for a binary state node layer.

    The expected mean is the sigmoid of the weighted, coupled parent predictions:

    .. math::

        \hat{\mu}_b = \sigma\!\left( W \, g(\hat{\mu}_a) \right),

    and the expected precision is the Bernoulli variance:

    .. math::

        \hat{\pi}_b = \hat{\mu}_b \, (1 - \hat{\mu}_b).

    .. note::

        The ``expected_precision`` field actually stores the **variance** (i.e. the
        uncertainty) of the Bernoulli distribution. Reusing this name avoids the
        need for a separate posterior update step for binary nodes: the continuous
        parent's posterior update can consume the binary prediction error directly.
        Binary leaves carry no AR-volatility random walk, so the conditional
        predicted precision :math:`\bar{\pi}_b` (``conditional_expected_precision``)
        is set equal to :math:`\hat{\pi}_b`, the leaf convention used by the
        parent's posterior-step relaxation.

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
    precision_clipping_value :
        The expected mean is clipped to
        ``[precision_clipping_value, 1 - precision_clipping_value]``.
    coupling_moments :
        ``"taylor"`` (default) evaluates the coupling function at the expected means
        of the parent layer. ``"exact"`` takes its mean
        :math:`\mathbb{E}[g(x_a)]` under the predicted belief
        :math:`x_a \sim \mathcal{N}(\hat{\mu}_a, 1/\hat{\pi}_a)` of each parent node
        (:meth:`pyhgf.coupling.Coupling.moments`).

    Returns
    -------
    LayerState
        Updated child layer state with binary expected values.
    """
    # Apply coupling to the parent activations only; the constant bias node is
    # always wired in linearly (g(1) = 1) regardless of coupling_fn.
    coupled_parents, _, _ = Coupling.from_callable(coupling_fn).linearise(
        parent_state.expected_mean,
        1.0 / parent_state.expected_precision,
        coupling_moments,
    )
    if parent_has_constant:
        coupled_parents = jnp.concatenate([coupled_parents, jnp.ones(1)])
    logit = jnp.matmul(weights, coupled_parents)

    # Sigmoid transform to get binary expected mean
    expected_mean = sigmoid(logit)
    # Bound away from 0/1 for numerical stability (configurable via
    # ``DeepNetwork(precision_clipping_value=...)``): a larger value (e.g. 1e-3,
    # matching the TAPAS HGF Toolbox) keeps the binary predicted precision from
    # collapsing the level-2 update in high-volatility regimes; a very small value
    # avoids flat, zero-gradient plateaus that hurt gradient-based inference.
    expected_mean = jnp.clip(
        expected_mean, precision_clipping_value, 1 - precision_clipping_value
    )

    # Binary variance = μ̂(1 − μ̂), stored as "expected_precision"
    expected_precision = expected_mean * (1 - expected_mean)

    return dataclasses.replace(
        child_state,
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        # Binary leaves carry no AR-volatility random walk, so the conditional
        # precision coincides with the marginal, the leaf convention used by the
        # parent's posterior update.
        conditional_expected_precision=expected_precision,
    )
