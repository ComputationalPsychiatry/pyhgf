# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

r"""Relaxed value-coupling updates of a layer whose child nodes have several parents.

A child node :math:`x_a = \sum_m W_{a,m} x_m + \varepsilon` with several value parents
:math:`m` passes each parent the message of the joint Gaussian of :math:`(x_a, x_m)` in
which the other parents of :math:`a` are integrated over their predicted beliefs. With
linear coupling, this message gives every parent its exact marginal posterior.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyhgf.math import leave_one_out_sum
from pyhgf.typing.vectorised import LayerState
from pyhgf.updates.vectorised.volatile.posterior import (
    vectorised_layer_posterior_update,
)

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64

WEIGHTS = np.array([[0.8, -1.3, 0.5]])
PARENT_PRECISIONS = np.array([1.2, 0.7, 2.0])
CONDITIONAL_PRECISION = 1.5
EVIDENCE_PRECISION = 2.5
RESIDUAL = 0.9


def _exact_parent_posteriors(
    evidence_precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Marginal posterior precision and mean shift of every parent, by conditioning.

    The child :math:`x_a = W x + \varepsilon`, with :math:`\varepsilon` of variance
    :math:`1/\bar{\pi}_a`, is observed with noise of variance :math:`1/\pi_y`, at
    ``RESIDUAL`` from its prediction.
    """
    weights = WEIGHTS[0]
    marginal_variance = 1 / CONDITIONAL_PRECISION + np.sum(
        weights**2 / PARENT_PRECISIONS
    )
    total_variance = marginal_variance + 1 / evidence_precision
    covariance = weights / PARENT_PRECISIONS
    variance = 1 / PARENT_PRECISIONS - covariance**2 / total_variance
    return 1 / variance, covariance / total_variance * RESIDUAL


def _child_and_parent(clamped: bool) -> tuple[LayerState, LayerState]:
    """Child and parent layer states for one observation of the child."""
    weights = WEIGHTS[0]
    marginal = 1 / (1 / CONDITIONAL_PRECISION + np.sum(weights**2 / PARENT_PRECISIONS))
    if clamped:
        # A clamped leaf keeps its stored precision at the conditional predicted
        # precision, and its prediction error is the raw residual.
        precision = expected = CONDITIONAL_PRECISION
        prediction_error = RESIDUAL
    else:
        expected = marginal
        precision = marginal + EVIDENCE_PRECISION
        prediction_error = EVIDENCE_PRECISION * RESIDUAL / precision
    child = dataclasses.replace(
        LayerState.create(1, has_volatility_parent=False),
        precision=jnp.array([precision]),
        expected_precision=jnp.array([expected]),
        conditional_expected_precision=jnp.array([CONDITIONAL_PRECISION]),
        value_prediction_error=jnp.array([prediction_error]),
    )
    parent = dataclasses.replace(
        LayerState.create(3, has_volatility_parent=False),
        expected_mean=jnp.zeros(3),
        expected_precision=jnp.asarray(PARENT_PRECISIONS),
    )
    return child, parent


def test_leave_one_out_sum_keeps_precision_with_a_dominant_entry():
    """Each entry is the sum of the other columns of its row, without cancellation."""
    values = np.array(
        [[1e6, 1e-3, 2e-3, 3e-3], [0.5, 1.5, 0.25, 3.0]], dtype=np.float32
    )
    expected = np.array([
        [sum(row) - row[j] for j in range(len(row))] for row in values.astype(float)
    ])
    out = np.asarray(leave_one_out_sum(jnp.asarray(values)))
    # Subtracting 1e-3 from a float32 total of about 1e6 would lose it entirely.
    assert np.allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize("clamped", [False, True], ids=["interior", "clamped"])
def test_integrating_other_parents_is_exact_for_linear_coupling(clamped):
    """Each parent of a child with three linear parents gets its exact posterior.

    Conditioning the child on every parent instead treats the other two as known, and
    every parent gains more precision than the exact posterior gives it.
    """
    with enable_x64(True):
        child, parent = _child_and_parent(clamped)
        evidence = np.inf if clamped else EVIDENCE_PRECISION
        exact_precision, exact_shift = _exact_parent_posteriors(evidence)

        updated = vectorised_layer_posterior_update(
            parent,
            child,
            jnp.asarray(WEIGHTS),
            lambda x: x,
            child_is_input_layer=clamped,
            integrate_other_parents=True,
        )
        assert np.allclose(np.asarray(updated.precision), exact_precision, rtol=1e-10)
        assert np.allclose(np.asarray(updated.mean), exact_shift, rtol=1e-10)

        conditioned_on_all = vectorised_layer_posterior_update(
            parent,
            child,
            jnp.asarray(WEIGHTS),
            lambda x: x,
            child_is_input_layer=clamped,
        )
        assert np.all(np.asarray(conditioned_on_all.precision) > exact_precision)


def test_integrating_other_parents_with_curved_coupling():
    """With a ``tanh`` coupling, the parents enter through their Taylor terms.

    The predicted precision of the child given one parent integrates the others with
    the derivative of the coupling at their predicted means, and the posterior
    precision keeps the second-order term weighted by the relaxed gain. The values are
    those of the Rust kernel test ``test_integrating_other_parents_with_curved_coupling``.
    """
    with enable_x64(True):
        child = dataclasses.replace(
            LayerState.create(2, has_volatility_parent=False),
            precision=jnp.array([3.0, 2.4]),
            expected_precision=jnp.array([2.0, 1.5]),
            conditional_expected_precision=jnp.array([2.5, 1.8]),
            value_prediction_error=jnp.array([0.4, -0.7]),
        )
        parent = dataclasses.replace(
            LayerState.create(3, has_volatility_parent=False),
            expected_mean=jnp.array([0.3, -0.6, 1.1]),
            expected_precision=jnp.array([1.2, 0.9, 2.2]),
        )
        weights = jnp.array([[0.8, -0.5, 0.3], [-0.4, 0.9, 0.6]])

        updated = vectorised_layer_posterior_update(
            parent, child, weights, jax.numpy.tanh, integrate_other_parents=True
        )

        # Direct evaluation of the relaxed update, one edge at a time.
        mu_hat = np.array([0.3, -0.6, 1.1])
        pi_hat = np.array([1.2, 0.9, 2.2])
        g1 = 1 - np.tanh(mu_hat) ** 2
        g2 = -2 * np.tanh(mu_hat) * g1
        w = np.asarray(weights)
        precision = pi_hat.copy()
        message = np.zeros(3)
        for a, (pi_a, pihat_a, pibar_a, delta) in enumerate([
            (3.0, 2.0, 2.5, 0.4),
            (2.4, 1.5, 1.8, -0.7),
        ]):
            pi_y = pi_a - pihat_a
            tau = (w[a] * g1) ** 2 / pi_hat
            for m in range(3):
                pibar_am = 1 / (1 / pibar_a + tau.sum() - tau[m])
                gain = pibar_am * pi_a / (pibar_am + pi_y)
                precision[m] += (
                    pibar_am * pi_y / (pibar_am + pi_y) * (w[a, m] * g1[m]) ** 2
                    - gain * w[a, m] * g2[m] * delta
                )
                message[m] += w[a, m] * gain * delta
        mean = mu_hat + message * g1 / np.maximum(precision, pi_hat)
        assert np.allclose(np.asarray(updated.precision), np.maximum(precision, pi_hat))
        assert np.allclose(np.asarray(updated.mean), mean)
