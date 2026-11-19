# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

r"""Relaxed value-coupling updates of a layer whose child nodes have several parents.

A child node :math:`x_a = \sum_m W_{a,m} x_m + \varepsilon` with several value parents
:math:`m` passes each parent the message of the joint Gaussian of :math:`(x_a, x_m)` in
which the other parents of :math:`a` are integrated over their predicted beliefs. With
linear coupling, this message gives every parent its exact marginal posterior from the
evidence of this child. The messages of several children add up to the exact marginal
posterior when no two of them share another parent, and to an approximation between two
fully connected layers.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork

from pyhgf.coupling import resolve_coupling_fn
from pyhgf.math import leave_one_out_sum
from pyhgf.model import DeepNetwork as JaxDeepNetwork
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


@pytest.mark.parametrize("coupling_moments", ["taylor", "exact"])
def test_integrating_other_parents_with_curved_coupling(coupling_moments):
    """With a ``tanh`` coupling, the parents enter through their linearised slopes.

    The predicted precision of the child given one parent integrates the others with
    the slope that the prediction step gives them: the derivative of the coupling at
    their predicted means, or its mean slope under their predicted beliefs with the
    exact coupling moments. The posterior precision is first-order in the coupling
    function. The values are those of the Rust kernel tests
    ``test_integrating_other_parents_with_curved_coupling`` and
    ``test_integrating_other_parents_with_exact_moments``.
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
            parent,
            child,
            weights,
            jax.numpy.tanh,
            coupling_moments=coupling_moments,
            integrate_other_parents=True,
        )

        # Direct evaluation of the relaxed update, one edge at a time.
        mu_hat = np.array([0.3, -0.6, 1.1])
        pi_hat = np.array([1.2, 0.9, 2.2])
        _, slope, _ = resolve_coupling_fn("tanh").linearise(
            jnp.asarray(mu_hat), 1 / jnp.asarray(pi_hat), coupling_moments
        )
        slope = np.asarray(slope)
        # The mean slope differs from the derivative at the predicted mean.
        derivative = 1 - np.tanh(mu_hat) ** 2
        assert np.allclose(slope, derivative) == (coupling_moments == "taylor")
        w = np.asarray(weights)
        precision = pi_hat.copy()
        message = np.zeros(3)
        for a, (pi_a, pihat_a, pibar_a, delta) in enumerate([
            (3.0, 2.0, 2.5, 0.4),
            (2.4, 1.5, 1.8, -0.7),
        ]):
            pi_y = pi_a - pihat_a
            tau = (w[a] * slope) ** 2 / pi_hat
            for m in range(3):
                pibar_am = 1 / (1 / pibar_a + tau.sum() - tau[m])
                gain = pibar_am * pi_a / (pibar_am + pi_y)
                precision[m] += (
                    pibar_am * pi_y / (pibar_am + pi_y) * (w[a, m] * slope[m]) ** 2
                )
                message[m] += w[a, m] * gain * delta
        mean = mu_hat + message * slope / precision
        assert np.allclose(np.asarray(updated.precision), precision)
        assert np.allclose(np.asarray(updated.mean), mean)


def test_children_sharing_parents_get_an_approximate_posterior():
    """Between two fully connected linear layers, the update is an approximation.

    Each edge integrates the other parents of its child over their predicted beliefs and
    leaves out what the other children say about them. With two clamped children that
    both depend on two parents through the same weights, the two observations carry the
    evidence of one about each parent, and the update counts both: the posterior
    precision is above the exact one. With opposite weights on the second parent, the
    two observations together determine each parent, and the update is below the exact
    posterior precision. Treating the other parents as known gives a higher precision
    than the update in both cases, and never less than the exact one.
    """
    parent_precisions = np.array([1.0, 1.0])
    noise_precision = 100.0
    with enable_x64(True):
        child = dataclasses.replace(
            LayerState.create(2, has_volatility_parent=False),
            precision=jnp.full(2, noise_precision),
            expected_precision=jnp.full(2, noise_precision),
            conditional_expected_precision=jnp.full(2, noise_precision),
            value_prediction_error=jnp.zeros(2),
        )
        parent = dataclasses.replace(
            LayerState.create(2, has_volatility_parent=False),
            expected_mean=jnp.zeros(2),
            expected_precision=jnp.asarray(parent_precisions),
        )
        for second_row, compare in (([1.0, 1.0], np.greater), ([1.0, -1.0], np.less)):
            weights = np.array([[1.0, 1.0], second_row])
            exact_precision = 1 / np.diag(
                np.linalg.inv(
                    np.diag(parent_precisions) + noise_precision * weights.T @ weights
                )
            )
            precisions = {
                integrate: np.asarray(
                    vectorised_layer_posterior_update(
                        parent,
                        child,
                        jnp.asarray(weights),
                        lambda x: x,
                        child_is_input_layer=True,
                        integrate_other_parents=integrate,
                    ).precision
                )
                for integrate in (True, False)
            }
            assert np.all(compare(precisions[True], exact_precision))
            assert np.all(precisions[False] > precisions[True])
            assert np.all(precisions[False] >= exact_precision * (1 - 1e-12))


@pytest.mark.parametrize("cls", [JaxDeepNetwork, RsDeepNetwork], ids=["jax", "rust"])
def test_frozen_precisions_treat_the_other_parents_as_known(cls):
    """Without ``predict_precision``, ``feedforward_uncertainty`` changes nothing.

    The prediction step leaves every predicted precision at its prior and adds no
    variance from the value parents, so the posterior step conditions each child on all
    its parents, as it does without ``feedforward_uncertainty``.
    """
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))

    def run(feedforward_uncertainty):
        net = cls(
            coupling_fn="tanh",
            feedforward_uncertainty=feedforward_uncertainty,
            predict_precision=False,
        )
        for size in (2, 4, 4, 3):
            if cls is RsDeepNetwork:
                net = net.add_layer(size, volatility_parent=False)
            else:
                net = net.add_layer(size=size, volatility_parent=False)
        if cls is RsDeepNetwork:
            weights_rng = np.random.default_rng(1)
            net.set_weights([
                weights_rng.normal(scale=0.5, size=w.shape) for w in net.get_weights()
            ])
            predictions = net.fit(x, y, optimiser="sgd", learning_rate=0.05)
            return np.asarray(predictions), [np.asarray(w) for w in net.get_weights()]
        net.fit(x, y, optax.sgd(0.05))
        return np.asarray(net.predictions), [
            np.asarray(layer.weights_mean) for layer in net.state.layers[1:]
        ]

    (predictions, weights), (reference, reference_weights) = run(True), run(False)
    assert np.all(np.isfinite(predictions))
    np.testing.assert_array_equal(predictions, reference)
    for updated, expected in zip(weights, reference_weights):
        np.testing.assert_array_equal(updated, expected)
