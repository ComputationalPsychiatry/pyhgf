# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Unit checks for the separable weight-update gradient."""

import jax
import jax.numpy as jnp
import pytest

from pyhgf.typing.vectorised import LayerState
from pyhgf.updates.vectorized.learning import vectorized_weight_gradient


@pytest.fixture(autouse=True)
def _x64():
    """Enable float64 for these exactness checks, then restore the session default.

    The tolerances below are at the float64 floor; leaving x64 on would leak into the
    shared pytest session and shift dtypes in unrelated tests.
    """
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _state(mean, precision, expected_mean, expected_precision):
    """Minimal value-level LayerState (volatility fields unused here)."""
    return LayerState(
        mean=mean,
        precision=precision,
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        conditional_expected_precision=expected_precision,
        effective_precision=expected_precision,
        value_prediction_error=jnp.zeros_like(mean),
        mean_vol=None,
        precision_vol=None,
        expected_mean_vol=None,
        expected_precision_vol=None,
        effective_precision_vol=None,
        volatility_prediction_error=None,
    )


def test_precision_weighted_uses_posterior_precision():
    """`precision_weighted` weights by the posterior precision, not the predicted one.

    A moved interior belief shifts by (routed error) / posterior precision, so weighting
    by that same posterior precision cancels the division and recovers backprop at any
    precision setting.
    """
    d_p, d_c = 5, 4
    k = jax.random.split(jax.random.PRNGKey(7), 6)
    a = jax.random.normal(k[0], (d_p,))
    pi_p = jnp.exp(jax.random.normal(k[1], (d_p,)))
    x = jax.random.normal(k[2], (d_c,))
    xhat = jax.random.normal(k[3], (d_c,))
    pi_post = jnp.exp(jax.random.normal(k[4], (d_c,)))  # posterior
    pi_pred = jnp.exp(jax.random.normal(k[5], (d_c,)))  # predicted (expected)
    parent = _state(a, pi_p, a, pi_p)
    child = _state(x, pi_post, xhat, pi_pred)  # distinct posterior vs predicted
    got = vectorized_weight_gradient(
        parent, child, lambda z: z, kind="precision_weighted"
    )
    ref = -((x - xhat) * pi_post)[:, None] * a[None, :]  # descent, posterior precision
    assert jnp.allclose(got, ref, atol=1e-12)
    wrong = -((x - xhat) * pi_pred)[:, None] * a[None, :]
    assert not jnp.allclose(got, wrong, atol=1e-3)
