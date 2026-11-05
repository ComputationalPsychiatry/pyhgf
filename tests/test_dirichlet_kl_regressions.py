"""Finite Dirichlet KL values checked independently against SciPy."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from pyhgf.math import dirichlet_kullback_leibler

try:
    from jax import enable_x64
except ImportError:  # JAX versions before enable_x64 became a top-level export.
    from jax.experimental import enable_x64


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("scale", [0.1, 1.0, 100.0, 1000000.0])
def test_dirichlet_self_kl_is_finite_zero(x64, scale):
    """Identical valid distributions have zero KL even after gamma overflows."""
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        alpha = jnp.asarray([scale, scale, 2 * scale], dtype=dtype)
        actual = float(dirichlet_kullback_leibler(alpha, alpha))
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "alpha,beta",
    [
        ([0.2, 0.7], [0.9, 0.3]),
        ([1.0, 1.0], [2.0, 3.0]),
        ([100.0, 100.0], [80.0, 120.0]),
        ([1000.0, 2000.0, 3000.0], [1200.0, 2300.0, 2700.0]),
    ],
)
def test_dirichlet_kl_is_nonnegative_and_matches_scipy(x64, alpha, beta):
    """Log-gamma normalization agrees with an independent special-function library."""
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        a, b = np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64)
        expected = (
            special.gammaln(a.sum())
            - special.gammaln(b.sum())
            + np.sum(special.gammaln(b) - special.gammaln(a))
            + np.sum((a - b) * (special.digamma(a) - special.digamma(a.sum())))
        )
        actual = float(
            dirichlet_kullback_leibler(
                jnp.asarray(a, dtype=dtype), jnp.asarray(b, dtype=dtype)
            )
        )
        assert np.isfinite(actual) and actual >= 0.0
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=3e-4 if not x64 else 3e-12,
            atol=2e-3 if not x64 else 2e-11,
        )


@pytest.mark.parametrize("x64", [False, True])
def test_dirichlet_kl_matches_integrated_log_density_ratio(x64):
    """A two-dimensional case also agrees with quadrature over SciPy densities."""
    alpha, beta = np.array([2.0, 3.0]), np.array([4.0, 5.0])

    def integrand(x):
        point = np.array([x, 1.0 - x])
        p_log = stats.dirichlet.logpdf(point, alpha)
        return np.exp(p_log) * (p_log - stats.dirichlet.logpdf(point, beta))

    expected, error = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-11)
    assert error < 1e-8
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        actual = float(
            dirichlet_kullback_leibler(
                jnp.asarray(alpha, dtype=dtype), jnp.asarray(beta, dtype=dtype)
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
