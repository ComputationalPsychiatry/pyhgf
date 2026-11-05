"""Independent normalization and likelihood checks for Gaussian densities."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from pyhgf.math import (
    binary_surprise_finite_precision,
    gaussian_density,
    gaussian_surprise,
)

try:
    from jax import enable_x64
except ImportError:  # JAX versions before enable_x64 became a top-level export.
    from jax.experimental import enable_x64


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("precision", [0.25, 1.0, 4.0, 100.0])
def test_gaussian_density_integrates_to_one(x64, precision):
    """Changing inverse variance must not change the total probability mass."""
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        mean, sigma = 0.7, 1.0 / np.sqrt(precision)
        grid = np.linspace(mean - 10 * sigma, mean + 10 * sigma, 20001)
        density = np.asarray(
            gaussian_density(jnp.asarray(grid, dtype=dtype), mean, precision)
        )
        mass = integrate.simpson(density.astype(np.float64), x=grid)
        np.testing.assert_allclose(mass, 1.0, atol=2e-6, rtol=0.0)


@pytest.mark.parametrize("x64", [False, True])
def test_gaussian_density_matches_scipy_and_negative_log_likelihood(x64):
    """Density and surprise describe the same normalized normal distribution."""
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        precision = jnp.asarray([0.25, 1.0, 4.0, 100.0], dtype=dtype)
        mean = jnp.asarray([-0.5, 0.0, 0.3, 1.0], dtype=dtype)
        x = mean + jnp.asarray([-1.5, -0.2, 0.7, 1.2], dtype=dtype) / jnp.sqrt(
            precision
        )
        actual = np.asarray(gaussian_density(x, mean, precision))
        expected = stats.norm.pdf(
            np.asarray(x),
            loc=np.asarray(mean),
            scale=1.0 / np.sqrt(np.asarray(precision)),
        )
        tolerance = 3e-6 if not x64 else 3e-13
        np.testing.assert_allclose(actual, expected, rtol=tolerance)
        np.testing.assert_allclose(
            actual,
            np.asarray(jnp.exp(-gaussian_surprise(x, mean, precision))),
            rtol=tolerance,
        )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("precision", [0.25, 4.0, 16.0])
def test_finite_precision_binary_surprise_matches_normal_mixture(x64, precision):
    """Finite-precision binary likelihoods retain both mixture normalization and scale."""
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        value = jnp.asarray([-0.5, 0.5, 1.5], dtype=dtype)
        probability = jnp.asarray([0.2, 0.8, 0.5], dtype=dtype)
        actual = np.asarray(
            binary_surprise_finite_precision(value, probability, precision)
        )
        sigma = 1.0 / np.sqrt(precision)
        expected_density = np.asarray(probability) * stats.norm.pdf(
            np.asarray(value), loc=1.0, scale=sigma
        ) + (1 - np.asarray(probability)) * stats.norm.pdf(
            np.asarray(value), loc=0.0, scale=sigma
        )
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(
            actual, -np.log(expected_density), rtol=3e-6, atol=3e-7
        )
        grid = np.linspace(-10 * sigma, 1.0 + 10 * sigma, 20001)
        mixture = np.asarray(
            jnp.exp(
                -binary_surprise_finite_precision(
                    jnp.asarray(grid, dtype=dtype), 0.3, precision
                )
            )
        )
        np.testing.assert_allclose(
            integrate.simpson(mixture.astype(np.float64), x=grid),
            1.0,
            rtol=0.0,
            atol=2e-6,
        )
