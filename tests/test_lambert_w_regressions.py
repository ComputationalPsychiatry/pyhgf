"""Principal Lambert W checks spanning finite float32 and float64 ranges."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from pyhgf.math import lambert_w0

try:
    from jax import enable_x64
except ImportError:  # JAX versions before enable_x64 became a top-level export.
    from jax.experimental import enable_x64


def _arguments(x64):
    maximum_power = 300 if x64 else 30
    values = np.concatenate(([0.0, 1.0, np.e], np.logspace(-12, maximum_power, 65)))
    return jnp.asarray(values, dtype=jnp.float64 if x64 else jnp.float32)


@pytest.mark.parametrize("x64", [False, True])
def test_lambert_w_matches_scipy_across_finite_range(x64):
    """Large finite inputs remain finite and match the principal-branch oracle."""
    with enable_x64(x64):
        z = _arguments(x64)
        actual = np.asarray(jax.jit(lambert_w0)(z))
        expected = special.lambertw(np.asarray(z).astype(np.float64)).real
        assert np.all(np.isfinite(actual))
        assert actual[0] == 0.0
        assert actual.dtype == (np.dtype("float64") if x64 else np.dtype("float32"))
        np.testing.assert_allclose(
            actual, expected, rtol=4e-6 if not x64 else 4e-13, atol=0.0
        )


@pytest.mark.parametrize("x64", [False, True])
def test_lambert_w_has_small_log_equation_residual(x64):
    """The defining equation holds without multiplying enormous exponentials."""
    with enable_x64(x64):
        z = _arguments(x64)[1:]
        w = np.asarray(lambert_w0(z)).astype(np.float64)
        assert np.all(w > 0.0)
        residual = np.log(w) + w - np.log(np.asarray(z).astype(np.float64))
        np.testing.assert_allclose(
            residual, 0.0, rtol=0.0, atol=2e-5 if not x64 else 2e-11
        )


@pytest.mark.parametrize("x64", [False, True])
def test_lambert_w_custom_jvp_is_finite_and_correct_at_zero(x64):
    """The implicit derivative survives the stable forward iteration, including zero."""
    with enable_x64(x64):
        z = _arguments(x64)
        _, derivative = jax.jvp(lambert_w0, (z,), (jnp.ones_like(z),))
        actual = np.asarray(derivative)
        reference_w = special.lambertw(np.asarray(z).astype(np.float64)).real
        expected = np.exp(-reference_w) / (1.0 + reference_w)
        assert np.all(np.isfinite(actual))
        assert actual[0] == 1.0
        np.testing.assert_allclose(
            actual, expected, rtol=7e-6 if not x64 else 7e-13, atol=0.0
        )
        zero = jnp.asarray(0.0, dtype=jnp.float64 if x64 else jnp.float32)
        assert float(jax.grad(lambert_w0)(zero)) == 1.0
