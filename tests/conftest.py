"""Shared test configuration."""

import jax
import jax.numpy as jnp
import pytest

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64

# jax >= 0.4.36 defaults to partitionable threefry key derivation, which changes
# every random stream. The suite's hardcoded expectations and seed-calibrated
# tolerances were sampled from the legacy streams, so pin those explicitly.
jax.config.update("jax_threefry_partitionable", False)


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["float32", "float64"])
def float_dtype(request):
    """Run a test once in single and once in double precision; yield the dtype."""
    with enable_x64(request.param == jnp.float64):
        yield request.param
