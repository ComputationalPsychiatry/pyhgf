"""Shared test configuration."""

import jax

# jax >= 0.4.36 defaults to partitionable threefry key derivation, which changes
# every random stream. The suite's hardcoded expectations and seed-calibrated
# tolerances were sampled from the legacy streams, so pin those explicitly.
jax.config.update("jax_threefry_partitionable", False)
