# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial
from jax import jit


@partial(jit, static_argnames=("node_idx",))
def posterior_update_precision_volatility_level(
    attributes: dict,
    node_idx: int,
) -> float:
    """Update the precision of the volatility level.

    Uses the value level's volatility prediction error (internal coupling).
    """
    # Start with expected precision
    posterior_precision = attributes[node_idx]["expected_precision_vol"]

    # Get internal volatility PE from value level
    volatility_pe = attributes[node_idx]["temp"]["volatility_prediction_error"]
    # Use the VALUE level's effective precision (not the volatility level's)
    effective_precision_value = attributes[node_idx]["temp"]["effective_precision"]
    volatility_coupling = attributes[node_idx]["volatility_coupling_internal"]

    # Update precision using volatility coupling formula
    posterior_precision += (
        0.5 * ((volatility_coupling * effective_precision_value) ** 2)
        + ((volatility_coupling * effective_precision_value) ** 2) * volatility_pe
        - 0.5 * (volatility_coupling**2) * effective_precision_value * volatility_pe
    )

    return posterior_precision


@partial(jit, static_argnames=("node_idx",))
def posterior_update_mean_volatility_level(
    attributes: dict,
    node_idx: int,
    node_precision: float,
) -> float:
    """Update the mean of the volatility level.

    Uses the value level's volatility prediction error (internal coupling).
    """
    # Start with expected mean
    posterior_mean = attributes[node_idx]["expected_mean_vol"]

    # Get internal volatility PE from value level
    volatility_pe = attributes[node_idx]["temp"]["volatility_prediction_error"]
    # Use the VALUE level's effective precision (not the volatility level's)
    effective_precision_value = attributes[node_idx]["temp"]["effective_precision"]
    volatility_coupling = attributes[node_idx]["volatility_coupling_internal"]

    # Check if value level was observed
    observed = attributes[node_idx]["observed"]

    # Update mean using volatility coupling formula
    precision_weighted_pe = (
        volatility_coupling * effective_precision_value * volatility_pe
    ) / (2 * node_precision)

    # Cancel if not observed
    precision_weighted_pe *= observed

    posterior_mean += precision_weighted_pe

    return posterior_mean
