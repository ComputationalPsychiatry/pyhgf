# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=("node_idx",))
def posterior_update_precision_volatility_level(
    attributes: dict,
    node_idx: int,
) -> float:
    """Update the precision of the volatility level.

    Uses the value level's volatility prediction error (internal coupling).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile-state node that will be updated.

    Returns
    -------
    posterior_precision :
        The new posterior precision of the volatility level.
    """
    # Start with expected precision
    posterior_precision = attributes[node_idx]["expected_precision_vol"]

    # Get internal volatility PE from value level
    volatility_pe = attributes[node_idx]["temp"]["volatility_prediction_error"]
    # Use the VALUE level's effective precision (not the volatility level's)
    effective_precision_value = attributes[node_idx]["temp"]["effective_precision"]

    # Update precision using the volatility coupling formula (coupling fixed at 1).
    posterior_precision += (
        0.5 * (effective_precision_value**2)
        + (effective_precision_value**2) * volatility_pe
        - 0.5 * effective_precision_value * volatility_pe
    )

    return posterior_precision


@partial(jit, static_argnames=("node_idx",))
def posterior_update_precision_volatility_level_ehgf(
    attributes: dict,
    node_idx: int,
) -> float:
    """Safe enhanced-HGF precision update for a fused volatile node.

    Mirror of :func:`_ehgf_volatility_precision_increment` for the implicit
    value/volatility coupling of a volatile-state node: the effective precision is
    recomputed from the volatility level's just-updated posterior mean (``mean_vol``)
    and the increment is floored at zero. Keeps the fused volatile-state node identical
    to the explicit continuous + volatility-parent construction under the eHGF.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile-state node that will be updated.

    Returns
    -------
    posterior_precision :
        The new posterior precision of the volatility level.
    """
    time_step = attributes[-1]["time_step"]
    mean_vol = attributes[node_idx]["mean_vol"]  # volatility-level posterior mean
    # value-level posterior variance at the previous step (σ = 1 / π)
    previous_variance = attributes[node_idx]["temp"]["current_variance"]

    # Volatility coupling is fixed at 1.
    predicted_volatility = time_step * jnp.exp(mean_vol)
    expected_precision = 1.0 / (previous_variance + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision
    volatility_error_weight = (
        predicted_volatility - previous_variance
    ) * expected_precision
    volatility_prediction_error = (
        1.0 / attributes[node_idx]["precision"]
        + (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2
    ) * expected_precision - 1.0

    increment = jnp.maximum(
        0.0,
        0.5
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error),
    )
    return attributes[node_idx]["expected_precision_vol"] + increment


@partial(jit, static_argnames=("node_idx",))
def posterior_update_mean_volatility_level(
    attributes: dict,
    node_idx: int,
    node_precision: float,
) -> float:
    """Update the mean of the volatility level.

    Uses the value level's volatility prediction error (internal coupling).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile-state node that will be updated.
    node_precision :
        The precision of the volatility level, used to weight the prediction error.

    Returns
    -------
    posterior_mean :
        The new posterior mean of the volatility level.
    """
    # Start with expected mean
    posterior_mean = attributes[node_idx]["expected_mean_vol"]

    # Get internal volatility PE from value level
    volatility_pe = attributes[node_idx]["temp"]["volatility_prediction_error"]
    # Use the VALUE level's effective precision (not the volatility level's)
    effective_precision_value = attributes[node_idx]["temp"]["effective_precision"]

    # Update mean using the volatility coupling formula (coupling fixed at 1).
    precision_weighted_pe = (effective_precision_value * volatility_pe) / (
        2 * node_precision
    )

    posterior_mean += precision_weighted_pe

    return posterior_mean
