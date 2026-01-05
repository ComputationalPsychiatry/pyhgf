# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial

from jax import jit

from pyhgf.typing import Edges

from .posterior_update_value_level import (
    posterior_update_mean_value_level,
    posterior_update_precision_value_level,
)
from .posterior_update_volatility_level import (
    posterior_update_mean_volatility_level,
    posterior_update_precision_volatility_level,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_posterior_update(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> dict:
    """Update both internal levels of the volatile_node.

    1. Update value level using children's value prediction errors
    2. Recompute volatility prediction error using updated value level
    3. Update volatility level using the fresh volatility prediction error

    Note: The volatility PE must be recomputed here (not reused from the
    prediction_error step) to match standard HGF behavior where volatility
    parents update in the same timestep using fresh volatility PEs.
    """
    # 1. UPDATE VALUE LEVEL (external facing)
    # Update precision first
    precision_value = posterior_update_precision_value_level(
        attributes, edges, node_idx
    )
    attributes[node_idx]["precision"] = precision_value

    # Update mean using new precision
    mean_value = posterior_update_mean_value_level(
        attributes, edges, node_idx, precision_value
    )
    attributes[node_idx]["mean"] = mean_value

    # 2. RECOMPUTE VOLATILITY PREDICTION ERROR
    # Now that value level has been updated, recompute the volatility PE
    # This is the value prediction error for the value level (using updated mean)
    value_prediction_error = mean_value - attributes[node_idx]["expected_mean"]

    # Volatility PE from value level (same formula as in prediction_error step)
    volatility_prediction_error = (
        (attributes[node_idx]["expected_precision"] / precision_value)
        + attributes[node_idx]["expected_precision"] * (value_prediction_error**2)
        - 1
    )

    # Store the fresh volatility PE
    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        volatility_prediction_error
    )

    # 3. UPDATE VOLATILITY LEVEL (implicit internal)
    # Update precision first
    precision_vol = posterior_update_precision_volatility_level(attributes, node_idx)
    attributes[node_idx]["precision_vol"] = precision_vol

    # Update mean using new precision
    mean_vol = posterior_update_mean_volatility_level(
        attributes, node_idx, precision_vol
    )
    attributes[node_idx]["mean_vol"] = mean_vol

    return attributes
