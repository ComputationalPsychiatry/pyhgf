# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.prediction_error.volatile import volatile_node_prediction_error

from .posterior_update_value_level import (
    posterior_update_mean_value_level,
    posterior_update_precision_value_level,
)
from .posterior_update_volatility_level import (
    posterior_update_mean_volatility_level,
    posterior_update_precision_volatility_level,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_posterior_update_ehgf(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> dict:
    """Update a volatile node using the eHGF update for the volatility level.

    The eHGF update differs from the standard update in the order of updates for
    the implicit volatility level: it updates the **mean first** using the expected
    precision as an approximation, and then updates the precision. This often reduces
    errors associated with impossible parameter spaces and improves sampling.

    1. Update value level using children's value prediction errors (standard order)
    2. Recompute volatility prediction error using updated value level
    3. Update volatility level mean first (using expected precision), then precision

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`.
    node_idx :
        Pointer to the volatile node that needs to be updated.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    volatile_node_posterior_update, volatile_node_posterior_update_unbounded

    """
    # 1. UPDATE VALUE LEVEL (external facing) - standard order
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

    # 2. COMPUTE PREDICTION ERROR
    # Now that value level has been updated, compute the value and volatility PE
    attributes = volatile_node_prediction_error(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    # 3. UPDATE VOLATILITY LEVEL (eHGF: mean first, then precision)
    # Update mean first using expected precision as approximation
    mean_vol = posterior_update_mean_volatility_level(
        attributes, node_idx, attributes[node_idx]["expected_precision_vol"]
    )
    attributes[node_idx]["mean_vol"] = mean_vol

    # Then update precision
    precision_vol = posterior_update_precision_volatility_level(attributes, node_idx)
    attributes[node_idx]["precision_vol"] = precision_vol

    return attributes
