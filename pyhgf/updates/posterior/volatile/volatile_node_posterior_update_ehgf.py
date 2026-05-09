# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges

from .posterior_update_volatility_level import (
    posterior_update_mean_volatility_level,
    posterior_update_precision_volatility_level,
)


@partial(jit, static_argnames=("edges", "node_idx", "max_posterior_precision"))
def volatile_node_posterior_update_ehgf(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    max_posterior_precision: float = 1e10,
) -> dict:
    """Update a volatile node using the eHGF update for the volatility level.

    The eHGF update differs from the standard update in the order of updates for
    the implicit volatility level: it updates the **mean first** using the expected
    precision as an approximation, and then updates the precision. This often reduces
    errors associated with impossible parameter spaces and improves sampling.

    Unlike the standard continuous-state posterior updates elsewhere in the toolbox, the
    volatile-state updates use the *expected* mean (i.e. the prediction) as the
    reference point rather than the posterior mean. This choice is made to better suit
    deep learning networks where the prediction serves as the natural reference for
    computing updates.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`.
    node_idx :
        Pointer to the volatile node that needs to be updated.
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision write.
        Default ``1e10``.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    volatile_node_posterior_update, volatile_node_posterior_update_unbounded

    """
    # UPDATE VOLATILITY LEVEL (eHGF: mean first, then precision) -----------------------
    # ----------------------------------------------------------------------------------

    # Update mean first using expected precision as approximation
    mean_vol = posterior_update_mean_volatility_level(
        attributes, node_idx, attributes[node_idx]["expected_precision_vol"]
    )
    attributes[node_idx]["mean_vol"] = mean_vol

    # Then update precision
    precision_vol = posterior_update_precision_volatility_level(attributes, node_idx)
    attributes[node_idx]["precision_vol"] = jnp.minimum(
        precision_vol, max_posterior_precision
    )

    return attributes
