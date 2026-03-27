# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial

from jax import jit

from pyhgf.typing import Edges

from .posterior_update_value_level import (
    posterior_update_mean_value_level,
    posterior_update_precision_value_level,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_posterior_update(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> dict:
    """Update a volatile node and the implied volatility parent.

    Unlike the standard continuous-state posterior updates elsewhere in the toolbox,
    the volatile-state updates use the *expected* mean (i.e. the prediction) as the
    reference point rather than the posterior mean. This choice is made to better suit
    deep learning networks where the prediction serves as the natural reference for
    computing updates.

    """
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

    return attributes
