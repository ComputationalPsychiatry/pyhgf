from typing import Tuple

import jax.numpy as jnp
from jax import random

from pyhgf.typing import Attributes, Edges
from pyhgf.updates.observation import set_observation
from pyhgf.utils.sample_node_distribution import sample_node_distribution


def handle_observation(
    attributes: Attributes,
    node_idx: int,
    rng_key: random.PRNGKey,
    sophisticated: bool,
    edges: Edges,
) -> Tuple[Attributes, random.PRNGKey]:
    """Handle the observation for a specific node based on the sophistication flag.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of node parameters.
    node_idx : int
        The index of the node being observed.
    rng_key : random.PRNGKey
        Random number generator key.
    sophisticated : bool
        Whether to use sophisticated sampling.
    edges : Edges
        Network edges containing node type and connections.

    Returns
    -------
    updated_attributes : Attributes
        Updated attributes dictionary.
    rng_key : random.PRNGKey
        Updated RNG key.

    """

    def sophisticated_mode(_):
        sampled_value, rng_key_new = sample_node_distribution(
            attributes, node_idx, rng_key, edges[node_idx].node_type
        )
        return sampled_value, 1, rng_key_new

    def non_sophisticated_mode(_):
        return jnp.nan, 0, rng_key

    if sophisticated:
        sampled_value, observed, rng_key_out = sophisticated_mode(None)
    else:
        sampled_value, observed, rng_key_out = non_sophisticated_mode(None)

    updated_attributes = set_observation(
        attributes=attributes,
        node_idx=node_idx,
        values=sampled_value,
        observed=observed,
    )

    return updated_attributes, rng_key_out
