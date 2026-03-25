# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Sylvain Estebe

import jax.numpy as jnp
from jax import random

from pyhgf.typing import Attributes, Edges


def sample_node_distribution(
    attributes: Attributes,
    edges: Edges,
    node_idx: int,
    rng_key: random.PRNGKey,
) -> tuple[float, random.PRNGKey]:
    """Sample a value from the distribution of an input node.

    Parameters
    ----------
    attributes :
        The dictionary of node parameters, keyed by node index.
    edges :
        Information on the network's edges.
    node_idx :
        The index of the child nodes whose distribution is to be sampled.
    rng_key :
        A PRNG key for random number generation.

    Returns
    -------
    sample :
        The sampled value from the node's distribution.

    """
    if edges[node_idx].node_type == 1:
        mu = attributes[node_idx]["expected_mean"]
        sample = jnp.float32(random.bernoulli(rng_key, p=mu))
    elif edges[node_idx].node_type == 2:
        mu = attributes[node_idx]["expected_mean"]
        precision = attributes[node_idx]["expected_precision"]
        sample = random.normal(rng_key) * (1.0 / jnp.sqrt(precision)) + mu

    return sample
