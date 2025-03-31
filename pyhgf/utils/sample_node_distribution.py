# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Sylvain Estebe

import jax.numpy as jnp
from jax import lax, random

from pyhgf.typing import Attributes


def sample_node_distribution(
    attributes: Attributes,
    node_idx: int,
    rng_key: random.PRNGKey,
    model_type: int,
) -> tuple[float, random.PRNGKey]:
    """Sample a value from the distribution of the specified node.

    Parameters
    ----------
    attributes : Dict[int, dict]
        The dictionary of node parameters, keyed by node index.
    node_idx : int
        The index of the child nodes whose distribution is to be sampled.
    rng_key : random.PRNGKey
        A PRNG key for random number generation.
    model_type : int
        Specifies the distribution type (e.g., discrete=1, continuous=2).

    Returns
    -------
    sample : float
        The sampled value from the node's distribution.
    rng_key : random.PRNGKey
        Updated PRNG key.

    """
    # Fetch parameters from attributes
    mu = attributes[node_idx]["expected_mean"]
    precision = attributes[node_idx]["expected_precision"]
    sigma = 1.0 / jnp.sqrt(precision)

    # Split RNG key
    rng_key, subkey = random.split(rng_key)

    # Sample conditionally based on model type
    sample = lax.cond(
        model_type == 2,
        lambda _: random.normal(subkey) * sigma + mu,  # Continuous distribution
        lambda _: jnp.float32(random.bernoulli(subkey)),  # Discrete (Bernoulli)
        operand=None,
    )
    return sample, rng_key
