from jax import lax, random
from jax.typing import ArrayLike

from pyhgf.typing import Attributes
from pyhgf.utils import sample_node_distribution


def scan_sampling(
    attributes: Attributes,
    node_idx: int,
    rng_key: random.PRNGKey,
    model_type: int,
    num_samples: int,
) -> ArrayLike:
    """Generate multiple samples from a node's distribution using JAX's `scan` function.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of nodes' parameters.
    node_idx : int
        The index of the node to sample from.
    rng_key : random.PRNGKey
        A PRNG key used for random number generation.
    model_type : int
        Specifies how to generate samples (e.g., 1 for discrete, 2 for continuous).
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    samples : ArrayLike
        An array of shape (num_samples,) containing the generated samples for the
        specified node's distribution.

    """

    def scan_fn(carry, _):
        rng_key_in = carry
        sample, rng_key_out = sample_node_distribution(
            attributes, node_idx, rng_key_in, model_type
        )
        return rng_key_out, sample

    _, samples = lax.scan(scan_fn, rng_key, None, length=num_samples)
    return samples
