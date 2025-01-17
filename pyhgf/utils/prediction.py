from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, random
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges, UpdateSequence
from pyhgf.updates.observation import set_observation


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
        rng_key = carry
        sample, new_rng_key = sample_node_distribution(
            attributes, node_idx, rng_key, model_type
        )
        return new_rng_key, sample

    rng_key, samples = jax.lax.scan(scan_fn, rng_key, None, length=num_samples)
    return samples


def sample_node_distribution(
    attributes: Dict[int, dict],
    node_idx: int,
    rng_key: random.PRNGKey,
    model_type: int,
) -> Tuple[float, random.PRNGKey]:
    """Sample a value from the distribution of the specified node.

    Parameters
    ----------
    attributes : Dict[int, dict]
        The dictionary of node parameters, keyed by node index.
    node_idx : int
        The index of the node whose distribution is to be sampled.
    rng_key : random.PRNGKey
        A PRNG key for random number generation.
    model_type : int
        Specifies the distribution type (e.g., discrete: 1, continuous: 2).

    Returns
    -------
    sample : float
        The sampled value from the node's distribution.
    rng_key : random.PRNGKey
        Updated PRNG key.

    """
    node_attr = attributes.get(node_idx, {})
    mu = node_attr.get("expected_mean", 0.0)
    precision = node_attr.get("expected_precision", 1.0)

    # Ensure precision is positive
    precision = jnp.where(precision > 0, precision, 1e-12)
    sigma = 1.0 / jnp.sqrt(precision)

    rng_key, subkey = random.split(rng_key)
    p = jnp.clip(mu, 0.0, 1.0)

    # Sample based on model type
    sample = jnp.where(
        model_type == 2,  # continuous
        random.normal(subkey) * sigma + mu,
        random.bernoulli(subkey, p),  # discrete
    )

    return sample, rng_key


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
        The dictionaries of nodes' parameters.
    node_idx : int
        Index of the node to handle.
    rng_key : random.PRNGKey
        Random number generator key.
    sophisticated : bool
        Determines whether to use sophisticated sampling or default.
    edges : Edges
        Information on the network's edges.

    Returns
    -------
    updated_attributes : Attributes
        Updated attributes after observation.
    rng_key : random.PRNGKey
        Updated RNG key.

    """

    def sophisticated_branch(_):
        sampled_value, new_rng_key = sample_node_distribution(
            attributes, node_idx, rng_key, edges[node_idx].node_type
        )
        return sampled_value, 1, new_rng_key

    def non_sophisticated_branch(_):
        return jnp.nan, 0, rng_key

    # Use lax.cond to branch on sophisticated or not
    sampled_value, observed, rng_key = lax.cond(
        sophisticated,
        sophisticated_branch,
        non_sophisticated_branch,
        operand=None,
    )

    # Assign observation to the node
    updated_attributes = set_observation(
        attributes=attributes,
        node_idx=node_idx,
        values=sampled_value,
        observed=observed,
    )
    return updated_attributes, rng_key


@partial(jit, static_argnames=("update_sequence", "edges", "input_idxs"))
def inference_prediction(
    attributes: Attributes,
    inputs: Tuple[ArrayLike, ...],
    update_sequence: UpdateSequence,
    edges: Edges,
    input_idxs: Tuple[int],
    sophisticated: bool = True,
    rng_seed: int = 42,
) -> Tuple[Dict, Dict]:
    """Perform inference and prediction steps in a network.

    This function updates the network's parameters based on new observations and a
    specified update sequence. It integrates prediction, observation handling,
    and update steps in three main stages:

    Parameters
    ----------
    attributes : Attributes
        The dictionaries of nodes' parameters. This variable is updated and returned
        after the inference and prediction steps.
    inputs : tuple of ArrayLike
        A tuple of arrays containing the new observation(s), the time steps, and
        additional input data.
    update_sequence : UpdateSequence
        The sequence of updates that will be applied to the node structure.
        Typically, this is a tuple of two lists:
          ([(node_idx, update_fn), ...],  # prediction steps
           [(node_idx, update_fn), ...])  # update steps
    edges : Edges
        Information on the network's edges, which may be used by update functions.
    input_idxs : tuple of int
        List of input node indexes that will receive new observations.
    sophisticated : bool, optional
        Determines whether to use sophisticated sampling during observations
        (default: True).
    rng_seed : int, optional
        Seed for the random number generator (default: 42).

    Returns
    -------
    attributes, attributes : tuple of Dict
        A tuple of parameter structures after the prediction and inference cycles
        have completed. Both entries are the same here (carryover vs. accumulated
        structures) but can be differentiated if needed in future extensions.

    """
    rng_key = random.PRNGKey(rng_seed)
    prediction_steps, update_steps = update_sequence

    time_step = inputs
    attributes[-1]["time_step"] = time_step

    # Prediction Sequence
    for node_idx, update_fn in prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # Observations
    for node_idx in input_idxs:
        attributes, rng_key = handle_observation(
            attributes, node_idx, rng_key, sophisticated, edges
        )

    # Update Sequence
    for node_idx, update_fn in update_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # Return updated attributes
    return attributes, attributes
