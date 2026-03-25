# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Sylvain Estebe

from functools import partial
from typing import TYPE_CHECKING

from jax import jit, random, vmap
from jax.lax import scan
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.typing import Attributes

if TYPE_CHECKING:
    from pyhgf.model import Network


def sample(
    network: "Network",
    time_steps: ArrayLike,
    n_predictions: int = 1,
    rng_key: ArrayLike = random.key(0),
) -> Attributes:
    """Sample n_predictions forward in generative mode.

    .. note:
        This function assumes that the network's update_sequence is already set.

    Parameters
    ----------
    network :
        The network instance used for generating predictions.
    time_steps :
        Array of time steps.
    n_predictions :
        Number of predictions to generate. Defaults to 1.
    rng_key :
        Random number generator key, by default PRNGKey(0).

    Returns
    -------
    predictions :
        Dictionary of predictions for each parameter/node.

    """
    # Prepare placeholders for the inputs and observations.
    # In generative mode, these are not used so we set them to None.
    values_tuple = tuple([None] * len(network.input_idxs))
    observed_tuple = tuple([None] * len(network.input_idxs))

    # Use the last attributes as initial state. If not available (no data provided), use
    # the network's nase attributes
    if network.last_attributes is not None:
        initial_state = network.last_attributes
    else:
        initial_state = network.attributes

    sample_fn = Partial(
        single_sample,
        initial_state=initial_state,
        time_steps=time_steps,
        sample_scan_fn=network.sample_scan_fn,
        values_tuple=values_tuple,
        observed_tuple=observed_tuple,
    )

    # Generate a batch of RNG keys, one for each prediction.
    rng_keys_batch = random.split(rng_key, num=n_predictions)

    # Use vmap to vectorize the single_prediction function over the batch of RNG keys.
    # This will return a dictionary of arrays.
    predictions = vmap(sample_fn)(rng_keys_batch)

    return predictions


@partial(jit, static_argnames=("sample_scan_fn", "values_tuple", "observed_tuple"))
def single_sample(
    rng_key,
    initial_state: "Attributes",
    time_steps,
    sample_scan_fn,
    values_tuple,
    observed_tuple,
) -> Attributes:
    """Perform a single prediction using the provided RNG key."""
    # Split the RNG key for each time step.
    rng_keys = random.split(rng_key, num=len(time_steps))
    inputs = (values_tuple, observed_tuple, time_steps, rng_keys)

    # Execute the belief propagation using scan, starting from the initial state
    # This returns the final state (last_attributes) and the node trajectories.
    _, node_trajectories = scan(sample_scan_fn, initial_state, inputs)

    # Return only the node trajectories.
    return node_trajectories
