from functools import partial
from typing import Dict, Tuple

from jax import jit, random
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges, UpdateSequence
from pyhgf.utils.handle_observation import handle_observation


@partial(
    jit, static_argnames=("update_sequence", "edges", "input_idxs", "sophisticated")
)
def inference_prediction(
    attributes: Attributes,
    inputs: Tuple[ArrayLike, ...],
    update_sequence: UpdateSequence,
    edges: Edges,
    input_idxs: Tuple[int],
    sophisticated: bool,
    rng_key: random.PRNGKey,
) -> Tuple[Dict, Dict, random.PRNGKey]:
    """Perform prediction steps with his own distribution of child node.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of node parameters.
    inputs : Tuple[ArrayLike, ...]
        Input data (e.g., time step or observations).
    update_sequence : UpdateSequence
        Two lists: (prediction_steps, update_steps).
    edges : Edges
        Network edges containing node types and connections.
    input_idxs : Tuple[int]
        Indices of nodes receiving observations at this time step.
    sophisticated : bool
        Whether to sample observations from his own node or not.
    rng_key : random.PRNGKey
        Random number generator key to be used for sampling.

    Returns
    -------
    updated_attributes : Dict
        The attributes after all prediction, observation, and update steps.
    duplicated_attributes : Dict
        A duplicate of updated attributes (as required by your interface).
    rng_key : random.PRNGKey
        Updated RNG key after sampling.

    """
    # Use the passed rng_key without reinitializing it.
    prediction_steps, update_steps = update_sequence

    # Assign time_step (e.g. input data) to a special key in attributes.
    attributes[-1]["time_step"] = inputs

    # Run prediction steps.
    for node_idx, update_fn in prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # Handle observations on specified input nodes.
    for node_idx in input_idxs:
        attributes, rng_key = handle_observation(
            attributes, node_idx, rng_key, sophisticated, edges
        )

    # Run update steps.
    for node_idx, update_fn in update_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    return attributes, attributes, rng_key
