# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

from jax import jit, random

from pyhgf.updates.observation import set_observation
from pyhgf.utils.handle_observation import handle_observation


@partial(
    jit, static_argnames=("update_sequence", "edges", "input_idxs", "sophisticated")
)
def beliefs_propagation(
    attributes: dict,
    inputs: tuple,
    update_sequence,
    edges,
    input_idxs: tuple,
    sophisticated: bool,
    rng_key: random.PRNGKey = None,
) -> tuple:
    """Update the networks parameters after observing new data point(s).

    This function performs the beliefs propagation step. Belief propagation consists in:
    1. A prediction sequence, from the leaves of the graph to the roots.
    2. The assignation of new observations to target nodes (usually the roots of the
    network)
    3. An inference step alternating between prediction errors and posterior updates,
    starting from the roots of the network to the leaves.
    This function returns a tuple of two new `parameter_structure` (i.e. the carryover
    and the accumulated in the context of :py:func:`jax.lax.scan`).

    Parameters
    ----------
    attributes :
        The dictionaries of nodes' parameters. This variable is updated and returned
        after the beliefs propagation step.
    inputs :
        A tuple of n by time steps arrays containing the new observation(s), the time
        steps as well as a boolean mask for observed values. The new observations are a
        tuple of array, with length equal to the number of input nodes. Each input node
        can receive observations  The time steps are the last
        column of the array, the default is unit incrementation.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    edges :
        Information on the network's edges.
    sophisticated :
        A boolean indicating whether the network use sophisticated inference or not.
    rng_key :
        Random key for the random number generator.
    input_idxs :
        List input indexes.

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).

    """
    if sophisticated:

        # Use the passed rng_key without reinitializing it.
        prediction_steps, update_steps = update_sequence

        # Assign time_step (e.g. input data) to a special key in attributes.
        attributes[-1]["time_step"] = inputs

        # Run prediction steps.
        for node_idx, update_fn in prediction_steps:
            attributes = update_fn(
                attributes=attributes, node_idx=node_idx, edges=edges
            )

        # Handle observations on specified input nodes.
        for node_idx in input_idxs:
            attributes, rng_key = handle_observation(
                attributes, node_idx, rng_key, sophisticated, edges
            )

        # Run update steps.
        for node_idx, update_fn in update_steps:
            attributes = update_fn(
                attributes=attributes, node_idx=node_idx, edges=edges
            )

        return attributes, attributes, rng_key

    else:

        prediction_steps, update_steps = update_sequence

        # unpack input data - ((ndarrays, ...), (1darrays, ...), float64)
        values_tuple, observed_tuple, time_step = inputs

        attributes[-1]["time_step"] = time_step

        # 1 - Séquence de prédiction
        for step in prediction_steps:
            node_idx, update_fn = step
            attributes = update_fn(
                attributes=attributes,
                node_idx=node_idx,
                edges=edges,
            )

        for values, observed, node_idx in zip(values_tuple, observed_tuple, input_idxs):

            attributes = set_observation(
                attributes=attributes,
                node_idx=node_idx,
                values=values.squeeze(),
                observed=observed,
            )

        # 3 - Update sequence
        # -------------------
        for step in update_steps:

            node_idx, update_fn = step

            attributes = update_fn(
                attributes=attributes,
                node_idx=node_idx,
                edges=edges,
            )
        return (
            attributes,
            attributes,
        )  # ("carryover", "accumulated")
