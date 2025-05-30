# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Sylvain Estebe


from functools import partial
from typing import Callable, Dict, Optional, Tuple

from jax import jit
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges, UpdateSequence
from pyhgf.updates.observation import set_observation
from pyhgf.utils.sample_node_distribution import sample_node_distribution


@partial(
    jit,
    static_argnames=(
        "update_sequence",
        "edges",
        "input_idxs",
        "observations",
        "action_fn",
    ),
)
def beliefs_propagation(
    attributes: Attributes,
    inputs: Tuple[ArrayLike, ...],
    update_sequence: UpdateSequence,
    edges: Edges,
    input_idxs: Tuple[int],
    observations: str = "external",
    action_fn: Optional[Callable[[Attributes, tuple], tuple[Attributes, tuple]]] = None,
) -> Tuple[Dict, Dict]:
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
    input_idxs :
        List input indexes.
    observations :
        A string indicating how the network receive new observations. Can be
        `"external"` (default) when new observation are provided, `"generative"` - in
        which case the network sample observation from its own predictive distribution,
        or `"deprived"` so no observation are provided.
    action_fn :
        Optional. When provided, can implement action, decisions or transformation in
        the environment. The function should receive and return the attributes of the
        network and the inputs. This function is called after prediction and before
        observation.

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).

    """
    prediction_steps, update_steps = update_sequence

    # when observations is "generative", only time_step is provided
    # and the other variables are None
    values_tuple, observed_tuple, time_step, rng_key = inputs

    # Assign the time_step (or input data) to the attributes.
    attributes[-1]["time_step"] = time_step

    # 1. Prediction sequence -----------------------------------------------------------
    # ----------------------------------------------------------------------------------
    for node_idx, update_fn in prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # act on the environment before observation
    if action_fn:
        # Call the action function if provided -
        # this function can transform attributes and inputs before the observation step
        attributes, inputs = action_fn(attributes, inputs)
        values_tuple, observed_tuple, time_step, rng_key = (
            inputs  # overwrite the inputs
        )

    # 2. Receive new observations ------------------------------------------------------
    # ----------------------------------------------------------------------------------
    if observations == "generative":
        # Inline handling of observation for each input node
        for node_idx in input_idxs:
            # Sample the node distribution
            sampled_value = sample_node_distribution(
                attributes=attributes,
                edges=edges,
                node_idx=node_idx,
                rng_key=rng_key,
            )
            # Set the observation (using a constant observation flag, here set as 1)
            attributes = set_observation(
                attributes=attributes,
                node_idx=node_idx,
                values=sampled_value,
                observed=1,
            )
    elif observations == "external":
        # Unpack observation data and update each input node.
        for values, observed, node_idx in zip(values_tuple, observed_tuple, input_idxs):
            attributes = set_observation(
                attributes=attributes,
                node_idx=node_idx,
                values=values.squeeze(),
                observed=observed,
            )
    elif observations == "deprived":
        pass

    else:
        # if observation is not parametrised correctly
        # return an empty dictionary and crash the scan iteration
        attributes = {}

    # 3. Update sequence (common for both modes) ---------------------------------------
    # ----------------------------------------------------------------------------------
    for node_idx, update_fn in update_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    return (
        attributes,
        attributes,
    )  # ("carryover", "accumulated")
