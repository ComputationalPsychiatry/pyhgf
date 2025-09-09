# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

from jax import jit
from jax.typing import ArrayLike
from pyhgf.typing import Attributes, Edges, LearningSequence
from pyhgf.updates.observation import set_predictors, set_observation
import jax


@partial(
    jit,
    static_argnames=(
        "learning_sequence",
        "edges",
        "inputs_x_idxs",
        "inputs_y_idxs",
    ),
)
def learning(
    attributes: Attributes,
    inputs: tuple[ArrayLike, ...],
    inputs_x_idxs: tuple[int],
    inputs_y_idxs: tuple[int],
    learning_sequence: LearningSequence,
    edges: Edges,
) -> tuple[dict, dict]:
    """Update the networks coupling parameters using prospective configuration.

    This algorithm implements a learning step, using a predictive coding scheme,
    inspired by the prospective configuration scheme proposed in [1]_.

    Parameters
    ----------
    attributes :
        The dictionaries of nodes' parameters. This variable is updated and returned
        after the beliefs propagation step.
    inputs :
        A tuple of n arrays containing the new predictors (x) and the expected
        predictions (y). Predictors values are set to the obervation nodes defined by
        `inputs_x_idxs` before the prediction steps. Predictions are observed in the
        observation steps in the nodes defined by `inputs_y_idxs`.
    inputs_x_idxs :
        The indexes of the nodes receiving the predictors (x).
    inputs_y_idxs :
        The indexes of the nodes receiving the predictions (y).
    learning_sequence :
        The sequence that will be applied to the node structure. It is expected that
        the sequence contains a prediction, an update and a learning set of updates.
    edges :
        Information on the network's edges.

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).

    References
    ----------
    .. [1] Song, Y., Millidge, B., Salvatori, T., Lukasiewicz, T., Xu, Z., & Bogacz, R.
        (2024). Inferring neural activity before plasticity as a foundation for
        learning beyond backpropagation. Nature Neuroscience, 27(2), 348–358.
        doi:10.1038/s41593-023-01514-1

    """
    # Unpack the inputs tuple.
    x, y = inputs

    # Assign the time_step (or input data) to the attributes.
    time_step = 1.0
    attributes[-1]["time_step"] = time_step

    # 1. Set predictor states ----------------------------------------------------------
    # ----------------------------------------------------------------------------------
    for values, node_idx in zip(x, inputs_x_idxs):
        attributes = set_predictors(
            attributes=attributes,
            node_idx=node_idx,
            values=values.squeeze(),
        )
    jax.debug.print("🤯 {x} 🤯", x=attributes[3]["expected_mean"])

    # 2. Prediction sequence -----------------------------------------------------------
    # ----------------------------------------------------------------------------------
    for node_idx, update_fn in learning_sequence.prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)
    jax.debug.print("🤯 {x} 🤯", x=attributes[3]["expected_mean"])

    # 3. Receive new observations ------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # Unpack observation data and update each input node.
    for values, node_idx in zip(y, inputs_y_idxs):
        attributes = set_observation(
            attributes=attributes,
            node_idx=node_idx,
            values=values.squeeze(),
            observed=1,
        )
    jax.debug.print("🤯 {x} 🤯", x=attributes[3]["expected_mean"])

    # 4. Update sequence ---------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    for node_idx, update_fn in learning_sequence.update_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)
    jax.debug.print("🤯 {x} 🤯", x=attributes[3]["expected_mean"])

    # 5. Learning sequence -------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    for node_idx, update_fn in learning_sequence.learning_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)
    jax.debug.print("🤯 {x} 🤯", x=attributes[3]["expected_mean"])

    return (
        attributes,
        attributes,
    )  # ("carryover", "accumulated")
