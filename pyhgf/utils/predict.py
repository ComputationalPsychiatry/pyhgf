# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges, Sequence
from pyhgf.updates.observation import set_predictors


@partial(
    jit,
    static_argnames=(
        "prediction_steps",
        "edges",
        "inputs_x_idxs",
        "inputs_y_idxs",
    ),
)
def predict_step(
    attributes: Attributes,
    x_row: ArrayLike,
    prediction_steps: Sequence,
    edges: Edges,
    inputs_x_idxs: tuple[int],
    inputs_y_idxs: tuple[int],
) -> ArrayLike:
    """Run a single forward (prediction-only) pass through the network.

    This is the per-sample function used by :py:meth:`Network.predict` via
    :py:func:`jax.vmap`.  It sets the predictor values, runs the prediction
    sequence top-down, and collects the ``expected_mean`` from the target
    nodes.

    Parameters
    ----------
    attributes :
        Current node attributes (shared across all samples).
    x_row :
        A single row of predictor values with shape ``(n_x_inputs,)``.
    prediction_steps :
        The prediction steps to execute (excluding predictor nodes).
    edges :
        The network's edge structure.
    inputs_x_idxs :
        Node indexes receiving the predictor values.
    inputs_y_idxs :
        Node indexes whose ``expected_mean`` is collected as output.

    Returns
    -------
    predictions :
        A 1-D array of ``expected_mean`` values, one per target node.

    """
    attributes[-1]["time_step"] = 1.0

    # 1. Set predictor values on the top-layer nodes
    for values, node_idx in zip(x_row, inputs_x_idxs):
        attributes = set_predictors(
            attributes=attributes, node_idx=node_idx, values=values.squeeze()
        )

    # 2. Run prediction steps (top-down)
    for node_idx, update_fn in prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # 3. Collect expected_mean from target (bottom) nodes
    predictions = jnp.array([attributes[idx]["expected_mean"] for idx in inputs_y_idxs])

    return predictions
