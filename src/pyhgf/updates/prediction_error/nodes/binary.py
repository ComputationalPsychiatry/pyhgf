# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import Array, jit


@partial(jit, static_argnames=("node_idx"))
def binary_state_node_prediction_error(
    attributes: Dict,
    node_idx: int,
) -> Array:
    """Compute the value prediction-errors and predicted precision of a binary node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    node_idx :
        Pointer to the binary state node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    # compute the prediction error of the binary state node
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # scale the prediction error so it can be used in the posterior update
    # (eq. 98, Weber et al., v1)
    value_prediction_error /= attributes[node_idx]["expected_precision"]

    # store the prediction errors in the binary node
    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error

    return attributes
