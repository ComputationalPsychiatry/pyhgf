# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def categorical_state_prediction_error(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Prediction error from a categorical state node.

    The update will pass the input observations to the binary state nodes.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        `"psis"` is the value coupling strength. It should have the same length as the
        volatility parents' indexes. `"volatility_coupling"` is the volatility coupling
        strength. It should have the same length as the volatility parents' indexes.
    node_idx :
        Pointer to the node that needs to be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

    Returns
    -------
    attributes :
        The updated parameters structure.

    See Also
    --------
    binary_input_update, continuous_input_update

    """
    # pass the mean to the binary state nodes
    for mean, value_parent_idx in zip(
        attributes[node_idx]["mean"],
        edges[node_idx].value_parents,  # type: ignore
    ):
        attributes[value_parent_idx]["mean"] = mean
        attributes[value_parent_idx]["observed"] = attributes[node_idx]["observed"]

    return attributes
