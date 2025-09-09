# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

from jax import jit
import jax
from pyhgf.typing import Attributes, Edges
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.utils import set_coupling


@partial(jit, static_argnames=("node_idx", "edges"))
def learning_weights(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
) -> Attributes:
    r"""Update the coupling strengths with child nodes from their prediction errors.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network.
    node_idx :
        Pointer to the input node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The attributes of the probabilistic network.

    """
    # 1. update the weights of the connections to the value children
    # --------------------------------------------------------------
    for value_parent_idx, value_coupling in zip(
        edges[node_idx].value_parents,  # type: ignore
        attributes[node_idx]["value_coupling_parents"],
    ):
        # find the coupling function for this node
        coupling_fn = edges[value_parent_idx].coupling_fn[
            edges[value_parent_idx].value_children.index(node_idx)
        ]

        expected_coupling = attributes[node_idx]["mean"] / coupling_fn(
            attributes[value_parent_idx]["expected_mean"]
        )
        # jax.debug.print("🤯 expected_coupling : {x} 🤯", x=expected_coupling)

        precision_weighting = attributes[value_parent_idx]["precision"] / (
            attributes[node_idx]["precision"]
            + attributes[value_parent_idx]["precision"]
        )
        # jax.debug.print("🤯 precision_weighting : {x} 🤯", x=precision_weighting)

        new_value_coupling = (
            value_coupling + (expected_coupling - value_coupling) * precision_weighting
        )
        # jax.debug.print("🤯 new_value_coupling : {x} 🤯", x=new_value_coupling)

        # update the coupling strength in the attributes dictionary for both nodes
        set_coupling(
            parent_idx=value_parent_idx,
            child_idx=node_idx,
            coupling=new_value_coupling,
            edges=edges,
            attributes=attributes,
        )

    # 2. call a new prediction step to update the node's mean and variance
    # --------------------------------------------------------------------
    # attributes = continuous_node_prediction(attributes, node_idx, edges)

    return attributes
