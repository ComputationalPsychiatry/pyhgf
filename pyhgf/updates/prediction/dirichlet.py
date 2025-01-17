# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict

import jax.numpy as jnp

from pyhgf.math import Normal
from pyhgf.typing import Attributes, Edges


def dirichlet_node_prediction(
    edges: Edges,
    attributes: Dict,
    node_idx: int,
    **args,
) -> Attributes:
    """Prediction of a Dirichlet process node.

    Parameters
    ----------
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the Dirichlet process input node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the neural network.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.
    dirichlet_node :
        Static parameters of the Dirichlet process node.

    """
    # get the parameter (mean and variance) from the exponential state parent nodes
    value_parent_idxs = edges[node_idx].value_parents
    if value_parent_idxs is not None:
        parameters = jnp.array(
            [
                Normal().parameters_from_sufficient_statistics(
                    xis=attributes[parent_idx]["xis"]
                )
                for parent_idx in value_parent_idxs
            ]
        )

        attributes[node_idx]["expected_means"] = parameters[:, 0]
        attributes[node_idx]["expected_sigmas"] = jnp.sqrt(parameters[:, 1])

    return attributes
