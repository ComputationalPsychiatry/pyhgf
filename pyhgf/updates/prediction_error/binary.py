# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx"))
def binary_state_node_prediction_error(
    attributes: Dict, node_idx: int, **args
) -> Array:
    """Compute the value prediction errors and predicted precision of a binary node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
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

    # cancel the prediction error if the value was not observed
    value_prediction_error *= attributes[node_idx]["observed"]

    # scale the prediction error so it can be used in the posterior update
    # (eq. 98, Weber et al., v1)
    value_prediction_error /= attributes[node_idx]["expected_precision"]

    # store the prediction errors in the binary node
    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error

    # here we also update the precision
    attributes[node_idx]["precision"] = attributes[node_idx]["expected_precision"]

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_finite_state_node_prediction_error(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the posterior of a binary node given finite precision of the input.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    value_child_idx = edges[node_idx].value_children[0]  # type: ignore

    delata0 = attributes[value_child_idx]["temp"]["value_prediction_error_0"]
    delata1 = attributes[value_child_idx]["temp"]["value_prediction_error_1"]
    expected_precision = attributes[value_child_idx]["expected_precision"]

    # Likelihood under eta1
    und1 = jnp.exp(-expected_precision / 2 * delata1**2)

    # Likelihood under eta0
    und0 = jnp.exp(-expected_precision / 2 * delata0**2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    expected_mean = attributes[node_idx]["expected_mean"]
    mean = expected_mean * und1 / (expected_mean * und1 + (1 - expected_mean) * und0)
    precision = 1 / (expected_mean * (1 - expected_mean))

    attributes[node_idx]["mean"] = mean
    attributes[node_idx]["precision"] = precision

    return attributes
