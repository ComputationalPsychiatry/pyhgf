# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges

from .posterior_update_mean_continuous_node import posterior_update_mean_continuous_node
from .posterior_update_precision_continuous_node import (
    posterior_update_precision_continuous_node,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_posterior_update(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the posterior of a continuous node using the standard HGF update.

    The standard HGF posterior update is a two-step process:
    1. Update the posterior precision.
    2. Update the posterior mean and assume that the posterior precision is the value
    updated in the first step.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    posterior_update_precision_continuous_node, posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # update the posterior mean and precision
    attributes[node_idx]["precision"] = posterior_update_precision_continuous_node(
        attributes, edges, node_idx
    )

    attributes[node_idx]["mean"] = posterior_update_mean_continuous_node(
        attributes, edges, node_idx, node_precision=attributes[node_idx]["precision"]
    )

    return attributes
