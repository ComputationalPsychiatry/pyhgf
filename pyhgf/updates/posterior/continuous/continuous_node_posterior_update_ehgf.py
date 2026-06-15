# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges

from .posterior_update_mean_continuous_node import (
    posterior_update_mean_continuous_node,
    posterior_update_mean_continuous_node_mean_field,
)
from .posterior_update_precision_continuous_node import (
    posterior_update_precision_continuous_node,
    posterior_update_precision_continuous_node_mean_field,
)


@partial(jit, static_argnames=("edges", "node_idx", "max_posterior_precision"))
def continuous_node_posterior_update_ehgf(
    attributes: dict,
    node_idx: int,
    edges: Edges,
    max_posterior_precision: float = 1e10,
    **args,
) -> dict:
    """Update the posterior of a continuous node using the eHGF update.

    The eHGF posterior update is a two-step process:
    1. Update the posterior mean and assume that the posterior precision is equal to
    the expected precision.
    2. Update the posterior precision.

    .. note::
        By updating the mean first, and approximating the precision using the expected,
        precision, this update step often perform better than the regular update and
        limit the occurence of negative precision that cause the model to fail under
        some circumstances

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
    max_posterior_precision :
        Upper bound applied to the posterior precision write. Default ``1e10``.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    posterior_update_precision_continuous_node, posterior_update_mean_continuous_node
    """
    # update the posterior mean and precision using the eHGF update step
    # we start with the mean update using the expected precision as an approximation
    posterior_mean = posterior_update_mean_continuous_node(
        attributes,
        edges,
        node_idx,
        node_precision=attributes[node_idx]["expected_precision"],
    )
    attributes[node_idx]["mean"] = posterior_mean

    posterior_precision = posterior_update_precision_continuous_node(
        attributes,
        edges,
        node_idx,
    )
    attributes[node_idx]["precision"] = jnp.minimum(
        posterior_precision, max_posterior_precision
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx", "max_posterior_precision"))
def continuous_node_posterior_update_ehgf_mean_field(
    attributes: dict,
    node_idx: int,
    edges: Edges,
    max_posterior_precision: float = 1e10,
    **args,
) -> dict:
    """Update the posterior of a continuous node using the eHGF update with mean-field.

    The eHGF posterior update is a two-step process:
    1. Update the posterior mean and assume that the posterior precision is equal to
    the expected precision.
    2. Update the posterior precision.

    See [1]_ for more details.

    .. note::
        By updating the mean first, and approximating the precision using the expected,
        precision, this update step often perform better than the regular update and
        limit the occurence of negative precision that cause the model to fail under
        some circumstances

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
    max_posterior_precision :
        Upper bound applied to the posterior precision write. Default ``1e10``.

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
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    posterior_mean = posterior_update_mean_continuous_node_mean_field(
        attributes,
        edges,
        node_idx,
        node_precision=attributes[node_idx]["expected_precision"],
    )
    attributes[node_idx]["mean"] = posterior_mean

    posterior_precision = posterior_update_precision_continuous_node_mean_field(
        attributes,
        edges,
        node_idx,
    )
    attributes[node_idx]["precision"] = jnp.minimum(
        posterior_precision, max_posterior_precision
    )

    return attributes
