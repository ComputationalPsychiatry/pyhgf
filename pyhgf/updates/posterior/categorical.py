# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.math import binary_surprise, dirichlet_kullback_leibler
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def categorical_state_update(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the categorical input node given an array of binary observations.

    This function should be called **after** the update of the implied binary HGFs. It
    receives a `None` as the boolean observations are passed to the binary inputs
    directly. This update uses the expected probability of the binary input to update
    an implied Dirichlet distribution.

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
    # get the expected values before the update
    expected_mean = jnp.array([
        attributes[value_parent_idx]["expected_mean"]
        for value_parent_idx in edges[node_idx].value_parents  # type: ignore
    ])

    # get the new values after the update from the continuous state
    updated_mean = jnp.array([
        attributes[edges[parent_idx].value_parents[0]]["mean"]
        for parent_idx in edges[node_idx].value_parents  # type: ignore
    ])
    updated_mean = 1 / (1 + jnp.exp(-updated_mean))  # logit transform

    # compute the prediction error (observed - expected) at time K
    pe = attributes[node_idx]["mean"] - expected_mean

    # the differential of expectations (parent posterior - parents expectation)
    delta_xi = updated_mean - expected_mean

    # using the new PE, we can update nu and the alpha vector
    nu = (pe / delta_xi) - 1
    alpha = (nu * expected_mean) + 1  # concentration parameters for the Dirichlet

    # in case alpha contains NaNs
    # alpha = jnp.where(jnp.isnan(alpha), 1.0, alpha)

    # compute Bayesian surprise as :
    # 1 - KL divergence from the concentration parameters
    attributes[node_idx]["kl_divergence"] = dirichlet_kullback_leibler(
        attributes[node_idx]["alpha"], alpha
    )
    # 2 - the sum of binary surprises observed in the parents nodes
    attributes[node_idx]["surprise"] = jnp.sum(
        binary_surprise(x=attributes[node_idx]["mean"], expected_mean=expected_mean)
    )

    # save the new concentration parameters
    attributes[node_idx]["alpha"] = alpha
    attributes[node_idx]["mean"] = updated_mean

    # prediction mean
    # attributes[node_idx]["mean"] = alpha / jnp.sum(alpha)

    return attributes
