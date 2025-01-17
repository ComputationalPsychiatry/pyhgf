# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Attributes, Edges


@partial(jit, static_argnames=("edges", "node_idx", "sufficient_stats_fn"))
def posterior_update_exponential_family_dynamic(
    attributes: Dict, edges: Edges, node_idx: int, **args
) -> Attributes:
    r"""Update the hyperparameters of an ef state node using HGF-implied learning rates.

    This posterior update step is usually moved at the end of the update sequence as we
    have to wait that all parent nodes tracking the expected sufficient statistics have
    been updated, and therefore being able to infer the implied learning rate to update
    the :math:`nu` vector. The new impled :math:`nu` is given by a ratio:

    .. math::
        \nu \leftarrow \frac{\delta}{\Delta}

    Where :math:`delta` is the prediction error (the new sufficient statistics compared
    to the expected sufficient statistic), and :math:`Delta` is the differential of
    expectation (what was expected before compared to what is expected after). This
    ratio quantifies how much the model is learning from new observations.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Mathys, C., & Weber, L. (2020). Hierarchical Gaussian Filtering of Sufficient
       Statistic Time Series for Active Inference. In Active Inference (pp. 52–58).
       Springer International Publishing. https://doi.org/10.1007/978-3-030-64919-7_7

    """
    # prediction error - expectation differential
    pe, ed = [], []
    for parent_idx in edges[node_idx].value_parents or []:

        pe.append(
            attributes[parent_idx]["mean"] - attributes[parent_idx]["expected_mean"]
        )

        parent_parent_idx = edges[parent_idx].value_parents[0]
        ed.append(
            attributes[parent_parent_idx]["mean"]
            - attributes[parent_parent_idx]["expected_mean"]
        )

    # implied learning rate
    attributes[node_idx]["nus"] = (jnp.array(pe) / jnp.array(ed)).mean()

    # apply the Bayesian update using fixed learning rates nus
    xis = attributes[node_idx]["xis"] + (1 / (1 + attributes[node_idx]["nus"])) * (
        attributes[node_idx]["observation_ss"] - attributes[node_idx]["xis"]
    )

    # blank update in the case of unobserved value
    attributes[node_idx]["xis"] = jnp.where(
        attributes[node_idx]["observed"], xis, attributes[node_idx]["xis"]
    )

    return attributes
