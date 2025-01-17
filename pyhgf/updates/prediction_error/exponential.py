# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Callable, Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Attributes, Edges


@partial(jit, static_argnames=("edges", "node_idx", "sufficient_stats_fn"))
def prediction_error_update_exponential_family_fixed(
    attributes: Dict, edges: Edges, node_idx: int, sufficient_stats_fn: Callable, **args
) -> Attributes:
    r"""Update the parameters of an exponential family distribution.

    Assuming that :math:`nu` is fixed, updating the hyperparameters of the distribution
    is given by:

    .. math::
        \xi \leftarrow \xi + \frac{1}{\nu + 1}(t(x)-\xi)

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
    sufficient_stats_fn :
        Compute the sufficient statistics of the probability distribution. This should
        be one of the method implemented in the distribution class in
        :py:class:`pyhgf.math.Normal`, for a univariate normal.

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
    # retrieve the expected sufficient statistics from new observations
    attributes[node_idx]["observation_ss"] = sufficient_stats_fn(
        x=attributes[node_idx]["mean"]
    )

    # apply the Bayesian update using fixed learning rates nus
    xis = attributes[node_idx]["xis"] + (1 / (1 + attributes[node_idx]["nus"])) * (
        attributes[node_idx]["observation_ss"] - attributes[node_idx]["xis"]
    )

    # blank update in the case of unobserved value
    attributes[node_idx]["xis"] = jnp.where(
        attributes[node_idx]["observed"], xis, attributes[node_idx]["xis"]
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx", "sufficient_stats_fn"))
def prediction_error_update_exponential_family_dynamic(
    attributes: Dict, edges: Edges, node_idx: int, sufficient_stats_fn: Callable, **args
) -> Attributes:
    r"""Pass the expected sufficient statistics to the implied continuous nodes.

    When updating an exponential family state node without assuming that :math:`nu` is
    fixed, the node convert the new observation into sufficient statistics and pass the
    values to the implied continuous nodes. The new values for the vector :math:`nu`
    are recovered in another posterior update, by observing the learning rate in the
    continuous nodes, usually at the end of the sequence.

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
    sufficient_stats_fn :
        Compute the sufficient statistics of the probability distribution. This should
        be one of the method implemented in the distribution class in
        :py:class:`pyhgf.math.Normal`, for a univariate normal.

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
    # retrieve the expected sufficient statistics from new observations
    attributes[node_idx]["observation_ss"] = sufficient_stats_fn(
        x=attributes[node_idx]["mean"]
    )

    # pass the expected sufficient statistics to the continuous parent nodes
    for parent_idx, value in zip(
        edges[node_idx].value_parents or [],
        attributes[node_idx]["observation_ss"],
        strict=True,
    ):

        # blank update in the case of unobserved value
        attributes[parent_idx]["observed"] = attributes[node_idx]["observed"]

        # pass the new value
        attributes[parent_idx]["mean"] = value

    return attributes
