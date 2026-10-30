# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node import (
    _has_observations,
)


@partial(jit, static_argnames=("node_idx", "edges"))
def continuous_node_value_prediction_error(
    attributes: dict,
    node_idx: int,
    edges: Edges = (),
) -> dict:
    r"""Compute the value prediction error of a state node.

    The value prediction error :math:`\delta_j^{(k)}` of a continuous state node is
    given by:

    .. math::

        \delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}

    See [1]_ for more details.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the value parent node that will be updated.
    edges :
        Unused. Kept for API compatibility with callers that pass edges.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_volatility_prediction_error, continuous_node_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1

    """
    # value prediction error
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # send to the value parent node for later use in the update step
    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error

    return attributes


@partial(jit, static_argnames=("node_idx"))
def continuous_node_volatility_prediction_error(
    attributes: dict, node_idx: int
) -> dict:
    r"""Compute the volatility prediction error of a state node.

    The volatility prediction error :math:`\Delta_j^{(k)}` of a state node
    :math:`j` is given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    Every volatility parent of the node receives the full :math:`\Delta_j^{(k)}`, as
    in the unbounded update. See [1]_ for more details.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_value_prediction_error, continuous_node_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1

    """
    # compute the volatility prediction error (VOPE)
    volatility_prediction_error = (
        (attributes[node_idx]["expected_precision"] / attributes[node_idx]["precision"])
        + attributes[node_idx]["expected_precision"]
        * (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2
        - 1
    )

    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        volatility_prediction_error
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction_error(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Store the prediction errors of a continuous state node.

    A node with value or volatility children also records whether at least one of
    them was observed at this step, in its ``observed`` flag. Its parents then treat
    it as unobserved after a step without observations: they receive no prediction
    error from it and keep their predicted precision (see
    :func:`pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node.precision_update_missing_values`).
    The flag of a node without children is the one set by the observation step.

    See [1]_ for more details.

    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the continuous node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. The tuple has the same length as the
        node number. For each node, the index lists the value and volatility parents
        and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_volatility_prediction_error, continuous_node_value_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    # A node with children is observed when at least one of its children was
    # ------------------------------------------------------------------------
    if (
        edges[node_idx].value_children is not None
        or edges[node_idx].volatility_children is not None
    ):
        attributes[node_idx]["observed"] = jnp.where(
            _has_observations(attributes, edges, node_idx), 1, 0
        )

    # Store value prediction errors
    # -----------------------------
    attributes = continuous_node_value_prediction_error(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    # Store volatility prediction errors
    # ----------------------------------
    attributes = continuous_node_volatility_prediction_error(
        attributes=attributes, node_idx=node_idx
    )

    return attributes
