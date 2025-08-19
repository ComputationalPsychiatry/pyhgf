# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx"))
def continuous_node_value_prediction_error(
    attributes: dict,
    node_idx: int,
) -> dict:
    r"""Compute the value prediction error of a state node.

    The value prediction error :math:`\delta_j^{(k)}` of a continuous state node is
    given by:

    .. math::

        \delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the value parent node that will be updated.

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
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # value prediction error
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # divide by the number of value parents
    if attributes[node_idx]["value_coupling_parents"] is not None:
        value_prediction_error /= len(attributes[node_idx]["value_coupling_parents"])

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
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # compute the volatility prediction error (VOPE)
    volatility_prediction_error = (
        (attributes[node_idx]["expected_precision"] / attributes[node_idx]["precision"])
        + attributes[node_idx]["expected_precision"]
        * (attributes[node_idx]["temp"]["value_prediction_error"]) ** 2
        - 1
    )

    # divide by the number of volatility parents
    if attributes[node_idx]["volatility_coupling_parents"] is not None:
        volatility_prediction_error /= len(
            attributes[node_idx]["volatility_coupling_parents"]
        )

    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        volatility_prediction_error
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction_error(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Store prediction errors in an input node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    node_idx :
        Pointer to the continuous node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

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
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Store value prediction errors
    # -----------------------------
    attributes = continuous_node_value_prediction_error(
        attributes=attributes, node_idx=node_idx
    )

    # Store volatility prediction errors
    # ----------------------------------
    attributes = continuous_node_volatility_prediction_error(
        attributes=attributes, node_idx=node_idx
    )

    return attributes
