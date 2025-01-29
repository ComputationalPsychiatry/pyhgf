# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import grad, jit
from jax.lax import cond
from jax.tree_util import Partial

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
) -> float:
    r"""Update the precision of a state node using the volatility prediction errors.

    #. Precision update from value coupling.

    The new precision of a state node :math:`b` value coupled with other input and/or
    state nodes :math:`j` at time :math:`k` is given by:

    For linear coupling (default)

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \kappa_j^2 \hat{\pi}_j^{(k)}

    Where :math:`\kappa_j` is the volatility coupling strength between the child node
    and the state node and :math:`\delta_j^{(k)}` is the value prediction error that
    was computed beforehand by
    :py:func:`pyhgf.updates.prediction_errors.continuous.continuous_node_value_prediction_error`.

    For non-linear value coupling:

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \hat{\pi}_j^{(k)} * (\kappa_j^2 * g'_{j,b}(\mu_b^(k-1))^2 -
            g''_{j,b}(\mu_b^(k-1))*\delta_j)

    #. Precision update from volatility coupling.

    The new precision of a state node :math:`b` volatility coupled with other input
    and/or state nodes :math:`j` at time :math:`k` is given by:

    .. math::

        \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
        \frac{1}{2} \left( \kappa_j \gamma_j^{(k)} \right) ^2 +
        \left( \kappa_j \gamma_j^{(k)} \right) ^2 \Delta_j^{(k)} -
        \frac{1}{2} \kappa_j^2 \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the effective precision of the prediction, given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    that was computed in the prediction step.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number
        of nodes. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision :
        The new posterior precision.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # ----------------------------------------------------------------------------------
    # Decide which update to use depending on the presence of observed value in the
    # children nodes. If no values were observed, the precision should increase
    # as a function of time using the function precision_missing_values(). Otherwise,
    # we use regular HGF updates for value and volatility couplings.
    # ----------------------------------------------------------------------------------

    # For all children, get the `observed` flag - if all these values are 0.0, the node
    # has not received any observations and we should call precision_missing_values()
    observations = []
    if edges[node_idx].value_children is not None:
        for children_idx in edges[node_idx].value_children:  # type: ignore
            observations.append(attributes[children_idx]["observed"])
    if edges[node_idx].volatility_children is not None:
        for children_idx in edges[node_idx].volatility_children:  # type: ignore
            observations.append(attributes[children_idx]["observed"])
    observations = jnp.any(jnp.array(observations))

    posterior_precision = cond(
        observations,
        Partial(precision_update, edges=edges, node_idx=node_idx),
        Partial(precision_update_missing_values, edges=edges, node_idx=node_idx),
        attributes,
    )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update(attributes: Dict, edges: Edges, node_idx: int) -> float:
    """Compute new precision in the case of observed values.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number
        of nodes. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision :
        The new posterior precision when at least one of the children has
        observed a new value. We then use the regular HGF update for volatility
        coupling.

    """
    # sum the prediction errors from both value and volatility coupling
    precision_weigthed_prediction_error = 0.0

    # Value coupling updates - update the precision of a value parent
    # ---------------------------------------------------------------
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            if coupling_fn is None:  # linear coupling
                coupling_fn_prime = 1
                coupling_fn_second = 0
            else:  # non-linear coupling
                coupling_fn_prime = grad(coupling_fn)(attributes[node_idx]["mean"]) ** 2
                value_prediction_error = attributes[value_child_idx]["temp"][
                    "value_prediction_error"
                ]
                coupling_fn_second = (
                    grad(grad(coupling_fn))(attributes[node_idx]["mean"])
                    * value_prediction_error
                )

            # cancel the prediction error if the child value was not observed
            precision_weigthed_prediction_error += (
                value_coupling**2
                * attributes[value_child_idx]["expected_precision"]
                * coupling_fn_prime
                - coupling_fn_second
            ) * attributes[value_child_idx]["observed"]

    # Volatility coupling updates - update the precision of a volatility parent
    # -------------------------------------------------------------------------
    if edges[node_idx].volatility_children is not None:
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            # volatility weigthed precision for the volatility child (γ)
            effective_precision = attributes[volatility_child_idx]["temp"][
                "effective_precision"
            ]

            # retrieve the volatility prediction error
            volatility_prediction_error = attributes[volatility_child_idx]["temp"][
                "volatility_prediction_error"
            ]

            # sum over all volatility children
            # cancel the prediction error if the child value was not observed
            precision_weigthed_prediction_error += (
                0.5 * (volatility_coupling * effective_precision) ** 2
                + (volatility_coupling * effective_precision) ** 2
                * volatility_prediction_error
                - 0.5
                * volatility_coupling**2
                * effective_precision
                * volatility_prediction_error
            ) * attributes[volatility_child_idx]["observed"]

    # Compute the new posterior precision
    # using value prediction errors from both value and volatility coupling
    posterior_precision = (
        attributes[node_idx]["expected_precision"] + precision_weigthed_prediction_error
    )

    # ensure the new precision is greater than 0
    posterior_precision = jnp.where(
        posterior_precision > 1e-128, posterior_precision, jnp.nan
    )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update_missing_values(
    attributes: Dict, edges: Edges, node_idx: int
) -> float:
    """Compute new precision in the case of missing observations.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number
        of nodes. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision_missing_values :
        The new posterior precision in the case of missing values in all child nodes.
        The new precision decreases proportionally to the time elapsed, accounting for
        the influence of volatility parents.

    """
    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling * attributes[volatility_parents_idx]["mean"]
            )

    # compute the new predicted_volatility from the total volatility
    time_step = attributes[-1]["time_step"]
    predicted_volatility = time_step * jnp.exp(total_volatility)

    # Estimate the new precision for the continuous state node
    posterior_precision_missing_values = 1 / (
        (1 / attributes[node_idx]["precision"]) + predicted_volatility
    )

    return posterior_precision_missing_values
