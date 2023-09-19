# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
    pi_value_parent: ArrayLike,
) -> Array:
    value_parent_value_parent_idxs = edges[value_parent_idx].value_parents

    # 3. get muhat_value_parent from value parent (x2)

    # 3.1
    driftrate = attributes[value_parent_idx]["rho"]

    # 3.2 Look at the (optional) value parent's value parents
    # and update driftrate accordingly
    if value_parent_value_parent_idxs is not None:
        for value_parent_value_parent_idx, psi in zip(
            value_parent_value_parent_idxs,
            attributes[value_parent_idx]["psis_parents"],
        ):
            driftrate += psi * attributes[value_parent_value_parent_idx]["mu"]

    # 3.3
    muhat_value_parent = attributes[value_parent_idx]["mu"] + time_step * driftrate

    # gather PE updates from other binray child nodes if the parent has many
    # this part corresponds to the sum of children required for the
    # multi-children situations
    pe_children = 0.0
    for child_idx, psi_child in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["psis_children"],
    ):
        vape_child = attributes[child_idx]["mu"] - attributes[child_idx]["muhat"]
        pe_children += (psi_child * vape_child) / pi_value_parent

    # 4.
    mu_value_parent = muhat_value_parent + pe_children

    return mu_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_precision_value_parent(
    attributes: Dict, edges: Edges, time_step: float, value_parent_idx: int
) -> Array:
    value_parent_volatility_parent_idxs = edges[value_parent_idx].volatility_parents

    # get logvolatility
    logvol = attributes[value_parent_idx]["omega"]

    # 1.1.2 Look at the (optional) va_pa's volatility parents
    # and update logvol accordingly
    if value_parent_volatility_parent_idxs is not None:
        for value_parent_volatility_parent_idx, k in zip(
            value_parent_volatility_parent_idxs,
            attributes[value_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[value_parent_volatility_parent_idx]["mu"]

    # 1.1.3 Compute new_nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # 1.2 Compute new value for nu and pihat
    pihat_value_parent, nu_value_parent = [
        1 / (1 / attributes[value_parent_idx]["pi"] + new_nu),
        new_nu,
    ]

    # 2.
    # gather precision updates from other binray input nodes
    # this part corresponds to the sum over children
    # required for the multi-children situations
    pi_children = 0.0
    for child_idx, psi_child in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["psis_children"],
    ):
        pihat_child = attributes[child_idx]["pihat"]
        pi_children += psi_child * (1 / pihat_child)

    pi_value_parent = pihat_value_parent + pi_children

    return pi_value_parent, nu_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    """Update the mean and precision of the value parent of a binary node.

    Updating the posterior distribution of the value parent is a two-step process:
    1. Update the posterior precision using
    :py:fun:`continuous_node_update_precision_value_parent`.
    2. Update the posterior mean value using
    :py:fun:`continuous_node_update_mean_value_parent`.

    Parameters
    ----------
    attributes :
        The nodes' parameters.
    edges :
        The edges of the network as a tuple of :py:class:`pyhgf.typing.Indexes` with
        the same length as node number. For each node, the index list value and
        volatility parents.
    time_step :
        The interval between the previous time point and the current time point.
    value_parent_idx :
        Pointer to the value parent node.

    Returns
    -------
    pi_value_parent :
        The precision (:math:`\\pi`) of the value parent.
    mu_value_parent :
        The mean (:math:`\\mu`) of the value parent.
    nu_value_parent :

    """
    pi_value_parent, nu_value_parent = prediction_error_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    mu_value_parent = prediction_error_mean_value_parent(
        attributes, edges, time_step, value_parent_idx, pi_value_parent
    )

    return pi_value_parent, mu_value_parent, nu_value_parent
