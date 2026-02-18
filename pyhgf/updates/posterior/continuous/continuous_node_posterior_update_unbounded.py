# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid

from pyhgf.math import smoothed_rectangular
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_posterior_update_unbounded(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the posterior of a continuous node with unbounded quadratic approximation.

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
    continuous_node_posterior_update_ehgf

    """
    posterior_precision, posterior_mean = posterior_update_unbounded(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    # update the posterior mean and precision using the unbounded update step
    attributes[node_idx]["precision"] = posterior_precision
    attributes[node_idx]["mean"] = posterior_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_unbounded(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> tuple[float, float]:
    """Update the posterior of a continuous node with unbounded quadratic approximation.

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
    continuous_node_posterior_update_ehgf

    """
    volatility_child_idx = edges[node_idx].volatility_children[0]  # type: ignore

    # # Recover the precision of the child node at the previous time step --------------
    previous_child_variance = attributes[volatility_child_idx]["temp"][
        "current_variance"
    ]

    # ----------------------------------------------------------------------------------
    # First quadratic approximation L1 -------------------------------------------------
    # ----------------------------------------------------------------------------------

    # Instead of computing exp(x) / (v + exp(x)) directly as in the equations, we use a
    # numerically stable form with w = 1 / (1 + v * exp(-x)) = sigmoid(x - log(v))
    x = (
        attributes[node_idx]["volatility_coupling_children"][0]
        * attributes[node_idx]["expected_mean"]
        + attributes[volatility_child_idx]["tonic_volatility"]
    )
    previous_child_variance = jnp.maximum(previous_child_variance, 1e-128)
    w_child = sigmoid(x - jnp.log(previous_child_variance))

    delta_child = (
        (1 / attributes[volatility_child_idx]["precision"])
        + (
            attributes[volatility_child_idx]["mean"]
            - (attributes[volatility_child_idx]["expected_mean"])
        )
        ** 2
    ) / (previous_child_variance + jnp.exp(jnp.clip(x, a_min=-80.0, a_max=80.0))) - 1.0

    pi_l1 = attributes[node_idx]["expected_precision"] + 0.5 * attributes[node_idx][
        "volatility_coupling_children"
    ][0] ** 2 * w_child * (1 - w_child)

    mu_l1 = (
        attributes[node_idx]["expected_mean"]
        + (
            (attributes[node_idx]["volatility_coupling_children"][0] * w_child)
            / (2 * pi_l1)
        )
        * delta_child
    )

    # ----------------------------------------------------------------------------------
    # Second quadratic approximation L2 ------------------------------------------------
    # ----------------------------------------------------------------------------------
    phi = jnp.log(previous_child_variance * (2 + jnp.sqrt(3)))

    w_phi = jnp.exp(
        attributes[node_idx]["volatility_coupling_children"][0] * phi
        + attributes[volatility_child_idx]["tonic_volatility"]
    ) / (
        previous_child_variance
        + jnp.exp(
            attributes[node_idx]["volatility_coupling_children"][0] * phi
            + attributes[volatility_child_idx]["tonic_volatility"]
        )
    )

    delta_phi = (
        (1 / attributes[volatility_child_idx]["precision"])
        + (
            attributes[volatility_child_idx]["mean"]
            - (attributes[volatility_child_idx]["expected_mean"])
        )
        ** 2
    ) / (
        previous_child_variance
        + jnp.exp(
            attributes[node_idx]["volatility_coupling_children"][0] * phi
            + attributes[volatility_child_idx]["tonic_volatility"]
        )
    ) - 1.0

    pi_l2 = attributes[node_idx]["expected_precision"] + 0.5 * attributes[node_idx][
        "volatility_coupling_children"
    ][0] ** 2 * w_phi * (w_phi + (2 * w_phi - 1) * delta_phi)

    mu_hat_phi = ((2.0 * pi_l2 - 1.0) * phi + attributes[node_idx]["expected_mean"]) / (
        2.0 * pi_l2
    )

    mu_l2 = (
        mu_hat_phi
        + (
            (attributes[node_idx]["volatility_coupling_children"][0] * w_phi)
            / (2 * pi_l2)
        )
        * delta_phi
    )

    # ----------------------------------------------------------------------------------
    # compute the full quadratic approximation -----------------------------------------
    # ----------------------------------------------------------------------------------
    theta_l = jnp.sqrt(
        1.2
        * (
            (
                (1 / attributes[volatility_child_idx]["precision"])
                + (
                    attributes[volatility_child_idx]["mean"]
                    - attributes[volatility_child_idx]["expected_mean"]
                )
                ** 2
            )
            / (previous_child_variance * pi_l1)
        )
    )

    # compute the weigthing of the two approximations
    # using the smoothed rectangular function b
    weigthing = smoothed_rectangular(
        x=attributes[node_idx]["expected_mean"],
        theta_l=theta_l,
        phi_l=8.0,
        theta_r=0.0,
        phi_r=1.0,
    )

    posterior_precision = (1 - weigthing) * pi_l1 + weigthing * pi_l2
    posterior_mean = (1 - weigthing) * mu_l1 + weigthing * mu_l2

    return posterior_precision, posterior_mean
