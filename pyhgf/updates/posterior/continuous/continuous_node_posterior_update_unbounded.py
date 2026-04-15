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

    # Canonical expansion point and its map back to native space
    ka = attributes[node_idx]["volatility_coupling_children"][0]
    om = attributes[volatility_child_idx]["tonic_volatility"]
    phi_canon = jnp.log(previous_child_variance * (2 + jnp.sqrt(3.0)))
    phi_full = (phi_canon - om) / ka

    # At phi_full, exp(ka*phi_full + om) = previous_child_variance*(2+sqrt(3))
    # by construction — use this directly to avoid numerical round-trips
    exp_at_phi = previous_child_variance * (2 + jnp.sqrt(3.0))

    w_phi = exp_at_phi / (previous_child_variance + exp_at_phi)

    delta_phi = (
        (1 / attributes[volatility_child_idx]["precision"])
        + (
            attributes[volatility_child_idx]["mean"]
            - attributes[volatility_child_idx]["expected_mean"]
        )
        ** 2
    ) / (previous_child_variance + exp_at_phi) - 1.0

    pi_l2 = attributes[node_idx]["expected_precision"] + 0.5 * ka**2 * w_phi * (
        w_phi + (2 * w_phi - 1) * delta_phi
    )

    mu_hat_phi = (
        (pi_l2 - attributes[node_idx]["expected_precision"]) * phi_full
        + attributes[node_idx]["expected_precision"]
        * attributes[node_idx]["expected_mean"]
    ) / pi_l2

    mu_l2 = mu_hat_phi + ((ka * w_phi) / (2 * pi_l2)) * delta_phi

    # ----------------------------------------------------------------------------------
    # compute the full quadratic approximation -----------------------------------------
    # ----------------------------------------------------------------------------------

    # Total posterior uncertainty at child level (be_aux in the Matlab code)
    be_aux = (1 / attributes[volatility_child_idx]["precision"]) + (
        attributes[volatility_child_idx]["mean"]
        - attributes[volatility_child_idx]["expected_mean"]
    ) ** 2

    # Blending operates in canonical exponent space y = ka * muhat + om
    y_pred = (
        ka * attributes[node_idx]["expected_mean"]
        + attributes[volatility_child_idx]["tonic_volatility"]
    )
    theta_l = -jnp.sqrt(1.2 * 2.0 * be_aux / previous_child_variance)

    # Compute the weighting of the two approximations
    # using the smoothed rectangular function b
    weighting = smoothed_rectangular(
        x=y_pred,
        theta_l=theta_l,
        phi_l=8.0,
        theta_r=0.0,
        phi_r=1.0,
    )

    posterior_precision = (1 - weighting) * pi_l1 + weighting * pi_l2
    posterior_mean = (1 - weighting) * mu_l1 + weighting * mu_l2

    return posterior_precision, posterior_mean
