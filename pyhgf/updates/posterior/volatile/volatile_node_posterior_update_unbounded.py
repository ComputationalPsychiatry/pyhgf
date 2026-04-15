# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid

from pyhgf.math import smoothed_rectangular


@partial(jit, static_argnames=("node_idx",))
def volatile_node_posterior_update_unbounded(
    attributes: dict,
    node_idx: int,
) -> dict:
    """Update the volatility level using an unbounded quadratic approximation.

    This adapts the continuous unbounded update to operate on the implicit volatility
    level within a volatile node. The "volatility child" in this context is the
    value level of the same node, and the "volatility parent" is the implicit
    volatility level.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile node.

    Returns
    -------
    posterior_precision :
        The updated precision of the volatility level.
    posterior_mean :
        The updated mean of the volatility level.

    """
    # The value level acts as the "volatility child" of the implicit volatility level.
    # Map variables:
    #   - Parent (volatility level): expected_mean_vol, expected_precision_vol
    #   - Child (value level): mean, expected_mean, precision, tonic_volatility
    #   - Coupling: volatility_coupling_internal
    volatility_coupling = attributes[node_idx]["volatility_coupling_internal"]

    # Previous variance of the value level (child)
    previous_child_variance = attributes[node_idx]["temp"]["current_variance"]
    previous_child_variance = jnp.maximum(previous_child_variance, 1e-128)

    # ----------------------------------------------------------------------------------
    # First quadratic approximation L1
    # ----------------------------------------------------------------------------------
    x = (
        volatility_coupling * attributes[node_idx]["expected_mean_vol"]
        + attributes[node_idx]["tonic_volatility"]
    )

    w_child = sigmoid(x - jnp.log(previous_child_variance))

    delta_child = (
        (1 / attributes[node_idx]["precision"])
        + (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2
    ) / (previous_child_variance + jnp.exp(jnp.clip(x, a_min=-80.0, a_max=80.0))) - 1.0

    pi_l1 = attributes[node_idx][
        "expected_precision_vol"
    ] + 0.5 * volatility_coupling**2 * w_child * (1 - w_child)

    mu_l1 = (
        attributes[node_idx]["expected_mean_vol"]
        + ((volatility_coupling * w_child) / (2 * pi_l1)) * delta_child
    )

    # ----------------------------------------------------------------------------------
    # Second quadratic approximation L2
    # ----------------------------------------------------------------------------------

    # Canonical expansion point and its map back to native space
    ka = volatility_coupling
    om = attributes[node_idx]["tonic_volatility"]
    phi_canon = jnp.log(previous_child_variance * (2 + jnp.sqrt(3.0)))
    phi_full = (phi_canon - om) / ka

    # At phi_full, exp(ka*phi_full + om) = previous_child_variance*(2+sqrt(3))
    # by construction — use this directly to avoid numerical round-trips
    exp_at_phi = previous_child_variance * (2 + jnp.sqrt(3.0))

    w_phi = exp_at_phi / (previous_child_variance + exp_at_phi)

    delta_phi = (
        (1 / attributes[node_idx]["precision"])
        + (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2
    ) / (previous_child_variance + exp_at_phi) - 1.0

    pi_l2 = attributes[node_idx]["expected_precision_vol"] + 0.5 * ka**2 * w_phi * (
        w_phi + (2 * w_phi - 1) * delta_phi
    )

    mu_hat_phi = (
        (pi_l2 - attributes[node_idx]["expected_precision_vol"]) * phi_full
        + attributes[node_idx]["expected_precision_vol"]
        * attributes[node_idx]["expected_mean_vol"]
    ) / pi_l2

    mu_l2 = mu_hat_phi + ((ka * w_phi) / (2 * pi_l2)) * delta_phi

    # ----------------------------------------------------------------------------------
    # Compute the full quadratic approximation
    # ----------------------------------------------------------------------------------

    # Total posterior uncertainty at child level (be_aux in the Matlab code)
    be_aux = (1 / attributes[node_idx]["precision"]) + (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    ) ** 2

    # Blending operates in canonical exponent space y = ka * muhat + om
    y_pred = ka * attributes[node_idx]["expected_mean_vol"] + om
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

    attributes[node_idx]["precision_vol"] = posterior_precision
    attributes[node_idx]["mean_vol"] = posterior_mean

    return attributes
