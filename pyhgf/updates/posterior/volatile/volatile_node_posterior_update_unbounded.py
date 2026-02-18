# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid

from pyhgf.math import smoothed_rectangular
from pyhgf.typing import Edges
from pyhgf.updates.prediction_error.volatile import volatile_node_prediction_error

from .posterior_update_value_level import (
    posterior_update_mean_value_level,
    posterior_update_precision_value_level,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_posterior_update_unbounded(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> dict:
    """Update a volatile node with unbounded quadratic approximation for volatility.

    This uses an unbounded quadratic approximation for the implicit volatility level
    update, analogous to the unbounded update for continuous nodes with volatility
    children. The value level is updated using the standard procedure.

    1. Update value level using children's value prediction errors (standard order)
    2. Recompute volatility prediction error using updated value level
    3. Update volatility level using unbounded quadratic approximation

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`.
    node_idx :
        Pointer to the volatile node that needs to be updated.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    volatile_node_posterior_update, volatile_node_posterior_update_ehgf

    """
    # 1. UPDATE VALUE LEVEL (external facing) - standard order
    # Update precision first
    precision_value = posterior_update_precision_value_level(
        attributes, edges, node_idx
    )
    attributes[node_idx]["precision"] = precision_value

    # Update mean using new precision
    mean_value = posterior_update_mean_value_level(
        attributes, edges, node_idx, precision_value
    )
    attributes[node_idx]["mean"] = mean_value

    # 2. COMPUTE PREDICTION ERROR
    # Now that value level has been updated, compute the value and volatility PE
    attributes = volatile_node_prediction_error(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    # 3. UPDATE VOLATILITY LEVEL (unbounded quadratic approximation)
    precision_vol, mean_vol = posterior_update_volatility_level_unbounded(
        attributes, node_idx
    )
    attributes[node_idx]["precision_vol"] = precision_vol
    attributes[node_idx]["mean_vol"] = mean_vol

    return attributes


@partial(jit, static_argnames=("node_idx",))
def posterior_update_volatility_level_unbounded(
    attributes: dict,
    node_idx: int,
) -> tuple[float, float]:
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
    phi = jnp.log(previous_child_variance * (2 + jnp.sqrt(3)))

    w_phi = jnp.exp(
        volatility_coupling * phi + attributes[node_idx]["tonic_volatility"]
    ) / (
        previous_child_variance
        + jnp.exp(volatility_coupling * phi + attributes[node_idx]["tonic_volatility"])
    )

    delta_phi = (
        (1 / attributes[node_idx]["precision"])
        + (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2
    ) / (
        previous_child_variance
        + jnp.exp(volatility_coupling * phi + attributes[node_idx]["tonic_volatility"])
    ) - 1.0

    pi_l2 = attributes[node_idx][
        "expected_precision_vol"
    ] + 0.5 * volatility_coupling**2 * w_phi * (w_phi + (2 * w_phi - 1) * delta_phi)

    mu_hat_phi = (
        (2.0 * pi_l2 - 1.0) * phi + attributes[node_idx]["expected_mean_vol"]
    ) / (2.0 * pi_l2)

    mu_l2 = mu_hat_phi + ((volatility_coupling * w_phi) / (2 * pi_l2)) * delta_phi

    # ----------------------------------------------------------------------------------
    # Compute the full quadratic approximation
    # ----------------------------------------------------------------------------------
    theta_l = jnp.sqrt(
        1.2
        * (
            (
                (1 / attributes[node_idx]["precision"])
                + (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"])
                ** 2
            )
            / (previous_child_variance * pi_l1)
        )
    )

    # Compute the weighting of the two approximations
    # using the smoothed rectangular function b
    weighting = smoothed_rectangular(
        x=attributes[node_idx]["expected_mean_vol"],
        theta_l=theta_l,
        phi_l=8.0,
        theta_r=0.0,
        phi_r=1.0,
    )

    posterior_precision = (1 - weighting) * pi_l1 + weighting * pi_l2
    posterior_mean = (1 - weighting) * mu_l1 + weighting * mu_l2

    return posterior_precision, posterior_mean
