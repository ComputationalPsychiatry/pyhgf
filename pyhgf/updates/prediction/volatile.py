from functools import partial

import jax.numpy as jnp
from jax import Array, grad, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx",))
def predict_precision_volatility_level(
    attributes: dict,
    node_idx: int,
) -> tuple[Array, Array]:
    """Predict the precision of the implicit volatility level."""
    time_step = attributes[-1]["time_step"]

    # Get volatility level parameters
    precision_vol = attributes[node_idx]["precision_vol"]
    tonic_volatility_vol = attributes[node_idx]["tonic_volatility_vol"]

    # Compute predicted volatility for the volatility level
    predicted_volatility_vol = time_step * jnp.exp(tonic_volatility_vol)
    predicted_volatility_vol = jnp.where(
        predicted_volatility_vol > 1e-128, predicted_volatility_vol, jnp.nan
    )

    # Expected precision
    expected_precision_vol = 1 / ((1 / precision_vol) + predicted_volatility_vol)

    # Effective precision
    effective_precision_vol = predicted_volatility_vol * expected_precision_vol

    return expected_precision_vol, effective_precision_vol


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_mean_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> Array:
    """Predict the mean of the value level (external facing).

    This uses value parents if they exist.
    """
    time_step = attributes[-1]["time_step"]

    # List the node's value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Get the drift rate from the node
    driftrate = 0.0

    # Look at the (optional) value parents for this node
    if value_parents_idxs is not None:
        for value_parent_idx, value_coupling_parent in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            # Get the coupling function
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            if coupling_fn is None:
                parent_value = attributes[value_parent_idx]["expected_mean"]
            else:
                parent_value = coupling_fn(
                    attributes[value_parent_idx]["expected_mean"]
                )

            driftrate += value_coupling_parent * parent_value

    # The new expected mean from the previous value
    expected_mean = (
        attributes[node_idx]["autoconnection_strength"] * attributes[node_idx]["mean"]
    ) + (time_step * driftrate)

    return expected_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> tuple[Array, Array]:
    """Predict the precision of the value level using the implicit volatility level.

    The volatility level's mean modulates the value level's precision. The implicit
    volatility level is treated as a full Gaussian parent: the exact
    moment-generating-function correction ``κ² / (2 · π̂_vol)`` is added inside the
    log-volatility exponent. In addition, each value parent contributes a first-order
    Laplace term ``(t · α · g'(μ̂_b))² / π̂_b`` to the marginal predictive variance. Both
    corrections vanish in the limit of perfectly known parents, recovering the canonical
    formula exactly.

    """
    time_step = attributes[-1]["time_step"]

    # Get value level parameters
    precision = attributes[node_idx]["precision"]
    tonic_volatility = attributes[node_idx]["tonic_volatility"]

    # Get volatility level's expected mean and precision (already computed)
    expected_mean_vol = attributes[node_idx]["expected_mean_vol"]
    expected_precision_vol = attributes[node_idx]["expected_precision_vol"]

    # Get internal coupling strength
    volatility_coupling_internal = attributes[node_idx]["volatility_coupling_internal"]

    # Total volatility = tonic + linear contribution of the implicit volatility
    # parent + closed-form moment-generating-function correction κ²/(2 π̂_vol)
    # that arises from marginalising over the volatility parent's Gaussian.
    total_volatility = (
        tonic_volatility
        + (volatility_coupling_internal * expected_mean_vol)
        + (volatility_coupling_internal**2) / (2.0 * expected_precision_vol)
    )

    # Compute predicted volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Laplace value-coupling correction. The conditional mean of the value level
    # is linearised around μ̂_b via a first-order Taylor expansion of the coupling
    # function g; the variance contribution from each value parent is then
    # (t · α · g'(μ̂_b))² / π̂_b. The factor t arises because the value-parent
    # contribution to the mean is scaled by the time step in
    # :func:`predict_mean_value_level`.
    value_parents_idxs = edges[node_idx].value_parents
    value_coupling_variance = jnp.zeros_like(predicted_volatility)
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            mu_b = attributes[value_parent_idx]["expected_mean"]
            if coupling_fn is None:
                g_prime = jnp.ones_like(mu_b)
            else:
                g_prime = grad(coupling_fn)(mu_b)

            value_coupling_variance += (time_step * psi * g_prime) ** 2 / attributes[
                value_parent_idx
            ]["expected_precision"]

    # Expected precision = inverse marginal predictive variance.
    expected_precision = 1 / (
        (1 / precision) + predicted_volatility + value_coupling_variance
    )

    # Effective precision (γ): only the volatility-driven part enters γ, since
    # γ is consumed by the volatility-coupling posterior update.
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, effective_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_prediction(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the expected mean and expected precision of a value-volatility node.

    This node has two internal levels:
    1. Volatility level (implicit, internal)
    2. Value level (external facing)

    The volatility level predicts first, then affects the value level's precision.
    """
    # Store current variance for potential unbounded updates
    attributes[node_idx]["temp"]["current_variance"] = (
        1 / attributes[node_idx]["precision"]
    )

    # 1. PREDICT VOLATILITY LEVEL (implicit internal state)
    expected_precision_vol, effective_precision_vol = (
        predict_precision_volatility_level(attributes, node_idx)
    )

    attributes[node_idx]["expected_mean_vol"] = attributes[node_idx]["mean_vol"]
    attributes[node_idx]["expected_precision_vol"] = expected_precision_vol
    attributes[node_idx]["temp"]["effective_precision_vol"] = effective_precision_vol

    # 2. PREDICT VALUE LEVEL (external facing)
    # Value level's precision depends on volatility level
    expected_precision, effective_precision = predict_precision_value_level(
        attributes, edges, node_idx
    )

    # Value level's mean
    expected_mean = predict_mean_value_level(attributes, edges, node_idx)

    attributes[node_idx]["expected_mean"] = expected_mean

    # Input/leaf override: an observed volatile node has no value children, so it
    # does not undergo a Gaussian random walk between observations. Skip the
    # tonic-volatility contribution at the value level and use the prior precision
    # directly, mirroring the continuous-node treatment in
    # :func:`continuous_node_prediction`.
    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]
        attributes[node_idx]["temp"]["effective_precision"] = jnp.zeros_like(
            effective_precision
        )
    else:
        attributes[node_idx]["expected_precision"] = expected_precision
        attributes[node_idx]["temp"]["effective_precision"] = effective_precision

    return attributes
