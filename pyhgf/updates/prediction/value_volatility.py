from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from pyhgf.typing import Edges

@partial(jit, static_argnames=("node_idx",))
def predict_mean_volatility_level(
    attributes: dict,
    node_idx: int,
) -> Array:
    """Predict the mean of the implicit volatility level.
    
    This is analogous to predicting a volatility parent node.
    """
    time_step = attributes[-1]["time_step"]

    # Get volatility level parameters
    autoconnection = attributes[node_idx]["autoconnection_strength_vol"]
    mean_vol = attributes[node_idx]["mean_vol"]
    drift_vol = attributes[node_idx]["tonic_drift_vol"]

    # Predict volatility level mean (simple random walk)
    expected_mean_vol = (autoconnection * mean_vol) + (time_step * drift_vol)

    return expected_mean_vol


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
    driftrate = attributes[node_idx]["tonic_drift"]

    # Look at the (optional) value parents for this node
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
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

            driftrate += psi * parent_value

    # The new expected mean from the previous value
    expected_mean = (
        attributes[node_idx]["autoconnection_strength"] * attributes[node_idx]["mean"]
    ) + (time_step * driftrate)

    return expected_mean


@partial(jit, static_argnames=("node_idx",))
def predict_precision_value_level(
    attributes: dict,
    node_idx: int,
) -> tuple[Array, Array]:
    """Predict the precision of the value level using the implicit volatility level.
    
    The volatility level's mean modulates the value level's precision.
    """
    time_step = attributes[-1]["time_step"]

    # Get value level parameters
    precision = attributes[node_idx]["precision"]
    tonic_volatility = attributes[node_idx]["tonic_volatility"]

    # Get volatility level's expected mean (already computed)
    expected_mean_vol = attributes[node_idx]["expected_mean_vol"]

    # Get internal coupling strength
    volatility_coupling_internal = attributes[node_idx]["volatility_coupling_internal"]

    # Total volatility = tonic + contribution from implicit volatility parent
    total_volatility = tonic_volatility + (volatility_coupling_internal * expected_mean_vol)

    # Compute predicted volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Expected precision
    expected_precision = 1 / ((1 / precision) + predicted_volatility)

    # Effective precision
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, effective_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def value_volatility_node_prediction(
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
    expected_mean_vol = predict_mean_volatility_level(attributes, node_idx)
    expected_precision_vol, effective_precision_vol = predict_precision_volatility_level(
        attributes, node_idx
    )

    attributes[node_idx]["expected_mean_vol"] = expected_mean_vol
    attributes[node_idx]["expected_precision_vol"] = expected_precision_vol
    attributes[node_idx]["temp"]["effective_precision_vol"] = effective_precision_vol

    # 2. PREDICT VALUE LEVEL (external facing)
    # Value level's precision depends on volatility level
    expected_precision, effective_precision = predict_precision_value_level(
        attributes, node_idx
    )

    # Value level's mean
    expected_mean = predict_mean_value_level(attributes, edges, node_idx)

    attributes[node_idx]["expected_mean"] = expected_mean
    attributes[node_idx]["expected_precision"] = expected_precision
    attributes[node_idx]["temp"]["effective_precision"] = effective_precision

    return attributes