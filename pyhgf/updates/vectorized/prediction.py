# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction update for volatile node layers."""

from typing import Callable

import jax.numpy as jnp

from pyhgf.model.vectorized_types import LayerParams, LayerState


def vectorized_layer_prediction(
    child_state: LayerState,
    parent_state: LayerState,
    weights: jnp.ndarray,
    params: LayerParams,
    time_step: float,
    coupling_fn: Callable = jnp.tanh,
    add_bias: bool = False,
) -> LayerState:
    """Predict expected mean/precision for all nodes in child layer (volatile node).

    This implements the full volatile node prediction with both value level
    and volatility level predictions.

    Parameters
    ----------
    child_state :
        Current state of the child layer (being predicted).
    parent_state :
        Current state of the parent layer (predictor).
    weights :
        Weight matrix connecting child to parent, shape (n_children, n_parents).
    params :
        Layer parameters for the child layer.
    time_step :
        Time step for the prediction.
    coupling_fn :
        Coupling function applied to parent means (default: tanh).

    Returns
    -------
    LayerState
        Updated child layer state with expected values filled in.
    """
    # 1. VOLATILITY LEVEL PREDICTION (internal)
    # Expected mean for volatility level (autoconnection = 1.0)
    expected_mean_vol = child_state.mean_vol

    # Predicted volatility for volatility level
    predicted_volatility_vol = time_step * jnp.exp(
        jnp.clip(params.tonic_volatility_vol, a_min=-80.0, a_max=80.0)
    )
    predicted_volatility_vol = jnp.maximum(predicted_volatility_vol, 1e-128)

    # Expected precision for volatility level
    expected_precision_vol = 1.0 / (
        1.0 / child_state.precision_vol + predicted_volatility_vol
    )

    # Effective precision for volatility level
    effective_precision_vol = predicted_volatility_vol * expected_precision_vol

    # 2. VALUE LEVEL PREDICTION (external)
    # Total volatility includes contribution from internal volatility level
    total_volatility = (
        params.tonic_volatility + params.volatility_coupling * expected_mean_vol
    )

    # Predicted volatility for value level
    predicted_volatility = time_step * jnp.exp(
        jnp.clip(total_volatility, a_min=-80.0, a_max=80.0)
    )
    predicted_volatility = jnp.maximum(predicted_volatility, 1e-128)

    # Expected precision for value level
    expected_precision = 1.0 / (1.0 / child_state.precision + predicted_volatility)

    # Effective precision for value level
    effective_precision = predicted_volatility * expected_precision

    # Mean prediction via matrix multiply
    # weights shape: (n_children, n_parents)
    # parent_state.expected_mean shape: (n_parents,)
    coupled_parents = coupling_fn(parent_state.expected_mean)
    drift = jnp.matmul(weights, coupled_parents)

    # Expected mean for value level
    # Note: autoconnection_strength = 0 for i.i.d. classification
    # (the previous observation should not bias the next prediction)
    if add_bias:
        expected_mean = time_step * drift + params.bias
    else:
        expected_mean = time_step * drift

    return child_state._replace(
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        effective_precision=effective_precision,
        expected_mean_vol=expected_mean_vol,
        expected_precision_vol=expected_precision_vol,
        effective_precision_vol=effective_precision_vol,
    )


def vectorized_input_layer_prediction(
    state: LayerState,
    params: LayerParams,
    time_step: float,
) -> LayerState:
    """Predict expected mean/precision for the input layer (no parents).

    This is a simplified prediction for the top layer that has no value parents.

    Parameters
    ----------
    state :
        Current state of the input layer.
    params :
        Layer parameters.
    time_step :
        Time step for the prediction.

    Returns
    -------
    LayerState
        Updated layer state with expected values.
    """
    # 1. VOLATILITY LEVEL PREDICTION (internal)
    expected_mean_vol = state.mean_vol

    predicted_volatility_vol = time_step * jnp.exp(
        jnp.clip(params.tonic_volatility_vol, a_min=-80.0, a_max=80.0)
    )
    predicted_volatility_vol = jnp.maximum(predicted_volatility_vol, 1e-128)

    expected_precision_vol = 1.0 / (
        1.0 / state.precision_vol + predicted_volatility_vol
    )
    effective_precision_vol = predicted_volatility_vol * expected_precision_vol

    # 2. VALUE LEVEL PREDICTION (external)
    total_volatility = (
        params.tonic_volatility + params.volatility_coupling * expected_mean_vol
    )
    predicted_volatility = time_step * jnp.exp(
        jnp.clip(total_volatility, a_min=-80.0, a_max=80.0)
    )
    predicted_volatility = jnp.maximum(predicted_volatility, 1e-128)

    expected_precision = 1.0 / (1.0 / state.precision + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision

    # No parents, so expected mean is just the current mean (no drift)
    expected_mean = state.mean

    return state._replace(
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        effective_precision=effective_precision,
        expected_mean_vol=expected_mean_vol,
        expected_precision_vol=expected_precision_vol,
        effective_precision_vol=effective_precision_vol,
    )
