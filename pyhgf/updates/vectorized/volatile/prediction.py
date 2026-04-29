# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction update for volatile node layers."""

from typing import Callable

import jax.numpy as jnp

from pyhgf.typing import LayerParams, LayerState


def vectorized_layer_prediction(
    child_state: LayerState,
    parent_state: LayerState,
    weights: jnp.ndarray,
    params: LayerParams,
    time_step: float,
    coupling_fn: Callable = jnp.tanh,
    parent_has_constant: bool = False,
    has_volatility_parent: bool = True,
    is_input_layer: bool = False,
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
        Weight matrix connecting child to parent, shape
        ``(n_children, n_parents)`` or ``(n_children, n_parents + 1)``
        when the parent layer includes a constant input node.
    params :
        Layer parameters for the child layer.
    time_step :
        Time step for the prediction.
    coupling_fn :
        Coupling function applied to parent means (default: tanh).
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0)
        appended to its activations.  The last column of *weights*
        carries the bias connections.
    has_volatility_parent :
        If True (default), the layer has an implied internal volatility parent
        whose state (mean_vol, precision_vol) is predicted and updated.
        If False, the volatility level is frozen: mean_vol and precision_vol
        are not propagated forward, and only tonic_volatility drives the
        expected precision for the value level.
    is_input_layer :
        If True, the layer is treated as an observed input/leaf — it does not
        undergo a Gaussian random walk between observations. The
        ``tonic_volatility`` contribution to the value-level expected precision
        is skipped and ``expected_precision`` is set to the prior precision,
        mirroring the continuous-node treatment in
        :func:`pyhgf.updates.prediction.continuous.continuous_node_prediction`.

    Returns
    -------
    LayerState
        Updated child layer state with expected values filled in.
    """
    # 1. VOLATILITY LEVEL PREDICTION (internal) ----------------------------------------
    # ----------------------------------------------------------------------------------
    if has_volatility_parent:
        # Expected mean for volatility level (autoconnection = 1.0)
        expected_mean_vol = child_state.mean_vol

        # Predicted volatility for volatility level
        predicted_volatility_vol = time_step * jnp.exp(params.tonic_volatility_vol)
        predicted_volatility_vol = jnp.where(
            predicted_volatility_vol > 1e-128, predicted_volatility_vol, jnp.nan
        )

        # Expected precision for volatility level
        expected_precision_vol = 1.0 / (
            1.0 / child_state.precision_vol + predicted_volatility_vol
        )

        # Effective precision for volatility level
        effective_precision_vol = predicted_volatility_vol * expected_precision_vol
    else:
        # Volatility level is frozen — pass through current values unchanged
        expected_mean_vol = child_state.mean_vol
        expected_precision_vol = child_state.precision_vol
        effective_precision_vol = child_state.effective_precision_vol

    # 2. VALUE LEVEL PREDICTION (external) ---------------------------------------------
    # ----------------------------------------------------------------------------------

    # Mean prediction via matrix multiply
    # weights shape: (n_children, n_parents) or (n_children, n_parents + 1)
    # parent_state.expected_mean shape: (n_parents,)
    parent_mean = parent_state.expected_mean
    if parent_has_constant:
        # Append constant 1.0 for bias node before applying coupling_fn
        parent_mean = jnp.concatenate([parent_mean, jnp.ones(1)])
    coupled_parents = coupling_fn(parent_mean)
    drift = jnp.matmul(weights, coupled_parents)

    # Expected mean for value level
    # Note: autoconnection_strength = 0 for i.i.d. classification
    # (the previous observation should not bias the next prediction)
    expected_mean = time_step * drift

    if has_volatility_parent:
        # Total volatility includes contribution from internal volatility level
        total_volatility = (
            params.tonic_volatility + params.volatility_coupling * expected_mean_vol
        )
    else:
        # Only tonic volatility — no mean_vol contribution
        total_volatility = params.tonic_volatility

    # Predicted volatility for value level
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Expected precision for value level
    expected_precision = 1.0 / (1.0 / child_state.precision + predicted_volatility)

    # Effective precision for value level
    effective_precision = predicted_volatility * expected_precision

    # Input/leaf override: an observed layer with no value children does not
    # undergo a Gaussian random walk between observations, so the
    # tonic-volatility contribution to the value-level expected precision is
    # dropped (matches the continuous-node treatment).
    if is_input_layer:
        expected_precision = child_state.precision
        effective_precision = jnp.zeros_like(effective_precision)

    return child_state._replace(
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        effective_precision=effective_precision,
        expected_mean_vol=expected_mean_vol,
        expected_precision_vol=expected_precision_vol,
        effective_precision_vol=effective_precision_vol,
    )
