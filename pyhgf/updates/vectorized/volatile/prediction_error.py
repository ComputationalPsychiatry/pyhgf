# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction error and volatility posterior for volatile node layers.

This module mirrors :mod:`pyhgf.updates.prediction_error.volatile` for vectorized
layers: it provides separate value and volatility prediction-error functions, per-
update-type volatility posterior functions, and a combined driver that calls them in
the correct order.
"""

import jax.numpy as jnp

from pyhgf.math import lambert_w0
from pyhgf.typing import LayerParams, LayerState

# ---------------------------------------------------------------------------
# 1.  Prediction errors
# ---------------------------------------------------------------------------


def vectorized_layer_value_prediction_error(
    layer: LayerState,
    n_parents: int,
) -> LayerState:
    """Compute the value prediction error for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.volatile.volatile_node_value_prediction_error`.

    Parameters
    ----------
    layer :
        Current layer with ``mean`` and ``expected_mean`` set.
    n_parents :
        Number of value parents for this layer (for normalization).

    Returns
    -------
    LayerState
        Updated layer state with ``value_prediction_error`` set.
    """
    raw_pe = layer.mean - layer.expected_mean
    value_pe = raw_pe / n_parents

    return layer._replace(value_prediction_error=value_pe)


def vectorized_layer_volatility_prediction_error(
    layer: LayerState,
) -> LayerState:
    """Compute the volatility prediction error for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.volatile.volatile_node_volatility_prediction_error`.

    Note that we are not dividing by the number of parents here, since volatile nodes
    only have one implied parent.

    Parameters
    ----------
    layer :
        Current layer.  Must already carry an up-to-date
        ``value_prediction_error`` (set by
        :func:`vectorized_layer_value_prediction_error`).

    Returns
    -------
    LayerState
        Updated layer state with ``volatility_prediction_error`` set.
    """
    volatility_pe = (
        (layer.expected_precision / layer.precision)
        + layer.expected_precision * ((layer.mean - layer.expected_mean) ** 2)
        - 1.0
    )

    return layer._replace(volatility_prediction_error=volatility_pe)


# ---------------------------------------------------------------------------
# 2.  Volatility-level posterior updates (one per update type)
# ---------------------------------------------------------------------------


def vectorized_layer_volatility_posterior_standard(
    layer: LayerState,
    params: LayerParams,
) -> LayerState:
    """Update the volatility level using the standard ordering.

    This is the vectorized equivalent of the standard volatility-level posterior
    update that first updates precision, then uses the updated precision to
    compute the mean update.

    Parameters
    ----------
    layer :
        Current layer state with ``volatility_prediction_error`` set.
    params :
        Layer parameters (provides ``volatility_coupling``).

    Returns
    -------
    LayerState
        Updated layer state with ``precision_vol`` and ``mean_vol`` set.

    See Also
    --------
    :mod:`pyhgf.updates.posterior.volatile.posterior_update_volatility_level`
    """
    volatility_pe = layer.volatility_prediction_error
    vol_coupling = params.volatility_coupling
    eff_prec = layer.effective_precision

    # Precision first
    precision_vol_contrib = (
        0.5 * ((vol_coupling * eff_prec) ** 2)
        + ((vol_coupling * eff_prec) ** 2) * volatility_pe
        - 0.5 * (vol_coupling**2) * eff_prec * volatility_pe
    )
    posterior_precision_vol = jnp.clip(
        layer.expected_precision_vol + precision_vol_contrib, a_max=1e8
    )

    # Mean using updated precision
    precision_weighted_pe_vol = (vol_coupling * eff_prec * volatility_pe) / (
        2.0 * posterior_precision_vol
    )
    posterior_mean_vol = layer.expected_mean_vol + precision_weighted_pe_vol

    return layer._replace(
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
    )


def vectorized_layer_volatility_posterior_ehgf(
    layer: LayerState,
    params: LayerParams,
) -> LayerState:
    """EHGF volatility-level posterior update (mean first, then precision).

    The eHGF update differs from the standard update in that it updates the
    **mean first** using the expected precision as an approximation, and then
    updates the precision.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.volatile_node_posterior_update_ehgf.volatile_node_posterior_update_ehgf`.

    Parameters
    ----------
    layer :
        Current layer state with ``volatility_prediction_error`` set.
    params :
        Layer parameters (provides ``volatility_coupling``).

    Returns
    -------
    LayerState
        Updated layer state with ``precision_vol`` and ``mean_vol`` set.
    """
    volatility_pe = layer.volatility_prediction_error
    vol_coupling = params.volatility_coupling
    eff_prec = layer.effective_precision

    # Mean first using expected_precision_vol as approximation
    precision_weighted_pe_vol = (vol_coupling * eff_prec * volatility_pe) / (
        2.0 * layer.expected_precision_vol
    )
    posterior_mean_vol = layer.expected_mean_vol + precision_weighted_pe_vol

    # Then precision
    precision_vol_contrib = (
        0.5 * ((vol_coupling * eff_prec) ** 2)
        + ((vol_coupling * eff_prec) ** 2) * volatility_pe
        - 0.5 * (vol_coupling**2) * eff_prec * volatility_pe
    )
    posterior_precision_vol = jnp.clip(
        layer.expected_precision_vol + precision_vol_contrib, a_max=1e8
    )

    return layer._replace(
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
    )


def vectorized_layer_volatility_posterior_unbounded(
    layer: LayerState,
    params: LayerParams,
    time_step: float,
) -> LayerState:
    """Unbounded volatility-level posterior update (Lambert W₀ dual-quadratic).

    Implements the uhgf update: two quadratic expansions blended via a
    variational energy-based softmax, with Gaussian mixture moment matching
    for the final posterior precision.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.volatile_node_posterior_update_unbounded.volatile_node_posterior_update_unbounded`.

    Parameters
    ----------
    layer :
        Current layer state with ``volatility_prediction_error`` set.
    params :
        Layer parameters (provides ``volatility_coupling`` and
        ``tonic_volatility``).
    time_step :
        Current time step (needed to reconstruct the pre-prediction variance).

    Returns
    -------
    LayerState
        Updated layer state with ``precision_vol`` and ``mean_vol`` set.
    """
    ka = params.volatility_coupling
    om = params.tonic_volatility

    # Reconstruct al_aux = 1/pi_prev_jm1
    predicted_volatility = time_step * jnp.exp(
        jnp.clip(om + ka * layer.expected_mean_vol, a_min=-80.0, a_max=80.0)
    )
    al_aux = jnp.maximum(1.0 / layer.expected_precision - predicted_volatility, 1e-128)
    be_aux = (1.0 / layer.precision) + (layer.mean - layer.expected_mean) ** 2

    muhat_j = layer.expected_mean_vol
    pihat_j = layer.expected_precision_vol

    # Canonical exponent at prediction: y = log(t_k) + ka*muhat_j + om
    gamma_c = jnp.log(time_step) + ka * muhat_j + om

    # Recompute v and w using muhat_j
    v_jm1 = jnp.exp(jnp.clip(gamma_c, a_min=-80.0, a_max=80.0))
    w_jm1 = v_jm1 / (al_aux + v_jm1)
    da_jm1 = be_aux / (al_aux + v_jm1) - 1.0

    # ------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ------------------------------------------------------------------
    pi1 = pihat_j + 0.5 * ka**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = muhat_j + (ka * w_jm1 / (2.0 * pi1)) * da_jm1

    # ------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W₀ approximate mode
    # ------------------------------------------------------------------
    pihat_y = pihat_j / ka**2
    W_arg = (be_aux / (2.0 * pihat_y)) * jnp.exp(
        jnp.clip(0.5 / pihat_y - gamma_c, a_min=-80.0, a_max=80.0)
    )
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - jnp.log(time_step) - om) / ka

    s2 = time_step * jnp.exp(jnp.clip(ka * x_star + om, a_min=-80.0, a_max=80.0))
    w2 = s2 / (al_aux + s2)
    da2 = be_aux / (al_aux + s2) - 1.0

    pi2_full = pihat_j + 0.5 * ka**2 * w2 * (w2 + (2.0 * w2 - 1.0) * da2)
    pi2 = jnp.where(
        pi2_full <= 0.0,
        pihat_j + 0.5 * ka**2 * w2 * (1.0 - w2),
        pi2_full,
    )
    mu2 = x_star + (0.5 * ka * w2 * da2 - pihat_j * (x_star - muhat_j)) / pi2

    # ------------------------------------------------------------------
    # Variational energy-based softmax blend
    # ------------------------------------------------------------------
    ey1 = time_step * jnp.exp(jnp.clip(ka * mu1 + om, a_min=-80.0, a_max=80.0))
    I1 = (
        -0.5 * jnp.log(al_aux + ey1)
        - 0.5 * be_aux / (al_aux + ey1)
        - 0.5 * pihat_j * (mu1 - muhat_j) ** 2
    )

    ey2 = time_step * jnp.exp(jnp.clip(ka * mu2 + om, a_min=-80.0, a_max=80.0))
    I2 = (
        -0.5 * jnp.log(al_aux + ey2)
        - 0.5 * be_aux / (al_aux + ey2)
        - 0.5 * pihat_j * (mu2 - muhat_j) ** 2
    )

    b = 1.0 / (1.0 + jnp.exp(I1 - I2))  # sigmoid(I2 - I1)

    # ------------------------------------------------------------------
    # Gaussian mixture moment matching
    # ------------------------------------------------------------------
    posterior_mean_vol = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) ** 2
    posterior_precision_vol = 1.0 / sig2

    return layer._replace(
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
    )


# ---------------------------------------------------------------------------
# 3.  Combined prediction-error + volatility-posterior driver
# ---------------------------------------------------------------------------


def vectorized_layer_prediction_error(
    layer: LayerState,
    n_parents: int,
    params: LayerParams,
    update_type: str = "eHGF",
    time_step: float = 1.0,
    has_volatility_parent: bool = True,
) -> LayerState:
    """Compute prediction errors and apply the volatility-level posterior update.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.volatile.volatile_node_prediction_error`.
    It first computes value and volatility prediction errors, then dispatches to
    the appropriate volatility-level posterior update depending on *update_type*.

    Parameters
    ----------
    layer :
        Current layer with ``mean`` and ``expected_mean`` set.
    n_parents :
        Number of value parents for this layer (for normalization).
    params :
        Layer parameters (needed by all volatility posterior updates).
    update_type :
        One of ``"eHGF"`` (default), ``"standard"``, or ``"unbounded"``.
    time_step :
        Current time step.  Only required when ``update_type="unbounded"``.
    has_volatility_parent :
        If True (default), compute the volatility prediction error and apply
        the volatility-level posterior update (mean_vol, precision_vol).
        If False, only the value prediction error is computed and the
        volatility level is left unchanged.

    Returns
    -------
    LayerState
        Updated layer state with prediction errors and volatility posterior.
    """
    # 1. Value prediction error (always computed)
    layer = vectorized_layer_value_prediction_error(layer, n_parents)

    if not has_volatility_parent:
        return layer

    # 2. Volatility prediction error and posterior update
    layer = vectorized_layer_volatility_prediction_error(layer)

    if update_type == "eHGF":
        layer = vectorized_layer_volatility_posterior_ehgf(layer, params)
    elif update_type == "standard":
        layer = vectorized_layer_volatility_posterior_standard(layer, params)
    elif update_type == "unbounded":
        layer = vectorized_layer_volatility_posterior_unbounded(
            layer, params, time_step
        )

    return layer
