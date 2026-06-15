# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction error and volatility posterior for volatile node layers.

This module mirrors :mod:`pyhgf.updates.prediction_error.volatile` for vectorized
layers: it provides separate value and volatility prediction-error functions, per-
update-type volatility posterior functions, and a combined driver that calls them in the
correct order.
"""

import dataclasses

import jax.numpy as jnp
from jax.nn import sigmoid

from pyhgf.math import lambert_w0
from pyhgf.typing.vectorised import LayerParams, LayerState

# ---------------------------------------------------------------------------
# 1.  Prediction errors
# ---------------------------------------------------------------------------


def vectorized_layer_value_prediction_error(
    layer: LayerState,
) -> LayerState:
    """Compute the value prediction error for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.volatile.volatile_node_value_prediction_error`.

    Parent-count normalisation is applied in the prediction step instead
    (the drift is divided by ``n_parents`` before setting ``expected_mean``),
    so the PE carries the full residual without further scaling.

    Parameters
    ----------
    layer :
        Current layer with ``mean`` and ``expected_mean`` set.

    Returns
    -------
    LayerState
        Updated layer state with ``value_prediction_error`` set.
    """
    value_pe = layer.mean - layer.expected_mean

    return dataclasses.replace(layer, value_prediction_error=value_pe)


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

    return dataclasses.replace(layer, volatility_prediction_error=volatility_pe)


# ---------------------------------------------------------------------------
# 2.  Volatility-level posterior updates (one per update type)
# ---------------------------------------------------------------------------


def vectorized_layer_volatility_posterior_standard(
    layer: LayerState,
    params: LayerParams,
    max_posterior_precision: float = 1e10,
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
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision.
        Default ``1e10``.

    Returns
    -------
    LayerState
        Updated layer state with ``precision_vol`` and ``mean_vol`` set.

    See Also
    --------
    :mod:`pyhgf.updates.posterior.volatile.posterior_update_volatility_level`
    """
    volatility_prediction_error = layer.volatility_prediction_error
    volatility_coupling = params.volatility_coupling
    effective_precision = layer.effective_precision

    # Precision first
    precision_vol_contrib = (
        0.5 * ((volatility_coupling * effective_precision) ** 2)
        + ((volatility_coupling * effective_precision) ** 2)
        * volatility_prediction_error
        - 0.5
        * (volatility_coupling**2)
        * effective_precision
        * volatility_prediction_error
    )
    posterior_precision_vol = jnp.clip(
        layer.expected_precision_vol + precision_vol_contrib,
        a_max=max_posterior_precision,
    )

    # Mean using updated precision
    precision_weighted_pe_vol = (
        volatility_coupling * effective_precision * volatility_prediction_error
    ) / (2.0 * posterior_precision_vol)
    posterior_mean_vol = layer.expected_mean_vol + precision_weighted_pe_vol

    return dataclasses.replace(
        layer,
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
    )


def vectorized_layer_volatility_posterior_ehgf(
    layer: LayerState,
    params: LayerParams,
    time_step: float,
    max_posterior_precision: float = 1e10,
) -> LayerState:
    """EHGF volatility-level posterior update (mean first, then safe precision).

    The eHGF update differs from the standard update in that it updates the
    **mean first** using the expected precision as an approximation, and then
    applies the safe precision update: the effective precision is recomputed from
    the just-updated posterior mean and the increment is floored at zero.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.volatile_node_posterior_update_ehgf.volatile_node_posterior_update_ehgf`.

    Parameters
    ----------
    layer :
        Current layer state with ``volatility_prediction_error`` set.
    params :
        Layer parameters (provides ``volatility_coupling`` and ``tonic_volatility``).
    time_step :
        Current time step (needed to recompute the effective precision and to
        reconstruct the pre-prediction variance).
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision.
        Default ``1e10``.

    Returns
    -------
    LayerState
        Updated layer state with ``precision_vol`` and ``mean_vol`` set.
    """
    volatility_coupling = params.volatility_coupling
    tonic_volatility = params.tonic_volatility

    # Mean first using expected_precision_vol as approximation
    precision_weighted_pe_vol = (
        volatility_coupling
        * layer.effective_precision
        * layer.volatility_prediction_error
    ) / (2.0 * layer.expected_precision_vol)
    posterior_mean_vol = layer.expected_mean_vol + precision_weighted_pe_vol

    # Safe precision update: recompute the effective precision from the posterior
    # mean (``posterior_mean_vol``) and floor the increment at zero. Mirrors
    # ``_ehgf_volatility_precision_increment`` in the node-based backend.
    #
    # Reconstruct the value level's previous variance (= 1/precision_prev) exactly:
    #   1/conditional_expected_precision = 1/precision_prev + predicted_volatility,
    # so subtracting predicted_volatility cancels back to 1/precision_prev.
    predicted_volatility = time_step * jnp.exp(
        tonic_volatility
        + volatility_coupling * layer.expected_mean_vol
        + volatility_coupling**2 / (2.0 * layer.expected_precision_vol)
    )
    previous_variance = jnp.maximum(
        1.0 / layer.conditional_expected_precision - predicted_volatility, 1e-128
    )

    # Re-predict the value level's volatility and precision from the posterior mean.
    repredicted_volatility = time_step * jnp.exp(
        volatility_coupling * posterior_mean_vol + tonic_volatility
    )
    expected_precision = 1.0 / (previous_variance + repredicted_volatility)
    effective_precision = repredicted_volatility * expected_precision
    volatility_error_weight = (
        repredicted_volatility - previous_variance
    ) * expected_precision
    volatility_prediction_error = (
        (1.0 / layer.precision) + (layer.mean - layer.expected_mean) ** 2
    ) * expected_precision - 1.0

    precision_vol_contrib = jnp.maximum(
        0.0,
        0.5
        * (volatility_coupling**2)
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error),
    )
    posterior_precision_vol = jnp.clip(
        layer.expected_precision_vol + precision_vol_contrib,
        a_max=max_posterior_precision,
    )

    return dataclasses.replace(
        layer,
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
    )


def vectorized_layer_volatility_posterior_unbounded(
    layer: LayerState,
    params: LayerParams,
    time_step: float,
    max_posterior_precision: float = 1e10,
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
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision.
        Default ``1e10``.

    Returns
    -------
    LayerState
        Updated layer state with ``precision_vol`` and ``mean_vol`` set.
    """
    volatility_coupling = params.volatility_coupling
    tonic_volatility = params.tonic_volatility

    # Reconstruct previous_variance = 1/pi_prev_jm1 exactly from the *conditional* predicted
    # precision. The prediction step sets
    #     1/conditional_expected_precision = 1/precision_prev + predicted_volatility,
    # where predicted_volatility carries the full log-volatility exponent, including
    # the MGF correction volatility_coupling^2 / (2 * pihat_vol). Subtracting the same
    # predicted_volatility cancels back to 1/precision_prev. Using the marginal
    # expected_precision here would leave the value-coupling variance behind, and
    # omitting the MGF term would leave that behind — both were latent errors that
    # only surfaced once value coupling shifted the operating point.
    predicted_volatility = time_step * jnp.exp(
        tonic_volatility
        + volatility_coupling * layer.expected_mean_vol
        + volatility_coupling**2 / (2.0 * layer.expected_precision_vol)
    )
    previous_variance = jnp.maximum(
        1.0 / layer.conditional_expected_precision - predicted_volatility, 1e-128
    )
    be_aux = (1.0 / layer.precision) + (layer.mean - layer.expected_mean) ** 2

    expected_mean_vol = layer.expected_mean_vol
    expected_precision_vol = layer.expected_precision_vol

    # Quantities that would otherwise pass through ``exp`` of a large number are
    # kept in log-space. ``v = exp(γ)`` is fine forward (saturating uses are
    # stable) but ``0 · ∞ = NaN`` poisons the backward pass — a single non-finite
    # gradient NaN-poisons the whole scan, which under NUTS forces step-size
    # collapse and blows up the leapfrog count per sample, and under gradient
    # learning kills the optimiser step. The ``sigmoid``/``logaddexp`` rewrites
    # match the direct forms for every finite input and stay gradient-safe at the
    # saturation limits. Mirrors ``continuous_node_posterior_update_unbounded``.
    log_time_step = jnp.log(time_step)
    log_previous_variance = jnp.log(previous_variance)

    # Canonical exponent at prediction: γ = log(time_step) + volatility_coupling*expected_mean_vol + tonic_volatility
    gamma_c = log_time_step + volatility_coupling * expected_mean_vol + tonic_volatility

    # w_jm1 = 1/(1 + previous_variance/exp(γ)) = sigmoid(γ − log α).
    w_jm1 = sigmoid(gamma_c - log_previous_variance)
    # Volatility prediction error da_jm1 = pihat * be_aux - 1, with pihat the
    # *marginal* predicted precision. Written explicitly rather than as
    # be_aux/(previous_variance+v_jm1) - 1: that identity only holds when previous_variance is the
    # reconstruction 1/expected_precision - v_jm1, which is no longer the case now
    # that previous_variance is the exact 1/precision_prev.
    da_jm1 = layer.expected_precision * be_aux - 1.0

    # ------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ------------------------------------------------------------------
    pi1 = expected_precision_vol + 0.5 * volatility_coupling**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = expected_mean_vol + (volatility_coupling * w_jm1 / (2.0 * pi1)) * da_jm1

    # ------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W₀ approximate mode
    # ------------------------------------------------------------------
    pihat_y = expected_precision_vol / volatility_coupling**2

    # Compute W_arg in log-space and cap at log(float_max) — matches MATLAB's
    # "W_arg = exp(min(log_W_arg, log(realmax)))".
    log_W_arg = jnp.log(be_aux) - jnp.log(2.0 * pihat_y) + 0.5 / pihat_y - gamma_c
    log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_W_arg)).max)
    W_arg = jnp.exp(jnp.minimum(log_W_arg, log_float_max))
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - log_time_step - tonic_volatility) / volatility_coupling

    # Log-space s2/w2/da2 — never materialise ``s2 = inf`` (0·∞ NaN gradients).
    log_s2 = log_time_step + volatility_coupling * x_star + tonic_volatility
    log_denom_s = jnp.logaddexp(
        log_previous_variance, log_s2
    )  # = log(previous_variance + s2)
    w2 = sigmoid(log_s2 - log_previous_variance)
    da2 = be_aux * jnp.exp(-log_denom_s) - 1.0

    pi2_full = expected_precision_vol + 0.5 * volatility_coupling**2 * w2 * (
        w2 + (2.0 * w2 - 1.0) * da2
    )
    pi2_safe = jnp.where(
        pi2_full <= 0.0,
        expected_precision_vol + 0.5 * volatility_coupling**2 * w2 * (1.0 - w2),
        pi2_full,
    )
    mu2_safe = (
        x_star
        + (
            0.5 * volatility_coupling * w2 * da2
            - expected_precision_vol * (x_star - expected_mean_vol)
        )
        / pi2_safe
    )

    # Fall back to Expansion 1 if Expansion 2 yields non-finite results —
    # matches MATLAB: "if ~isfinite(pi2) || ~isfinite(mu2), pi2 = pi1; mu2 = mu1".
    # Double-where masking: sanitise non-finite Expansion-2 results *before* the
    # outer ``where`` so its VJP never does ``0 * NaN``.
    exp2_finite = jnp.isfinite(pi2_safe) & jnp.isfinite(mu2_safe)
    pi2_safe_for_grad = jnp.where(exp2_finite, pi2_safe, 1.0)
    mu2_safe_for_grad = jnp.where(exp2_finite, mu2_safe, 0.0)
    pi2 = jnp.where(exp2_finite, pi2_safe_for_grad, pi1)
    mu2 = jnp.where(exp2_finite, mu2_safe_for_grad, mu1)

    # ------------------------------------------------------------------
    # Variational energy-based softmax blend (log-space form, gradient-safe).
    # ``ey = time_step * exp(volatility_coupling*mu + tonic_volatility)`` materialises ``inf`` for large exponents and
    # injects 0·∞ NaNs in the backward pass; ``logaddexp`` / ``exp(-positive)``
    # stay bounded both ways.
    # ------------------------------------------------------------------
    log_ey1 = log_time_step + volatility_coupling * mu1 + tonic_volatility
    log_denom_1 = jnp.logaddexp(
        log_previous_variance, log_ey1
    )  # = log(previous_variance + ey1)
    I1 = (
        -0.5 * log_denom_1
        - 0.5 * be_aux * jnp.exp(-log_denom_1)
        - 0.5 * expected_precision_vol * (mu1 - expected_mean_vol) ** 2
    )

    log_ey2 = log_time_step + volatility_coupling * mu2 + tonic_volatility
    log_denom_2 = jnp.logaddexp(log_previous_variance, log_ey2)
    I2 = (
        -0.5 * log_denom_2
        - 0.5 * be_aux * jnp.exp(-log_denom_2)
        - 0.5 * expected_precision_vol * (mu2 - expected_mean_vol) ** 2
    )

    # Stable sigmoid matches b = 1/(1 + exp(I1 - I2)) without NaN at ±∞.
    b = sigmoid(I2 - I1)

    # ------------------------------------------------------------------
    # Gaussian mixture moment matching
    # ------------------------------------------------------------------
    posterior_mean_vol = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) ** 2
    posterior_precision_vol = jnp.minimum(1.0 / sig2, max_posterior_precision)

    return dataclasses.replace(
        layer,
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
    )


# ---------------------------------------------------------------------------
# 3.  Combined prediction-error + volatility-posterior driver
# ---------------------------------------------------------------------------


def vectorized_layer_prediction_error(
    layer: LayerState,
    params: LayerParams,
    volatility_updates: str = "eHGF",
    time_step: float = 1.0,
    has_volatility_parent: bool = True,
    max_posterior_precision: float = 1e10,
) -> LayerState:
    """Compute prediction errors and apply the volatility-level posterior update.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.volatile.volatile_node_prediction_error`.
    It first computes value and volatility prediction errors, then dispatches to
    the appropriate volatility-level posterior update depending on *volatility_updates*.

    Parent-count normalisation is applied in the prediction step (drift divided by
    ``n_parents``), so the PE is the plain residual ``mean - expected_mean``.

    Parameters
    ----------
    layer :
        Current layer with ``mean`` and ``expected_mean`` set.
    params :
        Layer parameters (needed by all volatility posterior updates).
    volatility_updates :
        One of ``"eHGF"`` (default), ``"standard"``, or ``"unbounded"``.
    time_step :
        Current time step.  Only required when ``volatility_updates="unbounded"``.
    has_volatility_parent :
        If True (default), compute the volatility prediction error and apply
        the volatility-level posterior update (mean_vol, precision_vol).
        If False, only the value prediction error is computed and the
        volatility level is left unchanged.
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision.
        Default ``1e10``.

    Returns
    -------
    LayerState
        Updated layer state with prediction errors and volatility posterior.
    """
    # 1. Value prediction error (always computed)
    layer = vectorized_layer_value_prediction_error(layer)

    if not has_volatility_parent:
        return layer

    # 2. Volatility prediction error and posterior update
    layer = vectorized_layer_volatility_prediction_error(layer)

    if volatility_updates == "eHGF":
        layer = vectorized_layer_volatility_posterior_ehgf(
            layer, params, time_step, max_posterior_precision=max_posterior_precision
        )
    elif volatility_updates == "standard":
        layer = vectorized_layer_volatility_posterior_standard(
            layer, params, max_posterior_precision=max_posterior_precision
        )
    elif volatility_updates == "unbounded":
        layer = vectorized_layer_volatility_posterior_unbounded(
            layer,
            params,
            time_step,
            max_posterior_precision=max_posterior_precision,
        )

    return layer
