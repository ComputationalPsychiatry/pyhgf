# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid

from pyhgf.math import lambert_w0


@partial(jit, static_argnames=("node_idx", "max_posterior_precision"))
def volatile_node_posterior_update_unbounded(
    attributes: dict,
    node_idx: int,
    max_posterior_precision: float = 1e10,
) -> dict:
    """Update the volatility level using an unbounded quadratic approximation.

    Implements the uhgf update: two quadratic expansions are blended via a
    variational energy-based softmax weight, and the final posterior is the
    moment-matched Gaussian of the resulting mixture.

    Expansion 1 is centred at the prediction (prior mean).
    Expansion 2 is centred at the approximate posterior mode found via the Lambert
    W_0 function, which solves the mode equation exactly in the limit alpha -> 0.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile node.
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision write.
        Default ``1e10``.

    Returns
    -------
    dict
        Updated attributes with ``precision_vol`` and ``mean_vol`` set.
    """
    volatility_coupling = attributes[node_idx]["volatility_coupling_internal"]
    time_step = attributes[-1]["time_step"]

    previous_variance = jnp.maximum(
        attributes[node_idx]["temp"]["current_variance"], 1e-128
    )  # previous-step variance (= 1 / precision at the previous step)
    be_aux = (1.0 / attributes[node_idx]["precision"]) + (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    ) ** 2

    expected_mean_vol = attributes[node_idx]["expected_mean_vol"]
    expected_precision_vol = attributes[node_idx]["expected_precision_vol"]
    tonic_volatility = attributes[node_idx]["tonic_volatility"]

    # All quantities that would otherwise pass through ``exp`` of a potentially
    # large number are kept in log-space. Materialising ``v = exp(γ)`` is correct
    # in the forward pass (downstream saturating uses are stable), but corrupts
    # the backward pass: the local partial of a saturating expression is ``0``
    # while ``d v / d γ = exp(γ) = ∞``, and ``0 · ∞ = NaN``. A single non-finite
    # gradient anywhere in the scan turns the whole gradient into NaN, which
    # forces NUTS to reject the trajectory and shrink the step size — blowing up
    # the number of leapfrog evaluations per sample. The ``sigmoid``/``logaddexp``
    # rewrites below match the direct forms for every finite input and stay
    # gradient-safe at the saturation limits. Mirrors
    # ``continuous_node_posterior_update_unbounded``.
    log_time_step = jnp.log(time_step)
    log_previous_variance = jnp.log(previous_variance)

    # Canonical exponent at the prediction: γ = log(time_step) + volatility_coupling*expected_mean_vol + tonic_volatility
    gamma_c = log_time_step + volatility_coupling * expected_mean_vol + tonic_volatility

    # w_jm1 = 1/(1 + previous_variance/exp(γ)) = sigmoid(γ − log α).
    w_jm1 = sigmoid(gamma_c - log_previous_variance)

    # Volatility prediction error: da_jm1 = pihat_jm1 * be_aux - 1, with
    # pihat_jm1 = expected_precision (set in the prediction step at mu_prev_j).
    # Matches MATLAB/Julia, which pass da_jm1 in — not recomputed at expected_mean_vol.
    da_jm1 = attributes[node_idx]["expected_precision"] * be_aux - 1.0

    # ----------------------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ----------------------------------------------------------------------------------
    pi1 = expected_precision_vol + 0.5 * volatility_coupling**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = expected_mean_vol + (volatility_coupling * w_jm1 / (2.0 * pi1)) * da_jm1

    # ----------------------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W_0 approximate mode
    # ----------------------------------------------------------------------------------
    pihat_y = expected_precision_vol / volatility_coupling**2

    # Compute W_arg in log-space and cap at log(float_max) — matches MATLAB's
    # "W_arg = exp(min(log_W_arg, log(realmax)))".
    log_W_arg = jnp.log(be_aux) - jnp.log(2.0 * pihat_y) + 0.5 / pihat_y - gamma_c
    log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_W_arg)).max)
    W_arg = jnp.exp(jnp.minimum(log_W_arg, log_float_max))
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - log_time_step - tonic_volatility) / volatility_coupling

    # Log-space s2/w2/da2 — equivalent to the direct
    #   s2 = time_step * exp(volatility_coupling*x_star + tonic_volatility); w2 = 1/(1 + previous_variance/s2);
    #   da2 = be_aux/(previous_variance + s2) - 1
    # but without materialising ``s2 = inf`` (which injects 0·∞ NaN gradients).
    log_s2 = log_time_step + volatility_coupling * x_star + tonic_volatility
    log_denom_s = jnp.logaddexp(
        log_previous_variance, log_s2
    )  # = log(previous_variance + s2)
    w2 = sigmoid(log_s2 - log_previous_variance)
    da2 = be_aux * jnp.exp(-log_denom_s) - 1.0

    pi2_full = expected_precision_vol + 0.5 * volatility_coupling**2 * w2 * (
        w2 + (2.0 * w2 - 1.0) * da2
    )
    # Guard against negative precision (Matlab fallback: use w2*(1-w2) form)
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
    #
    # Double-where masking: replace any non-finite ``pi2_safe`` / ``mu2_safe``
    # with safe constants *before* they enter the outer ``where``. The bare form
    # ``jnp.where(c, pi2_safe, pi1)`` is correct forward but poisons the backward
    # pass — ``where``'s VJP routes a zero cotangent into the masked-out branch
    # and ``0 * NaN = NaN``.
    exp2_finite = jnp.isfinite(pi2_safe) & jnp.isfinite(mu2_safe)
    pi2_safe_for_grad = jnp.where(exp2_finite, pi2_safe, 1.0)
    mu2_safe_for_grad = jnp.where(exp2_finite, mu2_safe, 0.0)
    pi2 = jnp.where(exp2_finite, pi2_safe_for_grad, pi1)
    mu2 = jnp.where(exp2_finite, mu2_safe_for_grad, mu1)

    # ----------------------------------------------------------------------------------
    # Variational energy-based softmax blend (log-space form, gradient-safe).
    # The direct ``ey = time_step * exp(volatility_coupling*mu + tonic_volatility)`` materialises ``inf`` for large
    # ``volatility_coupling*mu + tonic_volatility`` and injects 0·∞ NaNs in the backward pass; ``logaddexp`` and
    # ``exp(-positive)`` stay bounded both forward and backward.
    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    # Gaussian mixture moment matching
    # ----------------------------------------------------------------------------------
    posterior_mean = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) ** 2
    posterior_precision = 1.0 / sig2

    attributes[node_idx]["precision_vol"] = jnp.minimum(
        posterior_precision, max_posterior_precision
    )
    attributes[node_idx]["mean_vol"] = posterior_mean

    return attributes
