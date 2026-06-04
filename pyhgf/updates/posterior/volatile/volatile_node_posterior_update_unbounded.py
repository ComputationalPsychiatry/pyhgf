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
    t_k = attributes[-1]["time_step"]

    al_aux = jnp.maximum(
        attributes[node_idx]["temp"]["current_variance"], 1e-128
    )  # 1/pi_prev_jm1
    be_aux = (1.0 / attributes[node_idx]["precision"]) + (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    ) ** 2

    muhat_j = attributes[node_idx]["expected_mean_vol"]
    pihat_j = attributes[node_idx]["expected_precision_vol"]
    ka = volatility_coupling
    om = attributes[node_idx]["tonic_volatility"]

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
    log_t_k = jnp.log(t_k)
    log_al_aux = jnp.log(al_aux)

    # Canonical exponent at the prediction: γ = log(t_k) + ka*muhat_j + om
    gamma_c = log_t_k + ka * muhat_j + om

    # w_jm1 = 1/(1 + al_aux/exp(γ)) = sigmoid(γ − log α).
    w_jm1 = sigmoid(gamma_c - log_al_aux)

    # Volatility prediction error: da_jm1 = pihat_jm1 * be_aux - 1, with
    # pihat_jm1 = expected_precision (set in the prediction step at mu_prev_j).
    # Matches MATLAB/Julia, which pass da_jm1 in — not recomputed at muhat_j.
    da_jm1 = attributes[node_idx]["expected_precision"] * be_aux - 1.0

    # ----------------------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ----------------------------------------------------------------------------------
    pi1 = pihat_j + 0.5 * ka**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = muhat_j + (ka * w_jm1 / (2.0 * pi1)) * da_jm1

    # ----------------------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W_0 approximate mode
    # ----------------------------------------------------------------------------------
    pihat_y = pihat_j / ka**2

    # Compute W_arg in log-space and cap at log(float_max) — matches MATLAB's
    # "W_arg = exp(min(log_W_arg, log(realmax)))".
    log_W_arg = jnp.log(be_aux) - jnp.log(2.0 * pihat_y) + 0.5 / pihat_y - gamma_c
    log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_W_arg)).max)
    W_arg = jnp.exp(jnp.minimum(log_W_arg, log_float_max))
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - log_t_k - om) / ka

    # Log-space s2/w2/da2 — equivalent to the direct
    #   s2 = t_k * exp(ka*x_star + om); w2 = 1/(1 + al_aux/s2);
    #   da2 = be_aux/(al_aux + s2) - 1
    # but without materialising ``s2 = inf`` (which injects 0·∞ NaN gradients).
    log_s2 = log_t_k + ka * x_star + om
    log_denom_s = jnp.logaddexp(log_al_aux, log_s2)  # = log(al_aux + s2)
    w2 = sigmoid(log_s2 - log_al_aux)
    da2 = be_aux * jnp.exp(-log_denom_s) - 1.0

    pi2_full = pihat_j + 0.5 * ka**2 * w2 * (w2 + (2.0 * w2 - 1.0) * da2)
    # Guard against negative precision (Matlab fallback: use w2*(1-w2) form)
    pi2_safe = jnp.where(
        pi2_full <= 0.0,
        pihat_j + 0.5 * ka**2 * w2 * (1.0 - w2),
        pi2_full,
    )
    mu2_safe = x_star + (0.5 * ka * w2 * da2 - pihat_j * (x_star - muhat_j)) / pi2_safe

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
    # The direct ``ey = t_k * exp(ka*mu + om)`` materialises ``inf`` for large
    # ``ka*mu + om`` and injects 0·∞ NaNs in the backward pass; ``logaddexp`` and
    # ``exp(-positive)`` stay bounded both forward and backward.
    # ----------------------------------------------------------------------------------
    log_ey1 = log_t_k + ka * mu1 + om
    log_denom_1 = jnp.logaddexp(log_al_aux, log_ey1)  # = log(al_aux + ey1)
    I1 = (
        -0.5 * log_denom_1
        - 0.5 * be_aux * jnp.exp(-log_denom_1)
        - 0.5 * pihat_j * (mu1 - muhat_j) ** 2
    )

    log_ey2 = log_t_k + ka * mu2 + om
    log_denom_2 = jnp.logaddexp(log_al_aux, log_ey2)
    I2 = (
        -0.5 * log_denom_2
        - 0.5 * be_aux * jnp.exp(-log_denom_2)
        - 0.5 * pihat_j * (mu2 - muhat_j) ** 2
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
