# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid

from pyhgf.math import lambert_w0
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx", "max_posterior_precision"))
def continuous_node_posterior_update_unbounded(
    attributes: dict,
    node_idx: int,
    edges: Edges,
    max_posterior_precision: float = 1e10,
    **args,
) -> dict:
    """Update the posterior of a continuous node with unbounded quadratic approximation.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated.
    edges :
        The edges of the probabilistic nodes.
    max_posterior_precision :
        Upper bound applied to the posterior precision write. Default ``1e10``.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.
    """
    posterior_precision, posterior_mean = posterior_update_unbounded(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    attributes[node_idx]["precision"] = jnp.minimum(
        posterior_precision, max_posterior_precision
    )
    attributes[node_idx]["mean"] = posterior_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_unbounded(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> tuple[float, float]:
    """Compute unbounded posterior for a continuous volatility-parent node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated.
    edges :
        The edges of the probabilistic nodes.

    Returns
    -------
    posterior_precision, posterior_mean :
        Updated precision and mean of the node.
    """
    volatility_child_idx = edges[node_idx].volatility_children[0]  # type: ignore
    time_step = attributes[-1]["time_step"]

    volatility_coupling = attributes[node_idx]["volatility_coupling_children"][0]
    tonic_volatility = attributes[volatility_child_idx]["tonic_volatility"]

    previous_variance = jnp.maximum(
        attributes[volatility_child_idx]["temp"]["current_variance"], 1e-128
    )  # previous-step variance (= 1 / precision at the previous step)
    be_aux = (1.0 / attributes[volatility_child_idx]["precision"]) + (
        attributes[volatility_child_idx]["mean"]
        - attributes[volatility_child_idx]["expected_mean"]
    ) ** 2

    expected_mean = attributes[node_idx]["expected_mean"]
    expected_precision = attributes[node_idx]["expected_precision"]

    # All quantities are kept in log-space whenever they would otherwise pass
    # through ``exp`` of a potentially large number. Forming ``v = exp(γ)``
    # explicitly is correct in the forward pass (downstream uses are
    # saturation-stable, e.g. ``1/(1 + α/v) → 1`` as ``v → ∞``) but corrupts
    # the backward pass: the local partial of the saturating expression is
    # ``0`` while ``d v / d γ = exp(γ) = ∞``, and ``0 · ∞ = NaN``. We rewrite
    # every such occurrence using ``sigmoid``/``logaddexp`` so neither the
    # forward nor the backward ever materialises ``inf``.
    log_time_step = jnp.log(time_step)
    log_previous_variance = jnp.log(previous_variance)

    # Canonical exponent at prediction: γ = log(time_step) + volatility_coupling*expected_mean + tonic_volatility
    gamma_c = log_time_step + volatility_coupling * expected_mean + tonic_volatility

    # w_jm1 = 1/(1 + previous_variance/exp(γ)) = sigmoid(γ − log α). Matches the original
    # ``1/(1 + previous_variance/v_jm1)`` exactly for every finite γ, and stays
    # gradient-safe when γ → ±∞.
    w_jm1 = sigmoid(gamma_c - log_previous_variance)

    # Volatility prediction error: da_jm1 = pihat_jm1 * be_aux - 1, with
    # pihat_jm1 = child's expected_precision (set in the prediction step at
    # mu_prev_j).
    da_jm1 = attributes[volatility_child_idx]["expected_precision"] * be_aux - 1.0

    # ----------------------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ----------------------------------------------------------------------------------
    pi1 = expected_precision + 0.5 * volatility_coupling**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = expected_mean + (volatility_coupling * w_jm1 / (2.0 * pi1)) * da_jm1

    # ----------------------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W0 approximate mode
    # ----------------------------------------------------------------------------------
    pihat_y = expected_precision / volatility_coupling**2

    # Compute W_arg in log-space and cap at log(float_max) — matches MATLAB's
    # "W_arg = exp(min(log_W_arg, log(realmax)))".
    log_W_arg = jnp.log(be_aux) - jnp.log(2.0 * pihat_y) + 0.5 / pihat_y - gamma_c
    log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_W_arg)).max)
    W_arg = jnp.exp(jnp.minimum(log_W_arg, log_float_max))
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - log_time_step - tonic_volatility) / volatility_coupling

    # Log-space form of s2, w2, da2 — equivalent to the original
    #   s2 = time_step * exp(volatility_coupling*x_star + tonic_volatility)
    #   w2 = 1 / (1 + previous_variance / s2)
    #   da2 = be_aux / (previous_variance + s2) - 1
    # but without ever materialising ``s2 = inf`` in the forward pass, which
    # would inject NaN gradients via 0·∞.
    log_s2 = log_time_step + volatility_coupling * x_star + tonic_volatility
    log_denom_s = jnp.logaddexp(
        log_previous_variance, log_s2
    )  # = log(previous_variance + s2)
    w2 = sigmoid(log_s2 - log_previous_variance)
    da2 = be_aux * jnp.exp(-log_denom_s) - 1.0

    pi2_full = expected_precision + 0.5 * volatility_coupling**2 * w2 * (
        w2 + (2.0 * w2 - 1.0) * da2
    )
    pi2_safe = jnp.where(
        pi2_full <= 0.0,
        expected_precision + 0.5 * volatility_coupling**2 * w2 * (1.0 - w2),
        pi2_full,
    )
    mu2_safe = (
        x_star
        + (
            0.5 * volatility_coupling * w2 * da2
            - expected_precision * (x_star - expected_mean)
        )
        / pi2_safe
    )

    # Fall back to Expansion 1 if Expansion 2 yields non-finite results —
    # matches MATLAB: "if ~isfinite(pi2) || ~isfinite(mu2), pi2 = pi1; mu2 = mu1".
    #
    # Double-where masking: replace any non-finite ``pi2_safe`` / ``mu2_safe``
    # with safe constants *before* they enter the outer ``where``. The bare
    # form ``jnp.where(c, pi2_safe, pi1)`` is correct in the forward pass but
    # poisons the backward pass: ``where``'s VJP routes a zero cotangent into
    # the masked-out branch, and ``0 * NaN = NaN`` in IEEE-754 — so a single
    # non-finite intermediate at any scan step turns the whole gradient into
    # NaN, which forces NUTS to reject the trajectory and adapt the step size
    # downward, blowing up the number of leapfrog evaluations per sample.
    exp2_finite = jnp.isfinite(pi2_safe) & jnp.isfinite(mu2_safe)
    pi2_safe_for_grad = jnp.where(exp2_finite, pi2_safe, 1.0)
    mu2_safe_for_grad = jnp.where(exp2_finite, mu2_safe, 0.0)
    pi2 = jnp.where(exp2_finite, pi2_safe_for_grad, pi1)
    mu2 = jnp.where(exp2_finite, mu2_safe_for_grad, mu1)

    # ----------------------------------------------------------------------------------
    # Variational energy-based softmax blend (direct form, matches MATLAB).
    #
    # The original form is
    #     ey  = time_step * exp(volatility_coupling*mu + tonic_volatility)
    #     I   = -0.5 * log(previous_variance + ey) - 0.5 * be_aux / (previous_variance + ey) - ...
    # which materialises ``ey = inf`` for large ``volatility_coupling*mu + tonic_volatility`` and then
    # injects 0·∞ NaNs in the backward pass even though the forward saturates
    # cleanly. The log-space rewrite below is mathematically identical for
    # every finite input and stays gradient-safe at the saturation limits:
    # ``logaddexp`` and ``exp(-positive)`` are both bounded forward and
    # backward.
    # ----------------------------------------------------------------------------------
    log_ey1 = log_time_step + volatility_coupling * mu1 + tonic_volatility
    log_denom_1 = jnp.logaddexp(
        log_previous_variance, log_ey1
    )  # = log(previous_variance + ey1)
    I1 = (
        -0.5 * log_denom_1
        - 0.5 * be_aux * jnp.exp(-log_denom_1)
        - 0.5 * expected_precision * (mu1 - expected_mean) ** 2
    )

    log_ey2 = log_time_step + volatility_coupling * mu2 + tonic_volatility
    log_denom_2 = jnp.logaddexp(log_previous_variance, log_ey2)
    I2 = (
        -0.5 * log_denom_2
        - 0.5 * be_aux * jnp.exp(-log_denom_2)
        - 0.5 * expected_precision * (mu2 - expected_mean) ** 2
    )

    # Stable sigmoid matches b = 1/(1 + exp(I1 - I2)) without NaN at ±∞.
    b = sigmoid(I2 - I1)

    # ----------------------------------------------------------------------------------
    # Gaussian mixture moment matching
    # ----------------------------------------------------------------------------------
    posterior_mean = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) ** 2
    posterior_precision = 1.0 / sig2

    return posterior_precision, posterior_mean
