# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.math import lambert_w0


@partial(jit, static_argnames=("node_idx",))
def volatile_node_posterior_update_unbounded(
    attributes: dict,
    node_idx: int,
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

    # Canonical exponent at the prediction: y = log(t_k) + ka*muhat_j + om
    gamma_c = jnp.log(t_k) + ka * muhat_j + om

    # Recompute v and w using muhat_j
    v_jm1 = jnp.exp(jnp.clip(gamma_c, a_min=-80.0, a_max=80.0))
    w_jm1 = v_jm1 / (al_aux + v_jm1)

    # Volatility prediction error (da_jm1 in Matlab)
    da_jm1 = be_aux / (al_aux + v_jm1) - 1.0

    # ----------------------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ----------------------------------------------------------------------------------
    pi1 = pihat_j + 0.5 * ka**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = muhat_j + (ka * w_jm1 / (2.0 * pi1)) * da_jm1

    # ----------------------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W_0 approximate mode
    # ----------------------------------------------------------------------------------
    pihat_y = pihat_j / ka**2

    W_arg = (be_aux / (2.0 * pihat_y)) * jnp.exp(
        jnp.clip(0.5 / pihat_y - gamma_c, a_min=-80.0, a_max=80.0)
    )
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - jnp.log(t_k) - om) / ka

    s2 = t_k * jnp.exp(jnp.clip(ka * x_star + om, a_min=-80.0, a_max=80.0))
    w2 = s2 / (al_aux + s2)
    da2 = be_aux / (al_aux + s2) - 1.0

    pi2_full = pihat_j + 0.5 * ka**2 * w2 * (w2 + (2.0 * w2 - 1.0) * da2)
    # Guard against negative precision (Matlab fallback: use w2*(1-w2) form)
    pi2 = jnp.where(
        pi2_full <= 0.0,
        pihat_j + 0.5 * ka**2 * w2 * (1.0 - w2),
        pi2_full,
    )
    mu2 = x_star + (0.5 * ka * w2 * da2 - pihat_j * (x_star - muhat_j)) / pi2

    # ----------------------------------------------------------------------------------
    # Variational energy-based softmax blend
    # ----------------------------------------------------------------------------------
    ey1 = t_k * jnp.exp(jnp.clip(ka * mu1 + om, a_min=-80.0, a_max=80.0))
    I1 = (
        -0.5 * jnp.log(al_aux + ey1)
        - 0.5 * be_aux / (al_aux + ey1)
        - 0.5 * pihat_j * (mu1 - muhat_j) ** 2
    )

    ey2 = t_k * jnp.exp(jnp.clip(ka * mu2 + om, a_min=-80.0, a_max=80.0))
    I2 = (
        -0.5 * jnp.log(al_aux + ey2)
        - 0.5 * be_aux / (al_aux + ey2)
        - 0.5 * pihat_j * (mu2 - muhat_j) ** 2
    )

    b = 1.0 / (
        1.0 + jnp.exp(I1 - I2)
    )  # sigmoid(I2 - I1): higher energy -> higher weight

    # ----------------------------------------------------------------------------------
    # Gaussian mixture moment matching
    # ----------------------------------------------------------------------------------
    posterior_mean = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) ** 2
    posterior_precision = 1.0 / sig2

    attributes[node_idx]["precision_vol"] = posterior_precision
    attributes[node_idx]["mean_vol"] = posterior_mean

    return attributes
