# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized posterior update for volatile node layers."""

from typing import Callable

import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing import LayerParams, LayerState


def vectorized_layer_posterior_update(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    params: LayerParams,
    coupling_fn_grad: Callable,
    n_value_parents: int = 1,
) -> LayerState:
    """Update posterior for all nodes in parent layer (volatile node - 5 steps).

    This implements the full volatile node posterior update with both value level and
    volatility level updates.

    Parameters
    ----------
    layer :
        Current state of the parent layer (being updated).
    child :
        Current state of the child layer (providing prediction errors).
    weights :
        Weight matrix connecting child to parent, shape (n_children, n_parents).
    params :
        Layer parameters for the parent layer.
    coupling_fn_grad :
        Gradient of the coupling function.

    Returns
    -------
    LayerState
        Updated parent layer state with posterior mean and precision.
    """
    # Coupling derivatives at parent means
    # vmap the gradient function over parent means
    coupling_prime = vmap(coupling_fn_grad)(layer.expected_mean)

    # === STEP 1: Update value level precision ===
    # Coupling second derivative at parent means (for second-order EKF correction)
    coupling_second = vmap(jgrad(coupling_fn_grad))(layer.expected_mean)

    # First-order term: weights.T shape (n_parents, n_children)
    precision_contrib_1 = jnp.matmul(weights.T**2, child.expected_precision) * (
        coupling_prime**2
    )

    # Second-order correction: matches posterior_update_precision_value_level
    # -f''(m_j) * sum_i(pi_i * vpe_i) — note: no weight factor in sum (mirrors standard code)
    sum_pi_vpe = jnp.dot(child.expected_precision, child.value_prediction_error)
    precision_contrib_2 = -coupling_second * sum_pi_vpe

    posterior_precision = jnp.clip(
        layer.expected_precision + precision_contrib_1 + precision_contrib_2,
        a_max=1e8,
    )

    # === STEP 2: Update value level mean ===
    # Weighted prediction error from children
    weighted_pe = (
        jnp.matmul(weights.T, child.expected_precision * child.value_prediction_error)
        * coupling_prime
        / posterior_precision
    )

    posterior_mean = layer.expected_mean + weighted_pe

    # === STEP 3: Recompute volatility PE with fresh values ===
    # This matches standard HGF behavior where volatility parents
    # update in the same timestep using fresh volatility PEs.
    # Divide by n_value_parents to match volatile_node_value_prediction_error.
    fresh_value_pe = (posterior_mean - layer.expected_mean) / n_value_parents

    volatility_pe = (
        (layer.expected_precision / posterior_precision)
        + layer.expected_precision * (fresh_value_pe**2)
        - 1.0
    )

    # === STEP 4 (eHGF order): Update volatility mean FIRST ===
    # eHGF updates mean before precision, using expected_precision_vol as approximation.
    # This matches Network(update_type="eHGF") which is now the default upstream.
    vol_coupling = params.volatility_coupling
    eff_prec = layer.effective_precision

    precision_weighted_pe_vol = (vol_coupling * eff_prec * volatility_pe) / (
        2.0 * layer.expected_precision_vol
    )  # use EXPECTED, not posterior

    posterior_mean_vol = layer.expected_mean_vol + precision_weighted_pe_vol

    # === STEP 5: Update volatility level precision ===
    precision_vol_contrib = (
        0.5 * ((vol_coupling * eff_prec) ** 2)
        + ((vol_coupling * eff_prec) ** 2) * volatility_pe
        - 0.5 * (vol_coupling**2) * eff_prec * volatility_pe
    )

    posterior_precision_vol = jnp.clip(
        layer.expected_precision_vol + precision_vol_contrib, a_max=1e8
    )

    return layer._replace(
        precision=posterior_precision,
        mean=posterior_mean,
        precision_vol=posterior_precision_vol,
        mean_vol=posterior_mean_vol,
        volatility_prediction_error=volatility_pe,
    )
