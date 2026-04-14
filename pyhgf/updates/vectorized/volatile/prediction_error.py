# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction error and volatility posterior for volatile node layers.

This module mirrors :mod:`pyhgf.updates.prediction_error.volatile` for vectorized
layers: it provides separate value and volatility prediction-error functions, per-
update-type volatility posterior functions, and a combined driver that calls them in
the correct order.
"""

import jax.numpy as jnp
from jax.nn import sigmoid

from pyhgf.math import smoothed_rectangular
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
        + layer.expected_precision * (layer.value_prediction_error**2)
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
    """Unbounded volatility-level posterior update (quadratic approximation).

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
    vol_coupling = params.volatility_coupling

    # Reconstruct pre-prediction variance (1/precision before prediction step).
    predicted_volatility = time_step * jnp.exp(
        jnp.clip(
            params.tonic_volatility + vol_coupling * layer.expected_mean_vol,
            a_min=-80.0,
            a_max=80.0,
        )
    )
    previous_child_variance = 1.0 / layer.expected_precision - predicted_volatility
    previous_child_variance = jnp.maximum(previous_child_variance, 1e-128)

    # Delta: normalised innovation
    delta_child = (
        (1.0 / layer.precision) + (layer.mean - layer.expected_mean) ** 2
    ) / (
        previous_child_variance
        + jnp.exp(
            jnp.clip(
                vol_coupling * layer.expected_mean_vol + params.tonic_volatility,
                a_min=-80.0,
                a_max=80.0,
            )
        )
    ) - 1.0

    # ------------------------------------------------------------------
    # First quadratic approximation L1
    # ------------------------------------------------------------------
    x = vol_coupling * layer.expected_mean_vol + params.tonic_volatility

    w_child = sigmoid(x - jnp.log(previous_child_variance))

    pi_l1 = layer.expected_precision_vol + 0.5 * vol_coupling**2 * w_child * (
        1.0 - w_child
    )

    mu_l1 = (
        layer.expected_mean_vol
        + ((vol_coupling * w_child) / (2.0 * pi_l1)) * delta_child
    )

    # ------------------------------------------------------------------
    # Second quadratic approximation L2
    # ------------------------------------------------------------------
    phi = jnp.log(previous_child_variance * (2.0 + jnp.sqrt(3.0)))

    w_phi = jnp.exp(vol_coupling * phi + params.tonic_volatility) / (
        previous_child_variance + jnp.exp(vol_coupling * phi + params.tonic_volatility)
    )

    delta_phi = ((1.0 / layer.precision) + (layer.mean - layer.expected_mean) ** 2) / (
        previous_child_variance + jnp.exp(vol_coupling * phi + params.tonic_volatility)
    ) - 1.0

    pi_l2 = layer.expected_precision_vol + 0.5 * vol_coupling**2 * w_phi * (
        w_phi + (2.0 * w_phi - 1.0) * delta_phi
    )

    mu_hat_phi = ((2.0 * pi_l2 - 1.0) * phi + layer.expected_mean_vol) / (2.0 * pi_l2)

    mu_l2 = mu_hat_phi + ((vol_coupling * w_phi) / (2.0 * pi_l2)) * delta_phi

    # ------------------------------------------------------------------
    # Full quadratic approximation: weighted combination of L1 and L2
    # ------------------------------------------------------------------
    theta_l = jnp.sqrt(
        1.2
        * (
            ((1.0 / layer.precision) + (layer.mean - layer.expected_mean) ** 2)
            / (previous_child_variance * pi_l1)
        )
    )

    weighting = smoothed_rectangular(
        x=layer.expected_mean_vol,
        theta_l=theta_l,
        phi_l=8.0,
        theta_r=0.0,
        phi_r=1.0,
    )

    posterior_precision_vol = (1.0 - weighting) * pi_l1 + weighting * pi_l2
    posterior_mean_vol = (1.0 - weighting) * mu_l1 + weighting * mu_l2

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

    Returns
    -------
    LayerState
        Updated layer state with prediction errors and volatility posterior.
    """
    # 1. Prediction errors
    layer = vectorized_layer_value_prediction_error(
        layer,
        n_parents,
    )
    layer = vectorized_layer_volatility_prediction_error(layer)

    # 2. Posterior updates for the volatility level
    if update_type == "eHGF":
        layer = vectorized_layer_volatility_posterior_ehgf(layer, params)
    elif update_type == "standard":
        layer = vectorized_layer_volatility_posterior_standard(layer, params)
    elif update_type == "unbounded":
        layer = vectorized_layer_volatility_posterior_unbounded(
            layer, params, time_step
        )

    return layer
