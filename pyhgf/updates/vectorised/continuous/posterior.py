# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

r"""Vectorised posterior updates for layers of regular continuous nodes.

This module mirrors :mod:`pyhgf.updates.posterior.continuous` for vectorised
layers. A continuous layer can be the *value parent* of one layer (connected by a
weight matrix :math:`W` entering the child's drift) and/or the *volatility parent*
of one layer (connected by a fixed coupling matrix :math:`\kappa` entering the
child's log-volatility); its posterior update integrates the prediction errors of
both children.

Three update schemes are provided, matching the nodalised backend's dispatch in
:func:`pyhgf.utils.get_update_sequence.get_update_sequence`: layers without a
volatility child always use the standard update; layers with one dispatch on
``volatility_updates`` (``"standard"``, ``"eHGF"``, or ``"unbounded"``).
"""

import dataclasses
from typing import Callable, NamedTuple, Optional

import jax.numpy as jnp
from jax import grad, vmap
from jax.nn import sigmoid

from pyhgf.math import lambert_w0
from pyhgf.typing.vectorised import LayerParams, LayerState


class ValueChild(NamedTuple):
    """The value child of a continuous layer, as seen by its posterior update.

    Parameters
    ----------
    state :
        The child layer's state, already carrying its posterior and
        ``value_prediction_error``.
    weights :
        The value-coupling matrix connecting the child into the parent, shape
        ``(n_child, n_parent)``.
    coupling_fn :
        The coupling function on the edge.
    precision_is_clamped :
        Whether the child's posterior precision is never updated — true for the
        clamped observation leaf. The smoothing (structured-Gaussian) factors
        then collapse to the canonical predicted precision; see
        :func:`_smoothing_factors`.
    """

    state: LayerState
    weights: jnp.ndarray
    coupling_fn: Callable
    precision_is_clamped: bool


class VolatilityChild(NamedTuple):
    """The volatility child of a continuous layer, as seen by its posterior update.

    Parameters
    ----------
    state :
        The child layer's state, already carrying its posterior and
        ``volatility_prediction_error``.
    kappa :
        The volatility-coupling matrix, shape ``(n_child, n_parent)``.
    params :
        The child layer's parameters (``tonic_volatility`` is read by the eHGF
        and unbounded updates).
    """

    state: LayerState
    kappa: jnp.ndarray
    params: LayerParams


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def _smoothing_factors(
    child: ValueChild, mean_field_updates: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Child-precision factors weighting the value messages sent to the parent.

    Returns ``(effective_precision, gain)`` — the factors multiplying the child's
    prediction error in :func:`_value_precision_pwpe` and :func:`_value_mean_pwpe`
    respectively. Both are structured-Gaussian (RTS-smoother) forms sharing a
    denominator:

    .. math::

        \pi^{\mathrm{eff}}_a = \frac{\hat{\pi}_a \, \pi^y_a}{\hat{\pi}_a + \pi^y_a},
        \qquad
        g_a = \frac{\hat{\pi}_a \, \pi_a}{\hat{\pi}_a + \pi^y_a},
        \qquad
        \pi^y_a = \pi_a - \tilde{\pi}_a.

    When the child's posterior precision is clamped there is no smoothing
    correction to make, and both collapse to the canonical predicted precision
    :math:`\tilde{\pi}_a`. Under ``mean_field_updates`` the correction is skipped
    for every child — the original mean-field update weights all messages by
    :math:`\tilde{\pi}_a`
    (:func:`pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node._mean_field_child_precision`).
    """
    if child.precision_is_clamped or mean_field_updates:
        return child.state.expected_precision, child.state.expected_precision

    conditional = child.state.conditional_expected_precision
    pi_y = child.state.precision - child.state.expected_precision
    denominator = conditional + pi_y
    return conditional * pi_y / denominator, conditional * child.state.precision / (
        denominator
    )


def _value_precision_pwpe(
    eval_mean: jnp.ndarray,
    child: ValueChild,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Value-coupling precision-weighted prediction error (eq.

    50 form).
        Per parent node :math:`m`:

        .. math::

            \sum_a \pi_a^{\mathrm{eff}} \left( W_{a,m}^2 \, g'(\mu_m)^2
                - W_{a,m} \, g''(\mu_m) \, \delta_a \right),

        with :math:`\pi^{\mathrm{eff}}_a` from :func:`_smoothing_factors`. Derivatives
        are evaluated at *eval_mean* — the parent's previous posterior mean in the
        standard scheme, its freshly updated posterior mean in the eHGF scheme
        (which updates the mean first), mirroring the nodalised call order.
    """
    effective_child_precision, _ = _smoothing_factors(child, mean_field_updates)

    coupling_prime = vmap(grad(child.coupling_fn))(eval_mean)
    coupling_second = vmap(grad(grad(child.coupling_fn)))(eval_mean)

    first_order = jnp.matmul(child.weights.T**2, effective_child_precision) * (
        coupling_prime**2
    )
    second_order = -coupling_second * jnp.matmul(
        child.weights.T,
        effective_child_precision * child.state.value_prediction_error,
    )
    return first_order + second_order


def _value_mean_pwpe(
    eval_mean: jnp.ndarray,
    child: ValueChild,
    node_precision: jnp.ndarray,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Value-coupling mean shift, accumulated across child nodes.

    .. math::

        \Delta \mu_m = \frac{g'(\mu_m)}{\pi_m}
            \sum_a W_{a,m} \, g_a \, \delta_a,

    with the joint-Gaussian gain :math:`g_a` from :func:`_smoothing_factors`.
    """
    _, gain = _smoothing_factors(child, mean_field_updates)

    coupling_prime = vmap(grad(child.coupling_fn))(eval_mean)
    return (
        jnp.matmul(child.weights.T, gain * child.state.value_prediction_error)
        * coupling_prime
        / node_precision
    )


def _volatility_precision_pwpe_standard(child: VolatilityChild) -> jnp.ndarray:
    r"""Compute the standard volatility-coupling precision increment.

    Per (child :math:`a`, parent :math:`m`) edge:
    :math:`\tfrac12 (\kappa \gamma)^2 + (\kappa \gamma)^2 \Delta
    - \tfrac12 \kappa^2 \gamma \Delta`, with :math:`\gamma` the child's
    effective precision from the prediction step and :math:`\Delta` its
    volatility prediction error.
    """
    gamma = child.state.effective_precision
    delta = child.state.volatility_prediction_error
    kappa_sq = child.kappa**2
    return (
        0.5 * jnp.matmul(kappa_sq.T, gamma**2)
        + jnp.matmul(kappa_sq.T, gamma**2 * delta)
        - 0.5 * jnp.matmul(kappa_sq.T, gamma * delta)
    )


def _volatility_mean_pwpe(
    child: VolatilityChild,
    node_precision: jnp.ndarray,
) -> jnp.ndarray:
    r"""Volatility-coupling mean shift, accumulated across child nodes.

    .. math::

        \Delta \mu_m = \frac{1}{2 \pi_m} \sum_a \kappa_{a,m} \,
            \gamma_a \, \Delta_a.
    """
    return jnp.matmul(
        child.kappa.T,
        child.state.effective_precision * child.state.volatility_prediction_error,
    ) / (2.0 * node_precision)


def _child_previous_variance(
    layer: LayerState,
    child: VolatilityChild,
    time_step: float,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Reconstruct the volatility child's previous-step posterior variance.

    The prediction step set
    :math:`1 / \hat{\pi}_a = 1 / \pi_a^{(k-1)} + \Omega_a` with the full
    predicted volatility :math:`\Omega_a` (volatility-parent contribution and
    MGF correction included — the latter absent under ``mean_field_updates``,
    when the prediction dropped it too), so subtracting the same
    :math:`\Omega_a` from the inverse *conditional* predicted precision cancels
    back to :math:`1 / \pi_a^{(k-1)}` — the quantity the nodalised backend
    stores as ``temp["current_variance"]``. The parent's prediction-time
    ``expected_mean`` / ``expected_precision`` are still intact when the
    posterior update runs, so the reconstruction is exact.
    """
    assert child.params.tonic_volatility is not None
    total_volatility = child.params.tonic_volatility + jnp.matmul(
        child.kappa, layer.expected_mean
    )
    if not mean_field_updates:
        total_volatility = total_volatility + jnp.matmul(
            child.kappa**2, 1.0 / (2.0 * layer.expected_precision)
        )
    predicted_volatility = time_step * jnp.exp(total_volatility)
    return jnp.maximum(
        1.0 / child.state.conditional_expected_precision - predicted_volatility,
        1e-128,
    )


def _volatility_precision_pwpe_ehgf(
    posterior_mean: jnp.ndarray,
    layer: LayerState,
    child: VolatilityChild,
    time_step: float,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Enhanced-HGF "safe" volatility-coupling precision increment.

    Mirrors
    :func:`pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node._ehgf_volatility_precision_increment`:
    the child's volatility is re-predicted from the parent's *just-updated*
    posterior mean through each :math:`\kappa_{a,m}` edge alone (single-parent
    linearisation, no MGF term), and each edge's increment
    :math:`\tfrac12 \kappa^2 \gamma (\gamma + w \Delta)` is floored at zero
    before summation, so the posterior precision can never fall below the
    predicted precision.
    """
    assert child.params.tonic_volatility is not None
    previous_variance = _child_previous_variance(
        layer, child, time_step, mean_field_updates
    )

    # Per-edge re-prediction, shape (n_child, n_parent).
    predicted_volatility = time_step * jnp.exp(
        child.kappa * posterior_mean[None, :] + child.params.tonic_volatility[:, None]
    )
    expected_precision = 1.0 / (previous_variance[:, None] + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision
    volatility_error_weight = (
        predicted_volatility - previous_variance[:, None]
    ) * expected_precision
    volatility_prediction_error = (
        1.0 / child.state.precision
        + (child.state.mean - child.state.expected_mean) ** 2
    )[:, None] * expected_precision - 1.0

    increment = jnp.maximum(
        0.0,
        0.5
        * child.kappa**2
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error),
    )
    return increment.sum(axis=0)


def _finalize_precision(
    expected_precision: jnp.ndarray,
    pwpe: jnp.ndarray,
    max_posterior_precision: float,
) -> jnp.ndarray:
    """Add the precision-weighted PE, guard positivity, and apply the cap."""
    posterior_precision = expected_precision + pwpe
    posterior_precision = jnp.where(
        posterior_precision > 1e-128, posterior_precision, jnp.nan
    )
    return jnp.minimum(posterior_precision, max_posterior_precision)


# ---------------------------------------------------------------------------
# Update schemes
# ---------------------------------------------------------------------------


def vectorised_continuous_posterior_update_standard(
    layer: LayerState,
    value_child: Optional[ValueChild] = None,
    volatility_child: Optional[VolatilityChild] = None,
    max_posterior_precision: float = 1e10,
    mean_field_updates: bool = False,
) -> LayerState:
    """Apply the standard HGF posterior update: precision first, then the mean.

    This is the vectorised equivalent of
    :func:`pyhgf.updates.posterior.continuous.continuous_node_posterior_update`.

    Parameters
    ----------
    layer :
        The parent layer's state (being updated).
    value_child :
        The value child, or ``None``.
    volatility_child :
        The volatility child, or ``None``.
    max_posterior_precision :
        Upper bound applied to the posterior precision write.
    mean_field_updates :
        If ``True``, weight the value-coupling messages by the canonical
        predicted precision instead of the smoothing factors — the original
        mean-field update
        (:func:`pyhgf.updates.posterior.continuous.continuous_node_posterior_update_mean_field`).
        The volatility-coupling increment is the same in both schemes.

    Returns
    -------
    LayerState
        Updated layer state with posterior ``precision`` and ``mean``.
    """
    eval_mean = layer.mean  # previous posterior mean

    pwpe = jnp.zeros_like(layer.precision)
    if value_child is not None:
        pwpe = pwpe + _value_precision_pwpe(eval_mean, value_child, mean_field_updates)
    if volatility_child is not None:
        pwpe = pwpe + _volatility_precision_pwpe_standard(volatility_child)
    posterior_precision = _finalize_precision(
        layer.expected_precision, pwpe, max_posterior_precision
    )

    mean_pwpe = jnp.zeros_like(layer.mean)
    if value_child is not None:
        mean_pwpe = mean_pwpe + _value_mean_pwpe(
            eval_mean, value_child, posterior_precision, mean_field_updates
        )
    if volatility_child is not None:
        mean_pwpe = mean_pwpe + _volatility_mean_pwpe(
            volatility_child, posterior_precision
        )
    posterior_mean = layer.expected_mean + mean_pwpe

    return dataclasses.replace(
        layer, precision=posterior_precision, mean=posterior_mean
    )


def vectorised_continuous_posterior_update_ehgf(
    layer: LayerState,
    value_child: Optional[ValueChild] = None,
    volatility_child: Optional[VolatilityChild] = None,
    time_step: float = 1.0,
    max_posterior_precision: float = 1e10,
    mean_field_updates: bool = False,
) -> LayerState:
    """Enhanced-HGF posterior update: mean first, then the safe precision.

    This is the vectorised equivalent of
    :func:`pyhgf.updates.posterior.continuous.continuous_node_posterior_update_ehgf`:
    the mean update approximates the node precision with the expected precision,
    and the volatility-coupling precision increment is recomputed from the
    posterior mean and floored at zero.

    Parameters
    ----------
    layer :
        The parent layer's state (being updated).
    value_child :
        The value child, or ``None``.
    volatility_child :
        The volatility child, or ``None``.
    time_step :
        Current time step (needed to re-predict the child's volatility).
    max_posterior_precision :
        Upper bound applied to the posterior precision write.

    Returns
    -------
    LayerState
        Updated layer state with posterior ``precision`` and ``mean``.
    """
    eval_mean = layer.mean  # previous posterior mean

    mean_pwpe = jnp.zeros_like(layer.mean)
    if value_child is not None:
        mean_pwpe = mean_pwpe + _value_mean_pwpe(
            eval_mean, value_child, layer.expected_precision, mean_field_updates
        )
    if volatility_child is not None:
        mean_pwpe = mean_pwpe + _volatility_mean_pwpe(
            volatility_child, layer.expected_precision
        )
    posterior_mean = layer.expected_mean + mean_pwpe

    # The nodalised backend writes the mean before the precision update, so the
    # value-coupling derivatives are evaluated at the *new* posterior mean.
    pwpe = jnp.zeros_like(layer.precision)
    if value_child is not None:
        pwpe = pwpe + _value_precision_pwpe(
            posterior_mean, value_child, mean_field_updates
        )
    if volatility_child is not None:
        pwpe = pwpe + _volatility_precision_pwpe_ehgf(
            posterior_mean, layer, volatility_child, time_step, mean_field_updates
        )
    posterior_precision = _finalize_precision(
        layer.expected_precision, pwpe, max_posterior_precision
    )

    return dataclasses.replace(
        layer, precision=posterior_precision, mean=posterior_mean
    )


def vectorised_continuous_posterior_update_unbounded(
    layer: LayerState,
    volatility_child: VolatilityChild,
    time_step: float = 1.0,
    max_posterior_precision: float = 1e10,
    mean_field_updates: bool = False,
) -> LayerState:
    """Unbounded posterior update for a pure volatility-parent layer.

    Vectorised port of
    :func:`pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded`
    (Lambert :math:`W_0` dual-quadratic expansion with a variational
    energy-based blend). The approximation is derived for a single volatility
    child per node, so the coupling must be one-to-one: ``kappa`` square and
    diagonal, each parent node coupled to exactly one child node. The layer must
    have no value child (the nodalised update ignores value children; the
    vectorised builder rejects such a topology instead of silently dropping the
    edge).

    Parameters
    ----------
    layer :
        The parent layer's state (being updated).
    volatility_child :
        The volatility child, with a square diagonal ``kappa``.
    time_step :
        Current time step.
    max_posterior_precision :
        Upper bound applied to the posterior precision write.

    Returns
    -------
    LayerState
        Updated layer state with posterior ``precision`` and ``mean``.
    """
    child = volatility_child
    assert child.kappa.shape[0] == child.kappa.shape[1], (
        "The unbounded update requires a one-to-one (square, diagonal) "
        "volatility coupling."
    )
    assert child.params.tonic_volatility is not None
    kappa = jnp.diagonal(child.kappa)
    tonic_volatility = child.params.tonic_volatility

    previous_variance = _child_previous_variance(
        layer, child, time_step, mean_field_updates
    )
    be_aux = (
        1.0 / child.state.precision
        + (child.state.mean - child.state.expected_mean) ** 2
    )

    expected_mean = layer.expected_mean
    expected_precision = layer.expected_precision

    # Log-space throughout: forming ``exp(γ)`` explicitly is forward-stable but
    # injects 0·∞ NaNs in the backward pass; the sigmoid/logaddexp rewrites
    # match the direct forms for every finite input and stay gradient-safe.
    # Mirrors ``continuous_node_posterior_update_unbounded``.
    log_time_step = jnp.log(time_step)
    log_previous_variance = jnp.log(previous_variance)

    # Canonical exponent at prediction:
    # γ = log(time_step) + κ μ̂_parent + ω_child.
    gamma_c = log_time_step + kappa * expected_mean + tonic_volatility

    w_jm1 = sigmoid(gamma_c - log_previous_variance)
    da_jm1 = child.state.expected_precision * be_aux - 1.0

    # ------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction (prior mean)
    # ------------------------------------------------------------------
    pi1 = expected_precision + 0.5 * kappa**2 * w_jm1 * (1.0 - w_jm1)
    mu1 = expected_mean + (kappa * w_jm1 / (2.0 * pi1)) * da_jm1

    # ------------------------------------------------------------------
    # Expansion 2: quadratic at the Lambert W₀ approximate mode
    # ------------------------------------------------------------------
    pihat_y = expected_precision / kappa**2

    log_W_arg = jnp.log(be_aux) - jnp.log(2.0 * pihat_y) + 0.5 / pihat_y - gamma_c
    log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_W_arg)).max)
    W_arg = jnp.exp(jnp.minimum(log_W_arg, log_float_max))
    v_W = lambert_w0(W_arg)
    y_star = gamma_c + v_W - 0.5 / pihat_y
    x_star = (y_star - log_time_step - tonic_volatility) / kappa

    log_s2 = log_time_step + kappa * x_star + tonic_volatility
    log_denom_s = jnp.logaddexp(log_previous_variance, log_s2)
    w2 = sigmoid(log_s2 - log_previous_variance)
    da2 = be_aux * jnp.exp(-log_denom_s) - 1.0

    pi2_full = expected_precision + 0.5 * kappa**2 * w2 * (w2 + (2.0 * w2 - 1.0) * da2)
    pi2_safe = jnp.where(
        pi2_full <= 0.0,
        expected_precision + 0.5 * kappa**2 * w2 * (1.0 - w2),
        pi2_full,
    )
    mu2_safe = (
        x_star
        + (0.5 * kappa * w2 * da2 - expected_precision * (x_star - expected_mean))
        / pi2_safe
    )

    # Double-where masking: sanitise non-finite Expansion-2 results before the
    # outer ``where`` so its VJP never does ``0 * NaN``.
    exp2_finite = jnp.isfinite(pi2_safe) & jnp.isfinite(mu2_safe)
    pi2_safe_for_grad = jnp.where(exp2_finite, pi2_safe, 1.0)
    mu2_safe_for_grad = jnp.where(exp2_finite, mu2_safe, 0.0)
    pi2 = jnp.where(exp2_finite, pi2_safe_for_grad, pi1)
    mu2 = jnp.where(exp2_finite, mu2_safe_for_grad, mu1)

    # ------------------------------------------------------------------
    # Variational energy-based softmax blend (log-space, gradient-safe)
    # ------------------------------------------------------------------
    log_ey1 = log_time_step + kappa * mu1 + tonic_volatility
    log_denom_1 = jnp.logaddexp(log_previous_variance, log_ey1)
    I1 = (
        -0.5 * log_denom_1
        - 0.5 * be_aux * jnp.exp(-log_denom_1)
        - 0.5 * expected_precision * (mu1 - expected_mean) ** 2
    )

    log_ey2 = log_time_step + kappa * mu2 + tonic_volatility
    log_denom_2 = jnp.logaddexp(log_previous_variance, log_ey2)
    I2 = (
        -0.5 * log_denom_2
        - 0.5 * be_aux * jnp.exp(-log_denom_2)
        - 0.5 * expected_precision * (mu2 - expected_mean) ** 2
    )

    b = sigmoid(I2 - I1)

    # ------------------------------------------------------------------
    # Gaussian mixture moment matching
    # ------------------------------------------------------------------
    posterior_mean = (1.0 - b) * mu1 + b * mu2
    sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2) ** 2
    posterior_precision = jnp.minimum(1.0 / sig2, max_posterior_precision)

    return dataclasses.replace(
        layer, precision=posterior_precision, mean=posterior_mean
    )


def vectorised_continuous_posterior_update(
    layer: LayerState,
    value_child: Optional[ValueChild] = None,
    volatility_child: Optional[VolatilityChild] = None,
    volatility_updates: str = "eHGF",
    time_step: float = 1.0,
    max_posterior_precision: float = 1e10,
    mean_field_updates: bool = False,
) -> LayerState:
    """Dispatch the continuous posterior update on the layer's children.

    Mirrors the assignment rule of
    :func:`pyhgf.utils.get_update_sequence.get_update_sequence`: layers without
    a volatility child always take the standard update; layers with one take
    the update named by *volatility_updates*.

    Parameters
    ----------
    layer :
        The parent layer's state (being updated).
    value_child :
        The value child, or ``None``.
    volatility_child :
        The volatility child, or ``None``.
    volatility_updates :
        One of ``"standard"``, ``"eHGF"``, or ``"unbounded"``.
    time_step :
        Current time step.
    max_posterior_precision :
        Upper bound applied to the posterior precision write.
    mean_field_updates :
        If ``True``, the value-coupling messages use the original mean-field
        forms and the variance reconstructions drop the MGF term the mean-field
        prediction never added; see the individual update schemes.

    Returns
    -------
    LayerState
        Updated layer state with posterior ``precision`` and ``mean``.
    """
    if volatility_child is None or volatility_updates == "standard":
        return vectorised_continuous_posterior_update_standard(
            layer,
            value_child=value_child,
            volatility_child=volatility_child,
            max_posterior_precision=max_posterior_precision,
            mean_field_updates=mean_field_updates,
        )
    if volatility_updates == "eHGF":
        return vectorised_continuous_posterior_update_ehgf(
            layer,
            value_child=value_child,
            volatility_child=volatility_child,
            time_step=time_step,
            max_posterior_precision=max_posterior_precision,
            mean_field_updates=mean_field_updates,
        )
    if volatility_updates == "unbounded":
        if value_child is not None:
            raise ValueError(
                "The unbounded update is derived for pure volatility parents; "
                "this layer also has a value child. Use 'standard' or 'eHGF' "
                "volatility updates for mixed parents."
            )
        return vectorised_continuous_posterior_update_unbounded(
            layer,
            volatility_child=volatility_child,
            time_step=time_step,
            max_posterior_precision=max_posterior_precision,
            mean_field_updates=mean_field_updates,
        )
    raise ValueError(
        f"Invalid volatility_updates {volatility_updates!r}. "
        "Choose from 'standard', 'eHGF', or 'unbounded'."
    )
