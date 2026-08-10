# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction update for volatile node layers."""

import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array, grad, vmap

from pyhgf.typing.vectorised import LayerParams, LayerState


def _predict_volatility_level(
    state: LayerState,
    params: LayerParams,
    time_step: float,
    has_volatility_parent: bool,
) -> tuple[Optional[Array], Optional[Array], Optional[Array], Array]:
    r"""Advance a layer's internal volatility level and read off its diffusion term.

    Shared by every value-level prediction, whether the layer is predicted from a
    parent (:func:`vectorized_layer_prediction`) or from nothing at all
    (:func:`vectorized_root_prediction`).

    Returns ``(expected_mean_vol, expected_precision_vol, effective_precision_vol,
    predicted_volatility)``. The last is :math:`\Omega_a^{(k)}`, the variance the
    value level accumulates over this time step; it carries the closed-form
    moment-generating-function correction :math:`1 / (2 \hat{\pi}_{\mathrm{vol}})`
    that comes from marginalising over the volatility level's Gaussian rather than
    collapsing it to a point estimate. Volatility coupling is fixed at 1 and the
    value level carries no tonic volatility of its own.

    Without a volatility parent the level is frozen: its three fields pass through
    unchanged and the diffusion term is zero, so the value level does not undergo a
    Gaussian random walk between observations.
    """
    if not has_volatility_parent:
        return (
            state.mean_vol,
            state.precision_vol,
            state.effective_precision_vol,
            jnp.zeros_like(state.precision),
        )

    assert state.mean_vol is not None
    assert state.precision_vol is not None

    # Autoconnection strength of the volatility level is fixed at 1.
    expected_mean_vol = state.mean_vol

    predicted_volatility_vol = time_step * jnp.exp(params.tonic_volatility_vol)
    predicted_volatility_vol = jnp.where(
        predicted_volatility_vol > 1e-128, predicted_volatility_vol, jnp.nan
    )
    expected_precision_vol = 1.0 / (
        1.0 / state.precision_vol + predicted_volatility_vol
    )
    effective_precision_vol = predicted_volatility_vol * expected_precision_vol

    total_volatility = expected_mean_vol + 1.0 / (2.0 * expected_precision_vol)
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    return (
        expected_mean_vol,
        expected_precision_vol,
        effective_precision_vol,
        predicted_volatility,
    )


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
    predict_precision: bool = True,
) -> LayerState:
    r"""Predict expected mean/precision for all nodes in a volatile-node layer.

    Computes both the value-level (external) and volatility-level (internal)
    predictions. Two predicted precisions of the value level are stored:

    .. math::

        \hat{\pi}_a^{(k)} = \left( \frac{1}{\pi_a^{(k-1)}} + \Omega_a^{(k)}
            \right)^{-1}, \qquad
        \frac{1}{\tilde{\pi}_a^{(k)}} = \frac{1}{\hat{\pi}_a^{(k)}}
            + \sum_b \frac{ (t^{(k)} \, \alpha_b \, g'(\hat{\mu}_b))^2 }
                          { \tilde{\pi}_b },

    where :math:`\hat{\pi}_a` (``conditional_expected_precision``) is the
    AR-plus-volatility chain precision without parent-uncertainty bleed-through and
    :math:`\tilde{\pi}_a` (``expected_precision``) adds the first-order Laplace
    value-coupling contribution from each value parent. The bleed-through term uses
    the parent's *marginal* predicted precision :math:`\tilde{\pi}_b`
    (``parent_state.expected_precision``), which generalises the artifact's
    two-node :math:`\hat{\pi}_b` to deep networks by propagating each parent's full
    marginal predictive variance. The volatility-coupling correction
    :math:`\kappa^2 / (2 \hat{\pi}_{\mathrm{vol}})` enters :math:`\Omega_a^{(k)}`
    inside the log-volatility exponent.

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
        Time step :math:`t^{(k)}` for the prediction.
    coupling_fn :
        Coupling function applied to parent means (default :func:`jax.numpy.tanh`).
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0)
        appended to its activations. The last column of *weights* carries the
        bias connections and is treated as linearly coupled (:math:`g(1) = 1`),
        regardless of *coupling_fn*; the constant node's derivative is zero and
        its predicted precision is infinite, so it contributes nothing to the
        value-coupling variance.
    predict_precision :
        Whether the prediction step advances the precisions; see
        :func:`vectorized_layer_prediction`.
    has_volatility_parent :
        If True (default), the layer has an implied internal volatility parent
        whose state (``mean_vol``, ``precision_vol``) is predicted and updated.
        If False, the value level has no volatility source at all, so it does not
        undergo a Gaussian random walk: the diffusion term is dropped and the
        conditional predicted precision equals the prior precision.
    predict_precision :
        Whether the prediction step advances the precisions at all. With ``False``
        both predicted precisions are held at the layer's prior precision, the
        effective precision is zero, and the volatility level is left untouched,
        so the only thing prediction produces is the expected mean. Precision then
        moves solely through the posterior update, if that is enabled. This is the
        same treatment ``is_input_layer`` gives an observed leaf, applied to every
        layer.
    is_input_layer :
        If True, the layer is treated as an observed input/leaf: it does not
        undergo a Gaussian random walk between observations. The volatility
        contribution to the value-level expected precision is skipped,
        ``expected_precision`` and ``conditional_expected_precision`` are both set
        to the prior precision, and the effective precision is zeroed — mirroring
        the continuous-node treatment in
        :func:`pyhgf.updates.prediction.continuous.continuous_node_prediction`.

    Returns
    -------
    LayerState
        Updated child layer state with predicted means and precisions populated
        for both the value and volatility levels.
    """
    # 1. VOLATILITY LEVEL PREDICTION (internal) ----------------------------------------
    # ----------------------------------------------------------------------------------
    if predict_precision:
        (
            expected_mean_vol,
            expected_precision_vol,
            effective_precision_vol,
            predicted_volatility,
        ) = _predict_volatility_level(
            child_state, params, time_step, has_volatility_parent
        )
    else:
        # Nothing downstream reads the volatility level when the value-level
        # precisions are frozen: its only route into the value level is the
        # diffusion term, and its own posterior update is driven by the effective
        # precision, which is zero below.
        expected_mean_vol = child_state.mean_vol
        expected_precision_vol = child_state.precision_vol
        effective_precision_vol = child_state.effective_precision_vol

    # 2. VALUE LEVEL PREDICTION (external) ---------------------------------------------
    # ----------------------------------------------------------------------------------

    # Mean prediction via matrix multiply
    # weights shape: (n_children, n_parents) or (n_children, n_parents + 1)
    # parent_state.expected_mean shape: (n_parents,)
    # Apply coupling to the parent activations only; the constant bias node is
    # always wired in linearly (g(1) = 1) regardless of coupling_fn — the same
    # convention as the binary prediction, the weight-learning step, the
    # per-node backend, and the Rust backend (constant-state nodes are forced
    # to identity coupling).
    coupled_parents = coupling_fn(parent_state.expected_mean)
    if parent_has_constant:
        coupled_parents = jnp.concatenate([coupled_parents, jnp.ones(1)])

    # Expected mean for value level
    # Note: autoconnection_strength = 0 for i.i.d. classification
    # (the previous observation should not bias the next prediction)
    # Here we remove the influence of time_step on the expected mean.
    expected_mean = jnp.matmul(weights, coupled_parents)

    if predict_precision:
        # Laplace value-coupling correction. Linearising g at μ̂_b and marginalising
        # over each parent's Gaussian yields, per child node i, the additional variance
        #     Σ_j (t · W[i, j] · g'(μ̂_j))² / π̃_j,
        # using the parent's marginal predicted precision π̃_j (= `expected_precision`).
        # The constant-bias parent (if any) has infinite precision and contributes zero.
        parent_precision = parent_state.expected_precision
        g_prime = vmap(grad(coupling_fn))(parent_state.expected_mean)
        if parent_has_constant:
            # The constant node never varies: zero derivative, infinite precision.
            parent_precision = jnp.concatenate([parent_precision, jnp.array([jnp.inf])])
            g_prime = jnp.concatenate([g_prime, jnp.zeros(1)])
        # Σ_j (t · W[i, j] · g'_j)² / π_j, computed as (W²) @ (g'² / π): the
        # weight-dependent factor is a constant matrix, so the per-node sum is a
        # matrix product with a per-node vector.
        # The variance carries no time step (deviation from the Network class).
        per_parent_variance = g_prime**2 / parent_precision
        value_coupling_variance = jnp.matmul(weights**2, per_parent_variance)

        # Conditional predicted precision π̂_a — the precision of x_a given a specific
        # value of x_b (own AR-plus-volatility variance only, no parent-uncertainty
        # bleed-through). This is the precision that enters the joint (x_a, x_b)
        # Gaussian's Schur complement at the parent's posterior-step (smoothing)
        # correction; substituting π̃_a there would double-count parent uncertainty.
        conditional_expected_precision = 1.0 / (
            1.0 / child_state.precision + predicted_volatility
        )

        # Marginal predicted precision π̃_a — inverse marginal predictive variance,
        # adding the law-of-total-variance bleed-through to the conditional variance.
        expected_precision = 1.0 / (
            1.0 / conditional_expected_precision + value_coupling_variance
        )

        # Effective precision γ — only the volatility-driven part enters γ, since γ
        # is consumed by the volatility-coupling posterior update.
        effective_precision = predicted_volatility * expected_precision
    else:
        # Frozen: both predicted precisions stay at the prior and nothing is
        # carried into the volatility-coupling update. The parent's derivative is
        # not needed either, so the autodiff call is skipped with it.
        conditional_expected_precision = child_state.precision
        expected_precision = child_state.precision
        effective_precision = jnp.zeros_like(child_state.precision)

    # Input/leaf override: an observed layer with no value children does not
    # undergo a Gaussian random walk between observations, so the
    # tonic-volatility contribution to the value-level expected precision is
    # dropped (matches the continuous-node treatment). The conditional and
    # marginal coincide in this regime since both lose the volatility term and
    # there is no parent-uncertainty bleed-through to apply.
    if is_input_layer:
        expected_precision = child_state.precision
        conditional_expected_precision = child_state.precision
        effective_precision = jnp.zeros_like(effective_precision)

    return dataclasses.replace(
        child_state,
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        conditional_expected_precision=conditional_expected_precision,
        effective_precision=effective_precision,
        expected_mean_vol=expected_mean_vol,
        expected_precision_vol=expected_precision_vol,
        effective_precision_vol=effective_precision_vol,
    )


def vectorized_root_prediction(
    layer_state: LayerState,
    params: LayerParams,
    time_step: float,
    has_volatility_parent: bool = True,
    predict_precision: bool = True,
) -> LayerState:
    r"""Predict the precisions of a layer that has no value parent above it.

    The top (input) layer of a deep network is clamped to the predictors, so its
    expected mean is supplied from outside the hierarchy rather than produced by a
    parent's top-down prediction. Everything else about it is still predicted: the
    volatility level advances exactly as it does anywhere else, and the value-level
    predicted precisions follow the same chain as
    :func:`vectorized_layer_prediction`, minus the value-coupling variance. The conditional and the
    marginal predicted precision therefore coincide:

    .. math::

        \hat{\pi}_a^{(k)} = \tilde{\pi}_a^{(k)}
            = \left( \frac{1}{\pi_a^{(k-1)}} + \Omega_a^{(k)} \right)^{-1}.

    This is what lets the input layer's precision *move*. Without it,
    ``expected_precision`` is frozen at whatever the layer was built with, and
    because the layer below reads it to form its own value-coupling variance, the
    whole hierarchy inherits a constant input precision. Running this kernel makes
    that quantity track the evidence the layer has accumulated: the posterior
    precision written by the bottom-up sweep enters here on the next step, damped by
    the volatility level.

    ``expected_mean`` is deliberately left untouched.

    Parameters
    ----------
    layer_state :
        Current state of the layer, carrying the posterior ``precision`` (and, with a
        volatility parent, ``precision_vol`` / ``mean_vol``) left by the previous step.
    params :
        Layer parameters for this layer.
    time_step :
        Time step :math:`t^{(k)}` for the prediction.
    has_volatility_parent :
        If True (default), the implied internal volatility parent is predicted and
        contributes a diffusion term to the value level. If False, the value level has
        no volatility source, so it does not drift between observations and the
        predicted precision is just the prior precision.

    Returns
    -------
    LayerState
        Updated layer state with the predicted precisions of both levels populated,
        and ``expected_mean`` unchanged.
    """
    if predict_precision:
        (
            expected_mean_vol,
            expected_precision_vol,
            effective_precision_vol,
            predicted_volatility,
        ) = _predict_volatility_level(
            layer_state, params, time_step, has_volatility_parent
        )

        # No value parent, hence no value-coupling variance: the conditional and
        # the marginal predicted precision are the same quantity.
        expected_precision = 1.0 / (1.0 / layer_state.precision + predicted_volatility)
        effective_precision = predicted_volatility * expected_precision
    else:
        # Frozen, as in :func:`vectorized_layer_prediction`.
        expected_mean_vol = layer_state.mean_vol
        expected_precision_vol = layer_state.precision_vol
        effective_precision_vol = layer_state.effective_precision_vol
        expected_precision = layer_state.precision
        effective_precision = jnp.zeros_like(layer_state.precision)

    return dataclasses.replace(
        layer_state,
        expected_precision=expected_precision,
        conditional_expected_precision=expected_precision,
        effective_precision=effective_precision,
        expected_mean_vol=expected_mean_vol,
        expected_precision_vol=expected_precision_vol,
        effective_precision_vol=effective_precision_vol,
    )
