# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorised prediction update for layers of regular continuous nodes."""

import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from jax import grad, vmap

from pyhgf.typing.vectorised import LayerParams, LayerState


def vectorised_continuous_prediction(
    child_state: LayerState,
    params: LayerParams,
    time_step: float,
    value_parent_state: Optional[LayerState] = None,
    weights: Optional[jnp.ndarray] = None,
    coupling_fn: Optional[Callable] = None,
    volatility_parent_state: Optional[LayerState] = None,
    volatility_weights: Optional[jnp.ndarray] = None,
    is_static_leaf: bool = False,
    mean_field_updates: bool = False,
) -> LayerState:
    r"""Predict expected mean and precisions for a layer of continuous nodes.

    This is the vectorised equivalent of
    :func:`pyhgf.updates.prediction.continuous.continuous_node_prediction`, applied
    to a whole layer at once with the value and volatility parents generalised to
    *layers* connected by matrices.

    The expected mean follows the standard HGF drift semantics,

    .. math::

        \hat{\mu}_a^{(k)} = \lambda_a \mu_a^{(k-1)}
            + t^{(k)} \left( \rho_a + \sum_b W_{a,b} \, g(\hat{\mu}_b) \right),

    so the value parent nudges the child's own autoregressive state rather than
    replacing it (contrast with
    :func:`pyhgf.updates.vectorised.volatile.prediction.vectorised_layer_prediction`,
    where the parent's prediction fully determines the child's expected mean).

    The two predicted precisions follow the improved (piHGF) scheme of the
    nodalised backend, with the volatility parent's moment-generating-function
    correction and the value parent's first-order Laplace term:

    .. math::

        \Omega_a^{(k)} = t^{(k)} \exp\!\left( \omega_a
            + \sum_j \left( \kappa_{a,j} \hat{\mu}_j
            + \frac{\kappa_{a,j}^2}{2 \tilde{\pi}_j} \right) \right), \qquad
        \hat{\pi}_a^{(k)} = \left( \frac{1}{\pi_a^{(k-1)}} + \Omega_a^{(k)}
            \right)^{-1},

    .. math::

        \frac{1}{\tilde{\pi}_a^{(k)}} = \frac{1}{\hat{\pi}_a^{(k)}}
            + \sum_b \frac{(t^{(k)} W_{a,b} \, g'(\hat{\mu}_b))^2}{\tilde{\pi}_b},
        \qquad
        \gamma_a^{(k)} = \Omega_a^{(k)} \, \tilde{\pi}_a^{(k)}.

    Unlike the volatile-layer kernel, the Laplace value-coupling term carries the
    time step (the drift contribution to the mean is scaled by :math:`t^{(k)}`),
    matching the nodalised backend exactly.

    Parameters
    ----------
    child_state :
        Current state of the layer being predicted.
    params :
        The layer's parameters; ``tonic_volatility``, ``tonic_drift`` and
        ``autoconnection_strength`` must be set (see
        :meth:`pyhgf.typing.vectorised.LayerParams.create_continuous`).
    time_step :
        Time step :math:`t^{(k)}` for the prediction.
    value_parent_state :
        State of the value-parent layer, or ``None`` when the layer has no value
        parent (the drift is then :math:`\rho` alone).
    weights :
        Value-coupling matrix connecting this layer to its value parent, shape
        ``(n_self, n_parent)``. Required with *value_parent_state*.
    coupling_fn :
        Coupling function applied elementwise to the value parent's expected
        means. Required with *value_parent_state*.
    volatility_parent_state :
        State of the volatility-parent layer, or ``None`` when the layer's
        volatility is tonic only.
    volatility_weights :
        Volatility-coupling matrix :math:`\kappa`, shape ``(n_self, n_parent)``.
        Required with *volatility_parent_state*.
    is_static_leaf :
        If True, the layer is the clamped observation leaf *without* a volatility
        parent: it does not undergo a Gaussian random walk between observations,
        so both predicted precisions are held at the prior precision (the
        nodalised backend's input-node convention). A leaf *with* a volatility
        parent does walk, so it takes the regular path and must be passed with
        ``is_static_leaf=False``. Distinct from
        :attr:`~pyhgf.updates.vectorised.continuous.posterior.ValueChild.precision_is_clamped`,
        which holds for *every* clamped leaf.
    mean_field_updates :
        If ``True``, use the original mean-field prediction: the volatility
        parent's MGF correction :math:`\kappa^2 / (2 \tilde{\pi})` and the value
        parent's Laplace variance term are both dropped, so the conditional and
        the marginal predicted precision coincide — matching
        :func:`pyhgf.updates.prediction.continuous.predict_precision_mean_field`.

    Returns
    -------
    LayerState
        Updated layer state with ``expected_mean``, ``expected_precision``,
        ``conditional_expected_precision`` and ``effective_precision`` populated.
    """
    assert params.tonic_drift is not None
    assert params.tonic_volatility is not None
    assert params.autoconnection_strength is not None

    # 1. Expected mean: autoregression plus time-scaled drift.
    driftrate = params.tonic_drift
    if value_parent_state is not None:
        assert weights is not None and coupling_fn is not None
        driftrate = driftrate + jnp.matmul(
            weights, coupling_fn(value_parent_state.expected_mean)
        )
    expected_mean = (
        params.autoconnection_strength * child_state.mean + time_step * driftrate
    )

    # 2. Total volatility, with the MGF correction κ²/(2 π̃) that treats the
    # volatility parent as a full Gaussian rather than a point estimate; the
    # mean-field scheme keeps the point estimate and drops the correction.
    total_volatility = params.tonic_volatility
    if volatility_parent_state is not None:
        assert volatility_weights is not None
        total_volatility = total_volatility + jnp.matmul(
            volatility_weights, volatility_parent_state.expected_mean
        )
        if not mean_field_updates:
            total_volatility = total_volatility + jnp.matmul(
                volatility_weights**2,
                1.0 / (2.0 * volatility_parent_state.expected_precision),
            )
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # 3. Laplace value-coupling variance: Σ_b (t · W_ab · g'(μ̂_b))² / π̃_b.
    # Absent under mean-field, where value parents enter at their means alone.
    value_coupling_variance = jnp.zeros_like(child_state.precision)
    if value_parent_state is not None and not mean_field_updates:
        assert weights is not None and coupling_fn is not None
        g_prime = vmap(grad(coupling_fn))(value_parent_state.expected_mean)
        value_coupling_variance = (time_step**2) * jnp.matmul(
            weights**2, g_prime**2 / value_parent_state.expected_precision
        )

    # 4. Conditional (π̂) and marginal (π̃) predicted precisions, effective
    # precision γ.
    conditional_expected_precision = 1.0 / (
        1.0 / child_state.precision + predicted_volatility
    )
    expected_precision = 1.0 / (
        1.0 / child_state.precision + predicted_volatility + value_coupling_variance
    )
    effective_precision = predicted_volatility * expected_precision

    # 5. Static-leaf override: a clamped leaf with no volatility parent takes no
    # random walk between observations — both predicted precisions stay at the
    # prior (the nodalised input-node convention).
    if is_static_leaf:
        expected_precision = child_state.precision
        conditional_expected_precision = child_state.precision

    return dataclasses.replace(
        child_state,
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        conditional_expected_precision=conditional_expected_precision,
        effective_precision=effective_precision,
    )
