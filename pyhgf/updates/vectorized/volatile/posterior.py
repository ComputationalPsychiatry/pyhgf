# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized posterior update for volatile node layers."""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing.vectorised import LayerState


def vectorized_posterior_update_precision_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    child_is_input_layer: bool = False,
) -> jnp.ndarray:
    r"""Update the precision of the value level for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.posterior_update_value_level.posterior_update_precision_value_level`.

    Implements the *posterior-step (smoothing) correction* of the relaxed HGF on
    value-coupling edges, in its **fully-corrected form** that pairs with the
    prediction-step (marginal-precision) correction. Lifting the mean-field
    delta-collapse approximation :math:`q(x_a, x_b) = q(x_a)\,q(x_b)` to a structured
    Gaussian on the value-coupling edge and applying the Schur complement to the joint
    precision matrix replaces the canonical child-precision factor by the harmonic
    combination

    .. math::

        \hat{\pi}_a^{(k)} \,\longmapsto\,
        \frac{\hat{\pi}_a^{(k)} \, \pi_y}{\hat{\pi}_a^{(k)} + \pi_y}, \qquad
        \pi_y = \pi_a^{(k)} - \tilde{\pi}_a^{(k)},

    in the value-coupling contribution to :math:`\pi_b^{(k)}`. Here
    :math:`\hat{\pi}_a` is the child's *conditional* predicted precision (own variance
    plus volatility, without the parent-uncertainty bleed-through term
    :math:`\alpha^2 g'^2 / \hat{\pi}_b`), stored as
    ``child.conditional_expected_precision``, and :math:`\tilde{\pi}_a` is its
    *marginal* predicted precision stored as ``child.expected_precision``. The two
    coincide in the canonical limit (no prediction-step correction); once the
    prediction-step correction is in play they diverge, and the Schur complement
    acts on the conditional. Substituting the marginal here would double-count
    parent uncertainty.

    The same harmonic combination scales both the :math:`(\kappa g')^2` and the
    :math:`\kappa g'' \, \delta_a` contributions. Reduces to the canonical formula
    when the child is fully observed (:math:`\pi_y \to \infty`); returns no
    contribution when the child gained no bottom-up information
    (:math:`\pi_y = 0`).

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the toolbox,
        the volatile-state updates evaluate coupling function derivatives at the
        *expected* mean (i.e. the prediction) rather than the posterior mean. This
        choice is made to better suit deep learning networks where the prediction serves
        as the natural reference point for computing updates.

    Parameters
    ----------
    layer :
        Current state of the parent layer (being updated).
    child :
        Current state of the child layer (providing prediction errors).
    weights :
        Weight matrix connecting child to parent, shape
        ``(n_children, n_parents)``.
    coupling_fn :
        Coupling function. First and second derivatives are computed inline
        via ``jax.grad``.
    child_is_input_layer :
        If True, the child is a clamped observation leaf (binary or continuous
        output). The paper's Limit 3 (:math:`\pi_a \to \infty`) applies and the
        smoothing correction reduces to the canonical predicted-precision factor.
        We short-circuit to ``child.conditional_expected_precision`` (which equals
        ``child.expected_precision`` for a leaf): pyhgf's leaf convention sets
        :math:`\pi_a = \tilde{\pi}_a` for the binary PE and never updates
        :math:`\pi_a` for a continuous leaf, so :math:`\pi_y = 0` and the
        harmonic-combination form would incorrectly zero out the contribution.
        Defaults to ``False`` (interior child, full smoothing correction).

    Returns
    -------
    jnp.ndarray
        Posterior precision for each node in the parent layer.
    """
    coupling_fn_grad = jgrad(coupling_fn)

    # Coupling derivatives at parent expected means
    coupling_prime = vmap(coupling_fn_grad)(layer.expected_mean)

    # Coupling second derivative (for second-order EKF correction)
    coupling_second = vmap(jgrad(coupling_fn_grad))(layer.expected_mean)

    # Effective child precision under the smoothing correction:
    #     π̂_a · π_y / (π̂_a + π_y),    π_y = π_a − π̃_a.
    # By the time this kernel runs for an interior child, the child layer's
    # bottom-up posterior update has already populated `child.precision` (π_a);
    # the marginal prediction-step value remains in `child.expected_precision`
    # (π̃_a) and the conditional prediction-step value in
    # `child.conditional_expected_precision` (π̂_a). The difference π_a − π̃_a
    # is the fresh bottom-up evidence the parent should be credited with, while
    # the conditional is the precision that appears in the joint (x_a, x_b)
    # precision matrix the Schur complement acts on. Using the marginal in both
    # numerator and denominator would double-count parent uncertainty.
    #
    # For a clamped leaf, pyhgf's convention sets π_a = π̃_a (= π̂_a, since
    # leaves carry no value-coupling bleed-through) so π_y = 0 and the harmonic
    # form would zero out. Short-circuit to the canonical predicted-precision
    # factor, matching the paper's Limit 3 (π_a → ∞).
    if child_is_input_layer:
        effective_child_precision = child.conditional_expected_precision
    else:
        pi_y = child.precision - child.expected_precision
        effective_child_precision = (
            child.conditional_expected_precision
            * pi_y
            / (child.conditional_expected_precision + pi_y)
        )

    # First-order term: Σ_i(w_ij² · effective_child_prec_i) · g'(m_j)²
    precision_contrib_1 = jnp.matmul(weights.T**2, effective_child_precision) * (
        coupling_prime**2
    )

    # Second-order EKF correction: -g''(m_j) · Σ_i(w_ij · effective_child_prec_i · δ_a_i).
    # The same effective child precision (harmonic combination above) scales the g''δ_a
    # term in eq. 50 of Weber et al. (2026), per the artifact §4.
    sum_pi_vpe = jnp.matmul(
        weights.T, effective_child_precision * child.value_prediction_error
    )
    precision_contrib_2 = -coupling_second * sum_pi_vpe

    posterior_precision = (
        layer.expected_precision + precision_contrib_1 + precision_contrib_2
    )

    return posterior_precision


def vectorized_posterior_update_mean_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    posterior_precision: jnp.ndarray,
) -> jnp.ndarray:
    r"""Update the mean of the value level for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.posterior_update_value_level.posterior_update_mean_value_level`.

    Uses the joint-Gaussian (RTS-smoother) gain. Each value child contributes

    .. math::

        \Delta \mu_b \mathrel{+}= \frac{\kappa \, g'(\hat{\mu}_b) \, g_a}
            {\pi_b} \, \delta_a, \qquad
        g_a = \frac{\hat{\pi}_a \, \pi_a}{\hat{\pi}_a + \pi_y}, \qquad
        \pi_y = \pi_a - \tilde{\pi}_a,

    where :math:`\pi_b` is the just-updated parent posterior precision
    (``posterior_precision``); contributions accumulate across children and are
    divided by :math:`\pi_b` once, after summation. For leaves :math:`\pi_y = 0`
    and :math:`g_a` collapses to :math:`\tilde{\pi}_a`, recovering the canonical
    gain.

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.

    Parameters
    ----------
    layer :
        Current state of the parent layer (being updated).
    child :
        Current state of the child layer (providing prediction errors).
    weights :
        Weight matrix connecting child to parent, shape
        ``(n_children, n_parents)``.
    coupling_fn :
        Coupling function. The first derivative is computed inline via
        ``jax.grad``.
    posterior_precision :
        Already-updated value-level posterior precision :math:`\pi_b` for the
        parent layer; the precision-weighted PE is divided by this once, after
        accumulating across children.

    Returns
    -------
    jnp.ndarray
        Posterior mean for each node in the parent layer.
    """
    # Coupling derivatives at parent expected means
    coupling_prime = vmap(jgrad(coupling_fn))(layer.expected_mean)

    # Precision-weighted PE from children. From the joint Gaussian posterior over
    # (x_a, x_b), the per-child contribution is
    #     Δμ_b = κ · g'(μ̂_b) · g_a · δ_a / π_b,
    #     g_a  = π̂_a · π_a / (π̂_a + π_y),    π_y = π_a − π̃_a,
    # using PyHGF's precision-weighted PE δ_a = π_y · w / π_a (so the raw residual
    # w = y − μ̂_a substitutes out). The artifact's "Step 5" identity
    # π_a = π̂_a + π_y collapses g_a → π̂_a only in the canonical scheme
    # (π̃_a = π̂_a); with the prediction-step correction active π_a = π̃_a + π_y
    # and g_a above is the joint-Gaussian-exact gain. The factor π_a/π_y relative
    # to the precision update's `effective_child_precision` reflects the different
    # element of P^{-1}h selected by the mean update versus the precision update.
    # For leaves PyHGF keeps π_a = π̃_a, so π_y = 0 and g_a → π̂_a, matching the
    # canonical leaf treatment used by the precision update.
    pi_y = child.precision - child.expected_precision
    gain_precision = (
        child.conditional_expected_precision
        * child.precision
        / (child.conditional_expected_precision + pi_y)
    )
    weighted_pe = (
        jnp.matmul(weights.T, gain_precision * child.value_prediction_error)
        * coupling_prime
        / posterior_precision
    )

    posterior_mean = layer.expected_mean + weighted_pe

    return posterior_mean


def vectorized_layer_posterior_update(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
    max_posterior_precision: float = 1e10,
    child_is_input_layer: bool = False,
) -> LayerState:
    """Update the value-level posterior for all nodes in a parent layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.volatile_node_posterior_update.volatile_node_posterior_update`.
    It updates the value level precision first, then the mean.

    Parameters
    ----------
    layer :
        Current state of the parent layer (being updated).
    child :
        Current state of the child layer (providing prediction errors).
    weights :
        Weight matrix connecting child to parent, shape
        ``(n_children, n_parents)`` or ``(n_children, n_parents + 1)``
        when the parent layer includes a constant input node.
    coupling_fn :
        Coupling function. The first (and second, for the precision update)
        derivatives are computed inline via ``jax.grad`` in the two helpers.
    parent_has_constant :
        If True, the last column of *weights* corresponds to the constant input node and
        is stripped before computing the posterior update.
    max_posterior_precision :
        Upper bound applied to the posterior precision. Default ``1e10``.
    child_is_input_layer :
        If True, the child is the clamped observation leaf (binary or
        continuous output). The smoothing correction reduces to the
        canonical contribution (paper's Limit 3). Defaults to ``False``.

    Returns
    -------
    LayerState
        Updated parent layer state with posterior mean and precision.
    """
    # Strip the bias column if the parent has a constant input node
    if parent_has_constant:
        weights = weights[:, :-1]

    # Update precision first, then mean
    posterior_precision = jnp.clip(
        vectorized_posterior_update_precision_value_level(
            layer,
            child,
            weights,
            coupling_fn,
            child_is_input_layer=child_is_input_layer,
        ),
        a_min=layer.expected_precision,
        a_max=max_posterior_precision,
    )

    posterior_mean = vectorized_posterior_update_mean_value_level(
        layer, child, weights, coupling_fn, posterior_precision
    )

    return dataclasses.replace(
        layer,
        precision=posterior_precision,
        mean=posterior_mean,
    )
