# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised posterior update for volatile node layers."""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.math import relaxed_edge_factors
from pyhgf.typing.vectorised import LayerState


def _value_messages(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    child_is_input_layer: bool,
    mean_field_updates: bool,
    integrate_other_parents: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Weighted sums of the child's precisions and prediction errors, per parent.

    Returns ``(precision_sum, message)``, both of shape ``(n_parent,)``:

    .. math::

        \sum_a W_{a,m}^2 \, \pi^{\mathrm{eff}}_{a|m},
        \qquad
        \sum_a W_{a,m} \, g_{a|m} \, \delta_a.

    With ``integrate_other_parents``, the factors are those of
    :func:`pyhgf.math.relaxed_edge_factors`, one per edge. Without it, every edge into the child
    :math:`a` uses the conditional predicted precision :math:`\bar{\pi}_a`, which
    treats the other parents of :math:`a` as known, and the sums are matrix products.
    Under ``mean_field_updates`` both factors are :math:`\hat{\pi}_a`.
    """
    if integrate_other_parents and not mean_field_updates:
        g_prime = vmap(jgrad(coupling_fn))(layer.expected_mean)
        edge_variance = (weights * g_prime[None, :]) ** 2 / (
            layer.expected_precision[None, :]
        )
        effective_precision, gain = relaxed_edge_factors(
            child.conditional_expected_precision,
            child.expected_precision,
            child.precision,
            edge_variance,
            child_is_input_layer,
        )
        precision_sum = jnp.sum(weights**2 * effective_precision, axis=0)
        message = jnp.sum(
            weights * gain * child.value_prediction_error[:, None], axis=0
        )
        return precision_sum, message

    if mean_field_updates:
        effective_child_precision = child.expected_precision
        gain_precision = child.expected_precision
    else:
        pi_y = child.precision - child.expected_precision
        denominator = child.conditional_expected_precision + pi_y
        if child_is_input_layer:
            effective_child_precision = child.conditional_expected_precision
        else:
            effective_child_precision = (
                child.conditional_expected_precision * pi_y / denominator
            )
        gain_precision = (
            child.conditional_expected_precision * child.precision / denominator
        )
    precision_sum = jnp.matmul(weights.T**2, effective_child_precision)
    message = jnp.matmul(weights.T, gain_precision * child.value_prediction_error)
    return precision_sum, message


def vectorised_posterior_update_precision_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
    integrate_other_parents: bool = False,
) -> jnp.ndarray:
    r"""Update the precision of the value level for all nodes in a layer.

    Implements the *posterior-step relaxation* of the relaxed HGF (rHGF) on
    value-coupling edges, which pairs with the prediction-step relaxation (the
    marginal predicted precision). Lifting the mean-field
    delta-collapse approximation :math:`q(x_a, x_b) = q(x_a)\,q(x_b)` to a joint
    Gaussian on the value-coupling edge and applying the Schur complement to the joint
    precision matrix gives the value-coupling contribution of each child :math:`a`
    to :math:`\pi_b^{(k)}`:

    .. math::

        \frac{\bar{\pi}_{a|b}^{(k)}}{\bar{\pi}_{a|b}^{(k)} + \pi_y} \left(
            \alpha^2 \, g'^2 \, \pi_y - \alpha \, g'' \, \pi_a^{(k)} \, \delta_a
            \right), \qquad
        \pi_y = \pi_a^{(k)} - \hat{\pi}_a^{(k)},

    with :math:`\alpha` the weight of the value-coupling edge and
    :math:`\hat{\pi}_a` the child's marginal predicted precision
    (``child.expected_precision``). :math:`\bar{\pi}_{a|b}` is the predicted
    precision of :math:`x_a` given :math:`x_b`. With ``integrate_other_parents``,
    it is the precision given :math:`x_b` alone, the other parents of :math:`a`
    integrated over their predicted beliefs (:func:`pyhgf.math.relaxed_edge_factors`), which
    gives the exact marginal posterior of :math:`x_b` for linear coupling. Without
    it, :math:`\bar{\pi}_{a|b}` is the conditional predicted precision
    :math:`\bar{\pi}_a` (``child.conditional_expected_precision``), which conditions
    on every parent of :math:`a`: the other parents are treated as known, as in a
    prediction step that does not propagate their uncertainty.

    The :math:`(\alpha g')^2` term is weighted by the relaxed child precision
    :math:`\bar{\pi}_{a|b} \pi_y / (\bar{\pi}_{a|b} + \pi_y)` and the
    :math:`\alpha g'' \, \delta_a` term by the relaxed gain
    :math:`g_{a|b} = \bar{\pi}_{a|b} \pi_a / (\bar{\pi}_{a|b} + \pi_y)` of
    :func:`vectorised_posterior_update_mean_value_level`: the second-order term is
    the derivative of the mean shift's :math:`g'` with respect to the parent mean.
    Reduces to the canonical formula when the child is fully observed
    (:math:`\pi_y \to \infty`). When :math:`\hat{\pi}_a = \bar{\pi}_a`, the relaxed
    gain equals :math:`\hat{\pi}_a` and the :math:`g''` term is the canonical one.
    The :math:`(\alpha g')^2` term vanishes when the child gained no bottom-up
    information (:math:`\pi_y = 0`), and so does :math:`\delta_a`.

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
        output). The fully observed limit (:math:`\pi_y \to \infty`) applies and
        the posterior-step relaxation reduces to the conditional predicted
        precision :math:`\bar{\pi}_a`. We short-circuit to
        ``child.conditional_expected_precision`` (which equals
        ``child.expected_precision`` for a leaf): pyhgf's leaf convention sets
        :math:`\pi_a = \hat{\pi}_a` for the binary PE and never updates
        :math:`\pi_a` for a continuous leaf, so :math:`\pi_y = 0` and the
        harmonic-combination form would incorrectly zero out the contribution.
        Defaults to ``False`` (interior child, full posterior-step relaxation).
    mean_field_updates :
        If ``True``, skip the posterior-step relaxation entirely and weight every
        child contribution by the canonical predicted precision
        :math:`\hat{\pi}_a`, as in the mean-field update.
    integrate_other_parents :
        If ``True``, condition the child on each parent alone and integrate its other
        parents over their predicted beliefs (:math:`\bar{\pi}_{a|b}`); if ``False``
        (default), condition it on every parent (:math:`\bar{\pi}_a`). The network
        sets it when the prediction step propagates the parents' uncertainty
        (``feedforward_uncertainty``) and the child is Gaussian.

    Returns
    -------
    jnp.ndarray
        Posterior precision for each node in the parent layer.
    """
    coupling_fn_grad = jgrad(coupling_fn)

    # Coupling derivatives at parent expected means
    coupling_prime = vmap(coupling_fn_grad)(layer.expected_mean)

    # Coupling second derivative (for the second-order Taylor term)
    coupling_second = vmap(jgrad(coupling_fn_grad))(layer.expected_mean)

    # Σ_i w_ij² π^eff_ij and Σ_i w_ij g_ij δ_i. By the time this kernel runs for an
    # interior child, the child layer's bottom-up posterior update has populated
    # `child.precision` (π_a); the prediction-step values remain in
    # `child.expected_precision` (π̂_a) and `child.conditional_expected_precision`
    # (π̄_a), so π_y = π_a − π̂_a is the bottom-up evidence the parent is credited
    # with. A clamped leaf keeps π_a = π̂_a, so π_y = 0 and the interior form would
    # zero out: it takes the fully observed limit instead (see `_value_messages`).
    precision_sum, message = _value_messages(
        layer,
        child,
        weights,
        coupling_fn,
        child_is_input_layer,
        mean_field_updates,
        integrate_other_parents,
    )

    # First-order term: Σ_i(w_ij² · π^eff_ij) · g'(m_j)²
    precision_contrib_1 = precision_sum * coupling_prime**2

    # Second-order Taylor term: -g''(m_j) · Σ_i(w_ij · g_ij · δ_i).
    precision_contrib_2 = -coupling_second * message

    posterior_precision = (
        layer.expected_precision + precision_contrib_1 + precision_contrib_2
    )

    return posterior_precision


def vectorised_posterior_update_mean_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    posterior_precision: jnp.ndarray,
    mean_field_updates: bool = False,
    child_is_input_layer: bool = False,
    integrate_other_parents: bool = False,
) -> jnp.ndarray:
    r"""Update the mean of the value level for all nodes in a layer.

    Uses the relaxed gain of the posterior-step relaxation (the joint-Gaussian
    Rauch-Tung-Striebel smoother gain). Each value child contributes

    .. math::

        \Delta \mu_b \mathrel{+}= \frac{\alpha \, g'(\hat{\mu}_b) \, g_{a|b}}
            {\pi_b} \, \delta_a, \qquad
        g_{a|b} = \frac{\bar{\pi}_{a|b} \, \pi_a}{\bar{\pi}_{a|b} + \pi_y}, \qquad
        \pi_y = \pi_a - \hat{\pi}_a,

    where :math:`\alpha` is the weight of the value-coupling edge,
    :math:`\pi_b` is the just-updated parent posterior precision
    (``posterior_precision``), and :math:`\bar{\pi}_{a|b}` is the predicted
    precision of :math:`x_a` given :math:`x_b` (see
    :func:`vectorised_posterior_update_precision_value_level`); contributions
    accumulate across children and are divided by :math:`\pi_b` once, after
    summation. A clamped leaf takes the fully observed limit
    :math:`g_{a|b} = \bar{\pi}_{a|b}`. Without ``integrate_other_parents``,
    :math:`\bar{\pi}_{a|b} = \bar{\pi}_a = \pi_a` for a leaf, so the gain is
    :math:`\pi_a` and the update is the canonical one.

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
    mean_field_updates :
        If ``True``, replace the joint-Gaussian gain by the canonical predicted
        precision :math:`\hat{\pi}_a`, as in the mean-field update.
    child_is_input_layer :
        If True, the child is the clamped observation leaf, which takes the fully
        observed limit of the gain. Defaults to ``False``.
    integrate_other_parents :
        If ``True``, condition the child on each parent alone and integrate its other
        parents over their predicted beliefs; see
        :func:`vectorised_posterior_update_precision_value_level`. Defaults to
        ``False``.

    Returns
    -------
    jnp.ndarray
        Posterior mean for each node in the parent layer.
    """
    # Coupling derivatives at parent expected means
    coupling_prime = vmap(jgrad(coupling_fn))(layer.expected_mean)

    # Precision-weighted PE from children. From the joint Gaussian posterior over
    # (x_a, x_b), the per-child contribution is
    #     Δμ_b = α · g'(μ̂_b) · g_ab · δ_a / π_b,
    #     g_ab = π̄_{a|b} · π_a / (π̄_{a|b} + π_y),    π_y = π_a − π̂_a,
    # using PyHGF's precision-weighted PE δ_a = π_y · w / π_a (so the raw residual
    # w = y − μ̂_a substitutes out). With the prediction-step relaxation active,
    # π_a = π̂_a + π_y and g_ab is the joint-Gaussian-exact gain. The factor π_a/π_y
    # relative to the effective precision of the precision update's first-order term
    # reflects the different element of P^{-1}h selected by the mean update versus
    # the precision update; the precision update's g'' term uses g_ab itself.
    _, message = _value_messages(
        layer,
        child,
        weights,
        coupling_fn,
        child_is_input_layer,
        mean_field_updates,
        integrate_other_parents,
    )
    weighted_pe = message * coupling_prime / posterior_precision

    posterior_mean = layer.expected_mean + weighted_pe

    return posterior_mean


def vectorised_layer_posterior_update(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
    max_posterior_precision: float = 1e10,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
    integrate_other_parents: bool = False,
) -> LayerState:
    """Update the value-level posterior for all nodes in a parent layer.

    Updates the value level precision first, then the mean.

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
        continuous output). The posterior-step relaxation reduces to its fully
        observed limit. Defaults to ``False``.
    mean_field_updates :
        If ``True``, both helpers use the mean-field forms (canonical child
        predicted precision, no posterior-step relaxation).
    integrate_other_parents :
        If ``True``, both helpers condition the child on each parent alone and
        integrate its other parents over their predicted beliefs; see
        :func:`vectorised_posterior_update_precision_value_level`. Defaults to
        ``False``.

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
        vectorised_posterior_update_precision_value_level(
            layer,
            child,
            weights,
            coupling_fn,
            child_is_input_layer=child_is_input_layer,
            mean_field_updates=mean_field_updates,
            integrate_other_parents=integrate_other_parents,
        ),
        layer.expected_precision,
        max_posterior_precision,
    )

    posterior_mean = vectorised_posterior_update_mean_value_level(
        layer,
        child,
        weights,
        coupling_fn,
        posterior_precision,
        mean_field_updates=mean_field_updates,
        child_is_input_layer=child_is_input_layer,
        integrate_other_parents=integrate_other_parents,
    )

    return dataclasses.replace(
        layer,
        precision=posterior_precision,
        mean=posterior_mean,
    )
