# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised posterior update for volatile node layers."""

import dataclasses
from typing import Callable

import jax.numpy as jnp

from pyhgf.coupling import Coupling
from pyhgf.math import relaxed_edge_factors
from pyhgf.typing.vectorised import LayerState


def _value_messages(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_prime: jnp.ndarray,
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
    :func:`pyhgf.math.relaxed_edge_factors`, one per edge. The other parents
    :math:`m'` of the child :math:`a` add the variance
    :math:`(W_{a,m'} A_{m'})^2 / \hat{\pi}_{m'}` to its prediction, with
    :math:`A_{m'}` (``coupling_prime``) the slope of the linear function that
    replaces the coupling function in the prediction step: its derivative at the
    expected mean of the parent or, with the exact coupling moments, its mean slope
    under the predicted belief of the parent. Without ``integrate_other_parents``,
    every edge into the child :math:`a` uses the conditional predicted precision
    :math:`\bar{\pi}_a`, which treats the other parents of :math:`a` as known, and
    the sums are matrix products. Under ``mean_field_updates`` both factors are
    :math:`\hat{\pi}_a`.
    """
    if integrate_other_parents and not mean_field_updates:
        edge_variance = (weights * coupling_prime[None, :]) ** 2 / (
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
    coupling_moments: str = "taylor",
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

        \frac{\bar{\pi}_{a|b}^{(k)} \, \pi_y}{\bar{\pi}_{a|b}^{(k)} + \pi_y} \,
            \alpha^2 \, g'^2, \qquad
        \pi_y = \pi_a^{(k)} - \hat{\pi}_a^{(k)},

    with :math:`\alpha` the weight of the value-coupling edge and
    :math:`\hat{\pi}_a` the child's marginal predicted precision
    (``child.expected_precision``). :math:`\bar{\pi}_{a|b}` is the predicted
    precision of :math:`x_a` given :math:`x_b`. With ``integrate_other_parents``,
    it is the precision given :math:`x_b` alone, the other parents of :math:`a`
    integrated over their predicted beliefs
    (:func:`pyhgf.math.relaxed_edge_factors`). For linear coupling, this gives the
    exact marginal posterior of :math:`x_b` when no two children of :math:`b`
    share another parent; between two fully connected layers, where every child
    shares its parents with the others, it is an approximation. Without
    it, :math:`\bar{\pi}_{a|b}` is the conditional predicted precision
    :math:`\bar{\pi}_a` (``child.conditional_expected_precision``), which conditions
    on every parent of :math:`a`: the other parents are treated as known, as in a
    prediction step that does not propagate their uncertainty.

    The :math:`(\alpha g')^2` term is weighted by the relaxed child precision
    :math:`\bar{\pi}_{a|b} \pi_y / (\bar{\pi}_{a|b} + \pi_y)`. It reduces to
    :math:`\bar{\pi}_{a|b} \, \alpha^2 g'^2` when the child is fully observed
    (:math:`\pi_y \to \infty`), which is the canonical term when
    :math:`\hat{\pi}_a = \bar{\pi}_{a|b}`, and it vanishes when the child gained no
    bottom-up information (:math:`\pi_y = 0`).

    .. note::

        This update deviates from the generalised HGF [1]_, for deep networks only.
        The value-coupling precision update (Eq. 50 of [1]_) adds a second-order
        term to the contribution of each child,
        :math:`-\alpha \, g'' \, \hat{\pi}_a \, \delta_a`, which is left out here:
        the update is first-order in the coupling function, and the posterior
        precision is never below the predicted precision :math:`\hat{\pi}_b`. The
        second-order term takes either sign and can cancel the first-order term.
        The mean update of :func:`vectorised_posterior_update_mean_value_level`
        divides the message by the posterior precision, so the mean step then
        grows as :math:`1 / \hat{\pi}_b`, the volatility level of the layer reads
        it as volatility, and :math:`\hat{\pi}_b` decreases further at the next
        prediction. The continuous state nodes of :class:`pyhgf.model.Network` and
        the continuous layers of :mod:`pyhgf.updates.vectorised.continuous` keep
        the second-order term.

    With ``coupling_moments="exact"``, the coupling function is replaced by its
    statistical linearisation under the predicted belief
    :math:`x_b \sim \mathcal{N}(\hat{\mu}_b, 1/\hat{\pi}_b)` of the layer, as in
    :func:`~pyhgf.updates.vectorised.volatile.prediction.vectorised_layer_prediction`:
    :math:`g'` is the mean slope :math:`A = \mathbb{E}[g'(x_b)]`, so that each child
    contributes
    :math:`(\alpha A)^2 \, \bar{\pi}_{a|b} \pi_y / (\bar{\pi}_{a|b} + \pi_y)`, with
    the variance of the residuals of the linear functions counted in
    :math:`\bar{\pi}_a` and the mean slopes of the other parents in
    :math:`\bar{\pi}_{a|b}`. A linear function has no second-order term, so this
    update is first-order under both settings.

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
        Coupling function, with its first derivative read from its
        :class:`pyhgf.coupling.Coupling`.
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
    coupling_moments :
        ``"taylor"`` (default) linearises the coupling function with its derivatives
        at the expected mean of the layer. ``"exact"`` uses its statistical
        linearisation under the predicted belief of the layer. Not read with
        ``mean_field_updates``.
    integrate_other_parents :
        If ``True``, condition the child on each parent alone and integrate its other
        parents over their predicted beliefs (:math:`\bar{\pi}_{a|b}`); if ``False``
        (default), condition it on every parent (:math:`\bar{\pi}_a`). The network
        sets it when the prediction step propagates the parents' uncertainty
        (``feedforward_uncertainty`` and ``predict_precision``) and the child is
        Gaussian.

    Returns
    -------
    jnp.ndarray
        Posterior precision for each node in the parent layer.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    # Slope of the coupling function at the parent expected means: its derivative
    # or, with the exact moments, its mean slope under the predicted belief.
    _, coupling_prime, _ = Coupling.from_callable(coupling_fn).linearise(
        layer.expected_mean,
        1.0 / layer.expected_precision,
        "taylor" if mean_field_updates else coupling_moments,
    )

    # Σ_i w_ij² π^eff_ij. By the time this kernel runs for an interior child, the
    # child layer's bottom-up posterior update has populated `child.precision` (π_a);
    # the prediction-step values remain in `child.expected_precision` (π̂_a) and
    # `child.conditional_expected_precision` (π̄_a), so π_y = π_a − π̂_a is the
    # bottom-up evidence the parent is credited with. A clamped leaf keeps
    # π_a = π̂_a, so π_y = 0 and the interior form would zero out: it takes the fully
    # observed limit instead (see `_value_messages`).
    precision_sum, _ = _value_messages(
        layer,
        child,
        weights,
        coupling_prime,
        child_is_input_layer,
        mean_field_updates,
        integrate_other_parents,
    )

    # First-order term: Σ_i(w_ij² · π^eff_ij) · g'(m_j)². The second-order term of
    # the coupling function is left out (see the docstring).
    posterior_precision = layer.expected_precision + precision_sum * coupling_prime**2

    return posterior_precision


def vectorised_posterior_update_mean_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    posterior_precision: jnp.ndarray,
    mean_field_updates: bool = False,
    coupling_moments: str = "taylor",
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
    :math:`\pi_a` and the update is the canonical one. With
    ``coupling_moments="exact"``, :math:`g'(\hat{\mu}_b)` is replaced by the mean
    slope :math:`\mathbb{E}[g'(x_b)]` under the predicted belief
    :math:`x_b \sim \mathcal{N}(\hat{\mu}_b, 1/\hat{\pi}_b)` of the layer.

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
        Coupling function, with its first derivative read from its
        :class:`pyhgf.coupling.Coupling`.
    posterior_precision :
        Already-updated value-level posterior precision :math:`\pi_b` for the
        parent layer; the precision-weighted PE is divided by this once, after
        accumulating across children.
    mean_field_updates :
        If ``True``, replace the joint-Gaussian gain by the canonical predicted
        precision :math:`\hat{\pi}_a`, as in the mean-field update.
    coupling_moments :
        ``"taylor"`` (default) takes the derivative of the coupling function at the
        expected mean of the layer. ``"exact"`` takes its mean slope under the
        predicted belief of the layer. Not read with ``mean_field_updates``.
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
    # Slope of the coupling function at the parent expected means: its derivative
    # or, with the exact moments, its mean slope under the predicted belief.
    _, coupling_prime, _ = Coupling.from_callable(coupling_fn).linearise(
        layer.expected_mean,
        1.0 / layer.expected_precision,
        "taylor" if mean_field_updates else coupling_moments,
    )

    # Precision-weighted PE from children. From the joint Gaussian posterior over
    # (x_a, x_b), the per-child contribution is
    #     Δμ_b = α · g'(μ̂_b) · g_ab · δ_a / π_b,
    #     g_ab = π̄_{a|b} · π_a / (π̄_{a|b} + π_y),    π_y = π_a − π̂_a,
    # using PyHGF's precision-weighted PE δ_a = π_y · w / π_a (so the raw residual
    # w = y − μ̂_a substitutes out). With the prediction-step relaxation active,
    # π_a = π̂_a + π_y and g_ab is the joint-Gaussian-exact gain. The factor π_a/π_y
    # relative to the effective precision of the precision update reflects the
    # different element of P^{-1}h selected by the mean update versus the precision
    # update.
    _, message = _value_messages(
        layer,
        child,
        weights,
        coupling_prime,
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
    coupling_moments: str = "taylor",
    integrate_other_parents: bool = False,
) -> LayerState:
    """Update the value-level posterior for all nodes in a parent layer.

    Updates the value level precision first, then the mean.
    The precision update is first-order in the coupling function, a deviation from the
    generalised HGF that is specific to deep networks (see
    :func:`vectorised_posterior_update_precision_value_level`).

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
        Coupling function. The first derivative is read from its
        :class:`pyhgf.coupling.Coupling` in the two helpers.
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
    coupling_moments :
        ``"taylor"`` (default) or ``"exact"``: whether both helpers linearise the
        coupling function with its derivatives or with its statistical linearisation
        under the predicted belief of the layer.
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
            coupling_moments=coupling_moments,
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
        coupling_moments=coupling_moments,
        child_is_input_layer=child_is_input_layer,
        integrate_other_parents=integrate_other_parents,
    )

    return dataclasses.replace(
        layer,
        precision=posterior_precision,
        mean=posterior_mean,
    )
