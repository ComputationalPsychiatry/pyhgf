# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised posterior update for volatile node layers."""

import dataclasses
from typing import Callable

import jax.numpy as jnp

from pyhgf.coupling import Coupling
from pyhgf.typing.vectorised import LayerState


def vectorised_posterior_update_precision_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
    coupling_moments: str = "taylor",
) -> jnp.ndarray:
    r"""Update the precision of the value level for all nodes in a layer.

    Implements the *posterior-step relaxation* of the relaxed HGF (rHGF) on
    value-coupling edges, which pairs with the prediction-step relaxation (the
    marginal predicted precision). Lifting the mean-field
    delta-collapse approximation :math:`q(x_a, x_b) = q(x_a)\,q(x_b)` to a structured
    Gaussian on the value-coupling edge and applying the Schur complement to the joint
    precision matrix gives the value-coupling contribution of each child :math:`a`
    to :math:`\pi_b^{(k)}`:

    .. math::

        \frac{\bar{\pi}_a^{(k)} \, \pi_y}{\bar{\pi}_a^{(k)} + \pi_y} \,
            \alpha^2 \, g'^2, \qquad
        \pi_y = \pi_a^{(k)} - \hat{\pi}_a^{(k)},

    with :math:`\alpha` the weight of the value-coupling edge. Here
    :math:`\bar{\pi}_a` is the child's *conditional* predicted precision (own variance
    plus volatility, without the parent-uncertainty bleed-through term
    :math:`\alpha^2 g'^2 / \hat{\pi}_b`), stored as
    ``child.conditional_expected_precision``, and :math:`\hat{\pi}_a` is its
    *marginal* predicted precision stored as ``child.expected_precision``. The two
    coincide in the canonical limit (no prediction-step relaxation); once the
    prediction-step relaxation is in play they diverge, and the Schur complement
    acts on the conditional. Substituting the marginal here would double-count
    parent uncertainty.

    The :math:`(\alpha g')^2` term is weighted by the relaxed child precision
    :math:`\bar{\pi}_a \pi_y / (\bar{\pi}_a + \pi_y)`. It reduces to
    :math:`\bar{\pi}_a \, \alpha^2 g'^2` when the child is fully observed
    (:math:`\pi_y \to \infty`), which is the canonical term when
    :math:`\hat{\pi}_a = \bar{\pi}_a`, and it vanishes when the child gained no
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
    contributes :math:`(\alpha A)^2 \, \bar{\pi}_a \pi_y / (\bar{\pi}_a + \pi_y)`,
    with the variance of the residual of the linear function counted in
    :math:`\bar{\pi}_a`. A linear function has no second-order term, so this update
    is first-order under both settings.

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
    coupling = Coupling.from_callable(coupling_fn)
    if coupling_moments == "exact" and not mean_field_updates:
        # Statistical linearisation: the mean slope E[g'] under the layer's predicted
        # belief.
        _, _, coupling_prime = coupling.moments(
            layer.expected_mean, 1.0 / layer.expected_precision
        )
    else:
        # Coupling derivatives at parent expected means
        coupling_prime = coupling.df(layer.expected_mean)

    # Effective child precision under the posterior-step relaxation:
    #     π̄_a · π_y / (π̄_a + π_y),    π_y = π_a − π̂_a.
    # By the time this kernel runs for an interior child, the child layer's
    # bottom-up posterior update has already populated `child.precision` (π_a);
    # the marginal prediction-step value remains in `child.expected_precision`
    # (π̂_a) and the conditional prediction-step value in
    # `child.conditional_expected_precision` (π̄_a). The difference π_a − π̂_a
    # is the fresh bottom-up evidence the parent should be credited with, while
    # the conditional is the precision that appears in the joint (x_a, x_b)
    # precision matrix the Schur complement acts on. Using the marginal in both
    # numerator and denominator would double-count parent uncertainty.
    #
    # For a clamped leaf, pyhgf's convention sets π_a = π̂_a (= π̄_a, since
    # leaves carry no value-coupling bleed-through) so π_y = 0 and the harmonic
    # form would zero out. Short-circuit to the conditional predicted precision
    # π̄_a, the fully observed limit (π_y → ∞) of the harmonic form.
    if mean_field_updates:
        # Mean-field: canonical predicted precision for every child, leaf or
        # interior (a leaf's marginal equals its conditional anyway).
        effective_child_precision = child.expected_precision
    elif child_is_input_layer:
        effective_child_precision = child.conditional_expected_precision
    else:
        pi_y = child.precision - child.expected_precision
        effective_child_precision = (
            child.conditional_expected_precision
            * pi_y
            / (child.conditional_expected_precision + pi_y)
        )

    # First-order term: Σ_i(w_ij² · effective_child_prec_i) · g'(m_j)². The
    # second-order term of the coupling function is left out (see the docstring).
    posterior_precision = layer.expected_precision + jnp.matmul(
        weights.T**2, effective_child_precision
    ) * (coupling_prime**2)

    return posterior_precision


def vectorised_posterior_update_mean_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    posterior_precision: jnp.ndarray,
    mean_field_updates: bool = False,
    coupling_moments: str = "taylor",
) -> jnp.ndarray:
    r"""Update the mean of the value level for all nodes in a layer.

    Uses the relaxed gain of the posterior-step relaxation (the joint-Gaussian
    Rauch-Tung-Striebel smoother gain). Each value child contributes

    .. math::

        \Delta \mu_b \mathrel{+}= \frac{\alpha \, g'(\hat{\mu}_b) \, g_a}
            {\pi_b} \, \delta_a, \qquad
        g_a = \frac{\bar{\pi}_a \, \pi_a}{\bar{\pi}_a + \pi_y}, \qquad
        \pi_y = \pi_a - \hat{\pi}_a,

    where :math:`\alpha` is the weight of the value-coupling edge,
    :math:`\pi_b` is the just-updated parent posterior precision
    (``posterior_precision``); contributions accumulate across children and are
    divided by :math:`\pi_b` once, after summation. For leaves :math:`\pi_y = 0`
    and :math:`g_a` collapses to :math:`\hat{\pi}_a`, recovering the canonical
    gain. With ``coupling_moments="exact"``, :math:`g'(\hat{\mu}_b)` is replaced by
    the mean slope :math:`\mathbb{E}[g'(x_b)]` under the predicted belief
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

    Returns
    -------
    jnp.ndarray
        Posterior mean for each node in the parent layer.
    """
    coupling = Coupling.from_callable(coupling_fn)
    if coupling_moments == "exact" and not mean_field_updates:
        # Mean slope E[g'] under the layer's predicted belief
        _, _, coupling_prime = coupling.moments(
            layer.expected_mean, 1.0 / layer.expected_precision
        )
    else:
        # Coupling derivatives at parent expected means
        coupling_prime = coupling.df(layer.expected_mean)

    # Precision-weighted PE from children. From the joint Gaussian posterior over
    # (x_a, x_b), the per-child contribution is
    #     Δμ_b = α · g'(μ̂_b) · g_a · δ_a / π_b,
    #     g_a  = π̄_a · π_a / (π̄_a + π_y),    π_y = π_a − π̂_a,
    # using PyHGF's precision-weighted PE δ_a = π_y · w / π_a (so the raw residual
    # w = y − μ̂_a substitutes out). The identity π_a = π̄_a + π_y, which collapses
    # g_a to π̄_a, holds only in the canonical scheme (π̂_a = π̄_a); with the
    # prediction-step relaxation active π_a = π̂_a + π_y and g_a above is the
    # joint-Gaussian-exact gain. The factor π_a/π_y relative to the
    # `effective_child_precision` of the precision update reflects the different
    # element of P^{-1}h selected by the mean update versus the precision update.
    # For leaves PyHGF keeps π_a = π̂_a, so π_y = 0 and g_a → π_a = π̂_a, matching
    # the canonical leaf treatment used by the precision update.
    if mean_field_updates:
        gain_precision = child.expected_precision
    else:
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
    )

    return dataclasses.replace(
        layer,
        precision=posterior_precision,
        mean=posterior_mean,
    )
