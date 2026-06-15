# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array, grad, jit
from jax.lax import cond
from jax.tree_util import Partial

from pyhgf.typing import Edges

# ----------------------------------------------------------------------------------
# Shared building blocks
#
# The four "observed value" precision updates are the cross-product of two orthogonal
# choices: the value-coupling factor (smoothing/relaxed vs. mean-field) and the
# volatility-coupling increment (standard γ-formula vs. enhanced-HGF safe update). The
# helpers below isolate each piece so the leaf updates are one-line compositions and
# the routing functions share the observation check.
# ----------------------------------------------------------------------------------


def _has_observations(attributes: dict, edges: Edges, node_idx: int) -> Array:
    """Return whether any value or volatility child reported an observation.

    If all children are unobserved the node simply ages its precision via
    :func:`precision_update_missing_values`; otherwise the regular prediction-error
    integration applies.
    """
    observations = []
    if edges[node_idx].value_children is not None:
        for children_idx in edges[node_idx].value_children:  # type: ignore
            observations.append(attributes[children_idx]["observed"])
    if edges[node_idx].volatility_children is not None:
        for children_idx in edges[node_idx].volatility_children:  # type: ignore
            observations.append(attributes[children_idx]["observed"])
    return jnp.any(jnp.array(observations))


def _smoothing_child_precision(
    attributes: dict, edges: Edges, value_child_idx: int
) -> Array:
    r"""Effective child precision under the relaxed posterior-step (smoothing) update.

    The child-precision factor :math:`\hat{\pi}_a` is replaced by the harmonic
    combination :math:`\hat{\pi}_a \pi_y / (\hat{\pi}_a + \pi_y)`, with
    :math:`\pi_y = \pi_a - \tilde{\pi}_a`, but only for Gaussian *interior* children
    (node types 2/6 that themselves have children). The Schur complement carries the
    *conditional* predicted precision :math:`\hat{\pi}_a` (stored in ``temp``); using
    the marginal would double-count parent uncertainty. All other cases — binary
    (type 1), categorical (5), input/constant (0), exponential family (3), Dirichlet
    (4), and any Gaussian leaf (types 2/6 with no children) — fall back to the
    canonical predicted-precision factor :math:`\tilde{\pi}_a` (:math:`\pi_a \to \infty`).
    """
    child_node_type = edges[value_child_idx].node_type
    child_is_gaussian_interior = child_node_type in (2, 6) and (
        edges[value_child_idx].value_children is not None
        or edges[value_child_idx].volatility_children is not None
    )
    child_expected_precision = attributes[value_child_idx]["expected_precision"]
    if child_is_gaussian_interior:
        # π_y = π_a − π̃_a (bottom-up evidence precision, against the marginal).
        child_precision = attributes[value_child_idx]["precision"]
        pi_y = child_precision - child_expected_precision
        child_cond = attributes[value_child_idx]["temp"][
            "conditional_expected_precision"
        ]
        return child_cond * pi_y / (child_cond + pi_y)
    return child_expected_precision


def _mean_field_child_precision(
    attributes: dict, edges: Edges, value_child_idx: int
) -> Array:
    r"""Effective child precision under the mean-field update (no smoothing correction).

    Always the canonical predicted precision :math:`\tilde{\pi}_a`.
    """
    return attributes[value_child_idx]["expected_precision"]


def _value_coupling_pwpe(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    effective_precision_fn: Callable[[dict, Edges, int], Array],
) -> Array:
    r"""Precision-weighted prediction error from value-coupled children.

    Implements eq. 50 of [1]_: each child contributes
    :math:`\pi_a (\kappa^2 g'^2 - \kappa g'' \delta_a)`, with the child-precision
    factor :math:`\pi_a` supplied by ``effective_precision_fn`` (smoothing or
    mean-field). The :math:`g'`/:math:`g''` derivatives reduce to the linear case
    (:math:`\kappa^2`, ``0``) when no coupling function is set.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    pwpe = 0.0
    if edges[node_idx].value_children is None:
        return pwpe

    for value_child_idx, value_coupling, coupling_fn in zip(
        edges[node_idx].value_children,  # type: ignore
        attributes[node_idx]["value_coupling_children"],
        edges[node_idx].coupling_fn,
    ):
        if coupling_fn is None:  # linear coupling
            coupling_fn_prime = value_coupling**2
            coupling_fn_second = 0
        else:  # non-linear coupling
            coupling_fn_prime = (
                value_coupling**2 * grad(coupling_fn)(attributes[node_idx]["mean"]) ** 2
            )
            value_prediction_error = attributes[value_child_idx]["temp"][
                "value_prediction_error"
            ]
            coupling_fn_second = (
                value_coupling
                * grad(grad(coupling_fn))(attributes[node_idx]["mean"])
                * value_prediction_error
            )

        effective_child_precision = effective_precision_fn(
            attributes, edges, value_child_idx
        )

        # cancel the prediction error if the child value was not observed
        pwpe += (
            effective_child_precision
            * (coupling_fn_prime - coupling_fn_second)
            * attributes[value_child_idx]["observed"]
        )

    return pwpe


def _standard_volatility_increment(
    attributes: dict,
    node_idx: int,
    volatility_child_idx: int,
    volatility_coupling: float,
) -> Array:
    r"""Compute the standard HGF volatility-coupling precision increment.

    With :math:`\gamma` the effective precision computed at the prediction step and
    :math:`\Delta` the child's volatility prediction error, the increment is
    :math:`\tfrac12 (\kappa \gamma)^2 + (\kappa \gamma)^2 \Delta - \tfrac12 \kappa^2
    \gamma \Delta`.
    """
    # volatility weigthed precision for the volatility child (γ)
    effective_precision = attributes[volatility_child_idx]["temp"][
        "effective_precision"
    ]

    # retrieve the volatility prediction error
    volatility_prediction_error = attributes[volatility_child_idx]["temp"][
        "volatility_prediction_error"
    ]

    # cancel the prediction error if the child value was not observed
    return (
        0.5 * (volatility_coupling * effective_precision) ** 2
        + (volatility_coupling * effective_precision) ** 2 * volatility_prediction_error
        - 0.5
        * volatility_coupling**2
        * effective_precision
        * volatility_prediction_error
    ) * attributes[volatility_child_idx]["observed"]


def _volatility_coupling_pwpe(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    increment_fn: Callable[[dict, int, int, float], Array],
) -> Array:
    """Precision-weighted prediction error from volatility-coupled children.

    Sums ``increment_fn`` (standard or enhanced-HGF) over the volatility children.
    """
    pwpe = 0.0
    if edges[node_idx].volatility_children is None:
        return pwpe

    for volatility_child_idx, volatility_coupling in zip(
        edges[node_idx].volatility_children,  # type: ignore
        attributes[node_idx]["volatility_coupling_children"],
    ):
        pwpe += increment_fn(
            attributes, node_idx, volatility_child_idx, volatility_coupling
        )

    return pwpe


def _finalize_precision(expected_precision: Array, pwpe: Array) -> Array:
    """Add the precision-weighted prediction error and clamp to a positive value."""
    posterior_precision = expected_precision + pwpe
    # ensure the new precision is greater than 0
    return jnp.where(posterior_precision > 1e-128, posterior_precision, jnp.nan)


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> float:
    r"""Update the precision of a state node using the volatility prediction errors.

    #. Precision update from value coupling.

    The new precision of a state node :math:`b` value coupled with other input and/or
    state nodes :math:`j` at time :math:`k` is given by:

    For linear coupling (default)

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \kappa_j^2 \hat{\pi}_j^{(k)}

    Where :math:`\kappa_j` is the volatility coupling strength between the child node
    and the state node and :math:`\delta_j^{(k)}` is the value prediction error that
    was computed beforehand by
    :py:func:`pyhgf.updates.prediction_errors.continuous.continuous_node_value_prediction_error`.

    For non-linear value coupling we use equation 50 from [1]_.:

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \hat{\pi}_j^{(k)} * (\kappa_j^2 * g'_{j,b}(\mu_b^(k-1))^2 -
            g''_{j,b}(\mu_b^(k-1))*\delta_j)

    #. Precision update from volatility coupling.

    The new precision of a state node :math:`b` volatility coupled with other input
    and/or state nodes :math:`j` at time :math:`k` is given by:

    .. math::

        \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
        \frac{1}{2} \left( \kappa_j \gamma_j^{(k)} \right) ^2 +
        \left( \kappa_j \gamma_j^{(k)} \right) ^2 \Delta_j^{(k)} -
        \frac{1}{2} \kappa_j^2 \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the effective precision of the prediction, given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    that was computed in the prediction step. See [1]_ for more details.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number
        of nodes. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision :
        The new posterior precision.


    See Also
    --------
    posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1

    """
    # ----------------------------------------------------------------------------------
    # Decide which update to use depending on the presence of observed value in the
    # children nodes. If no values were observed, the precision should increase
    # as a function of time using the function precision_missing_values(). Otherwise,
    # we use regular HGF updates for value and volatility couplings.
    # ----------------------------------------------------------------------------------
    return cond(
        _has_observations(attributes, edges, node_idx),
        Partial(precision_update, edges=edges, node_idx=node_idx),
        Partial(precision_update_missing_values, edges=edges, node_idx=node_idx),
        attributes,
    )


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update(attributes: dict, edges: Edges, node_idx: int) -> Array:
    r"""Compute new precision in the case of observed values.

    Applies the canonical value- and volatility-coupling posterior-precision updates
    (eq. 50 of [1]_), augmented by the *relaxed* posterior-step (smoothing) correction
    on each Gaussian-interior value child: the child-precision factor
    :math:`\hat{\pi}_a` is replaced by the harmonic combination.

    .. math::

        \hat{\pi}_a \longmapsto
        \frac{\hat{\pi}_a \, \pi_y}{\hat{\pi}_a + \pi_y}, \qquad
        \pi_y = \pi_a - \tilde{\pi}_a,

    where :math:`\hat{\pi}_a` is the child's *conditional* predicted precision
    (stored as ``child.temp["conditional_expected_precision"]``) and
    :math:`\tilde{\pi}_a` its *marginal* predicted precision
    (``child.expected_precision``). The Schur complement uses the conditional;
    substituting the marginal would double-count parent uncertainty. Non-Gaussian
    children and Gaussian leaves fall back to the canonical predicted-precision
    factor :math:`\tilde{\pi}_a` (:math:`\pi_a \to \infty`).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`.
    node_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    posterior_precision :
        The new posterior precision when at least one of the children has
        observed a new value.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    pwpe = _value_coupling_pwpe(attributes, edges, node_idx, _smoothing_child_precision)
    pwpe += _volatility_coupling_pwpe(
        attributes, edges, node_idx, _standard_volatility_increment
    )
    return _finalize_precision(attributes[node_idx]["expected_precision"], pwpe)


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update_missing_values(
    attributes: dict, edges: Edges, node_idx: int
) -> float:
    """Compute new precision in the case of missing observations.

    When no value or volatility child reports an observation at the current step,
    the node simply ages its precision by one step of its random walk: there are no
    prediction errors to integrate, so the new precision is the canonical predicted
    precision under the volatility-parent random walk.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`.
    node_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    posterior_precision_missing_values :
        The new posterior precision in the case of missing values in all child
        nodes. The new precision decreases proportionally to the time elapsed,
        accounting for the influence of volatility parents.
    """
    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling * attributes[volatility_parents_idx]["mean"]
            )

    # compute the new predicted_volatility from the total volatility
    time_step = attributes[-1]["time_step"]
    predicted_volatility = time_step * jnp.exp(total_volatility)

    # Estimate the new precision for the continuous state node
    posterior_precision_missing_values = 1 / (
        (1 / attributes[node_idx]["precision"]) + predicted_volatility
    )

    return posterior_precision_missing_values


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update_mean_field(attributes: dict, edges: Edges, node_idx: int) -> Array:
    """Compute new precision in the case of observed values.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number
        of nodes. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision :
        The new posterior precision when at least one of the children has
        observed a new value. We then use the regular HGF update for volatility
        coupling.
    """
    pwpe = _value_coupling_pwpe(
        attributes, edges, node_idx, _mean_field_child_precision
    )
    pwpe += _volatility_coupling_pwpe(
        attributes, edges, node_idx, _standard_volatility_increment
    )
    return _finalize_precision(attributes[node_idx]["expected_precision"], pwpe)


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node_mean_field(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> float:
    r"""Update the precision of a state node using the volatility prediction errors.

    #. Precision update from value coupling.

    The new precision of a state node :math:`b` value coupled with other input and/or
    state nodes :math:`j` at time :math:`k` is given by:

    For linear coupling (default)

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \kappa_j^2 \hat{\pi}_j^{(k)}

    Where :math:`\kappa_j` is the volatility coupling strength between the child node
    and the state node and :math:`\delta_j^{(k)}` is the value prediction error that
    was computed beforehand by
    :py:func:`pyhgf.updates.prediction_errors.continuous.continuous_node_value_prediction_error`.

    For non-linear value coupling we use equation 50 from [1]_.:

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \hat{\pi}_j^{(k)} * (\kappa_j^2 * g'_{j,b}(\mu_b^(k-1))^2 -
            g''_{j,b}(\mu_b^(k-1))*\delta_j)

    #. Precision update from volatility coupling.

    The new precision of a state node :math:`b` volatility coupled with other input
    and/or state nodes :math:`j` at time :math:`k` is given by:

    .. math::

        \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
        \frac{1}{2} \left( \kappa_j \gamma_j^{(k)} \right) ^2 +
        \left( \kappa_j \gamma_j^{(k)} \right) ^2 \Delta_j^{(k)} -
        \frac{1}{2} \kappa_j^2 \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the effective precision of the prediction, given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    that was computed in the prediction step. See [1]_ for more details.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number
        of nodes. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision :
        The new posterior precision.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized Hierarchical Gaussian Filter. eLife Sciences
       Publications, Ltd. https://doi.org/10.7554/elife.110174.1

    """
    return cond(
        _has_observations(attributes, edges, node_idx),
        Partial(precision_update_mean_field, edges=edges, node_idx=node_idx),
        Partial(precision_update_missing_values, edges=edges, node_idx=node_idx),
        attributes,
    )


def _ehgf_volatility_precision_increment(
    attributes: dict,
    node_idx: int,
    volatility_child_idx: int,
    volatility_coupling: float,
) -> Array:
    r"""Enhanced-HGF "safe" volatility-coupling precision increment.

    Implements the volatility-coupling precision update of the enhanced HGF (Mathys
    et al.; TAPAS ``hgf_volatility_update`` ``'ehgf'`` branch). Two features
    distinguish it from the standard increment:

    1. The effective precision is **recomputed from the parent's just-updated
       posterior mean** :math:`\mu_b^{(k)}` (the eHGF updates the mean first), rather
       than reusing the prediction-step effective precision evaluated at
       :math:`\hat{\mu}_b^{(k)}`.
    2. The increment is **floored at zero**, which guarantees the posterior precision
       never drops below the (positive) predicted precision and so cannot collapse to
       a negative value / ``NaN``.

    With :math:`\nu` the re-predicted child volatility, :math:`\hat{\pi}` the
    re-predicted child precision (``expected_precision``),
    :math:`\gamma = \nu \hat{\pi}` the ``effective_precision``,
    :math:`w = (\nu - \sigma^{(k-1)}) \hat{\pi}` the volatility-error weight and
    :math:`\Delta` the child's ``volatility_prediction_error``, the increment is
    :math:`\max(0, \tfrac12 \kappa^2 \gamma (\gamma + w \Delta))`.
    """
    time_step = attributes[-1]["time_step"]
    # Parent posterior mean (the eHGF performs the mean update first).
    mean = attributes[node_idx]["mean"]
    tonic_volatility = attributes[volatility_child_idx]["tonic_volatility"]
    # Child posterior variance at the previous time step (σ = 1 / π).
    previous_variance = attributes[volatility_child_idx]["temp"]["current_variance"]

    # Re-predict the child's volatility and precision from the parent posterior mean.
    predicted_volatility = time_step * jnp.exp(
        volatility_coupling * mean + tonic_volatility
    )
    expected_precision = 1.0 / (previous_variance + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision
    volatility_error_weight = (
        predicted_volatility - previous_variance
    ) * expected_precision
    volatility_prediction_error = (
        1.0 / attributes[volatility_child_idx]["precision"]
        + (
            attributes[volatility_child_idx]["mean"]
            - attributes[volatility_child_idx]["expected_mean"]
        )
        ** 2
    ) * expected_precision - 1.0

    increment = jnp.maximum(
        0.0,
        0.5
        * volatility_coupling**2
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error),
    )
    # cancel the contribution if the child value was not observed
    return increment * attributes[volatility_child_idx]["observed"]


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update_ehgf(attributes: dict, edges: Edges, node_idx: int) -> Array:
    """Enhanced-HGF precision update for observed values (relaxed value coupling).

    The value-coupling contribution is identical to :func:`precision_update` (the
    relaxed posterior-step correction); the volatility-coupling contribution uses the
    enhanced-HGF safe update via :func:`_ehgf_volatility_precision_increment`.
    """
    pwpe = _value_coupling_pwpe(attributes, edges, node_idx, _smoothing_child_precision)
    pwpe += _volatility_coupling_pwpe(
        attributes, edges, node_idx, _ehgf_volatility_precision_increment
    )
    return _finalize_precision(attributes[node_idx]["expected_precision"], pwpe)


@partial(jit, static_argnames=("edges", "node_idx"))
def precision_update_ehgf_mean_field(
    attributes: dict, edges: Edges, node_idx: int
) -> Array:
    """Enhanced-HGF precision update for observed values (mean-field value coupling).

    The value-coupling contribution is identical to :func:`precision_update_mean_field`
    (no smoothing correction); the volatility-coupling contribution uses the enhanced-
    HGF safe update via :func:`_ehgf_volatility_precision_increment`.
    """
    pwpe = _value_coupling_pwpe(
        attributes, edges, node_idx, _mean_field_child_precision
    )
    pwpe += _volatility_coupling_pwpe(
        attributes, edges, node_idx, _ehgf_volatility_precision_increment
    )
    return _finalize_precision(attributes[node_idx]["expected_precision"], pwpe)


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node_ehgf(
    attributes: dict, edges: Edges, node_idx: int
) -> float:
    """Route to the relaxed enhanced-HGF precision or missing-value update."""
    return cond(
        _has_observations(attributes, edges, node_idx),
        Partial(precision_update_ehgf, edges=edges, node_idx=node_idx),
        Partial(precision_update_missing_values, edges=edges, node_idx=node_idx),
        attributes,
    )


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node_ehgf_mean_field(
    attributes: dict, edges: Edges, node_idx: int
) -> float:
    """Route to the enhanced-HGF precision update (mean-field) or the missing path."""
    return cond(
        _has_observations(attributes, edges, node_idx),
        Partial(precision_update_ehgf_mean_field, edges=edges, node_idx=node_idx),
        Partial(precision_update_missing_values, edges=edges, node_idx=node_idx),
        attributes,
    )
