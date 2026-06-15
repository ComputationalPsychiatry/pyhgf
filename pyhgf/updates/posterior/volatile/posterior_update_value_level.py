# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial

from jax import grad, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> float:
    r"""Update the precision of the value level using value children's PEs.

    Implements the posterior-step (smoothing) correction of the relaxed HGF in its
    fully-corrected form, paired with the prediction-step marginal-precision
    correction. Lifting the mean-field assumption :math:`q(x_a, x_b) = q(x_a)\,q(x_b)`
    on the value-coupling edge to a structured Gaussian and applying the Schur
    complement to the joint :math:`(x_a, x_b)` precision matrix replaces the canonical
    child-precision factor by the harmonic combination

    .. math::

        \hat{\pi}_a \longmapsto
        \frac{\hat{\pi}_a \, \pi_y}{\hat{\pi}_a + \pi_y}, \qquad
        \pi_y = \pi_a - \tilde{\pi}_a,

    where :math:`\hat{\pi}_a` is the child's *conditional* predicted precision (stored
    on the child as ``temp["conditional_expected_precision"]``: own variance plus
    volatility, without the parent-uncertainty value-coupling term) and
    :math:`\tilde{\pi}_a` its *marginal* predicted precision (``expected_precision``).
    The Schur complement carries the conditional; substituting the marginal would
    double-count parent uncertainty. The same factor scales both the
    :math:`(\kappa g')^2` and :math:`\kappa g''\,\delta_a` contributions to
    :math:`\pi_b^{(k)}`. Reduces to the canonical formula when the child is fully
    observed (:math:`\pi_y \to \infty`); returns no contribution when the child gained
    no bottom-up information (:math:`\pi_y = 0`).

    The smoothing form is only applied when the child is a *Gaussian interior* node
    (continuous-state type 2 or volatile-state type 6, with children of its own).
    Other children — binary, categorical, input/constant, exponential-family,
    Dirichlet, and Gaussian *leaves* — fall back to the canonical contribution using
    :math:`\tilde{\pi}_a`. Leaves are handled specially because pyhgf keeps
    :math:`\pi_a = \tilde{\pi}_a` for clamped observations, which would otherwise zero
    out the smoothing form via :math:`\pi_y = 0`.

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`.
    node_idx :
        Index of the volatile state node whose value-level precision is updated.

    Returns
    -------
    posterior_precision :
        The new posterior precision of the value level.
    """
    # Start with expected precision
    posterior_precision = attributes[node_idx]["expected_precision"]

    # Add contributions from value children
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # Effective child precision under the smoothing correction. The Schur
            # derivation assumes a Gaussian-Gaussian value-coupling edge, so the
            # correction only applies when the child carries a Gaussian belief
            # (continuous-state type 2 or volatile-state type 6) AND is interior
            # (has children of its own). Binary, categorical, input/constant,
            # exponential-family, and Dirichlet children, plus any Gaussian leaf,
            # fall back to the canonical predicted-precision factor — the marginal
            # π̃_a stored in `child.expected_precision`.
            child_node_type = edges[value_child_idx].node_type
            child_is_gaussian_interior = child_node_type in (2, 6) and (
                edges[value_child_idx].value_children is not None
                or edges[value_child_idx].volatility_children is not None
            )
            child_expected_precision = attributes[value_child_idx]["expected_precision"]
            if child_is_gaussian_interior:
                # Bottom-up evidence precision π_y = π_a − π̃_a, measured against
                # the child's marginal predicted precision π̃_a.
                child_precision = attributes[value_child_idx]["precision"]
                pi_y = child_precision - child_expected_precision
                # The Schur complement acts on the joint (x_a, x_b) precision matrix,
                # which carries the conditional predicted precision π̂_a (stored on
                # the child as `temp["conditional_expected_precision"]` for both
                # continuous- and volatile-state children). Using π̃_a here would
                # double-count parent uncertainty.
                child_cond = attributes[value_child_idx]["temp"][
                    "conditional_expected_precision"
                ]
                effective_child_precision = child_cond * pi_y / (child_cond + pi_y)
            else:
                effective_child_precision = child_expected_precision

            # Linear coupling
            if coupling_fn is None:
                posterior_precision += (value_coupling**2) * effective_child_precision
            else:
                # Non-linear coupling (with gradient)
                coupling_fn_prime = grad(coupling_fn)(
                    attributes[node_idx]["expected_mean"]
                )
                coupling_fn_second = grad(grad(coupling_fn))(
                    attributes[node_idx]["expected_mean"]
                )
                value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

                posterior_precision += effective_child_precision * (
                    (value_coupling**2) * (coupling_fn_prime**2)
                    - value_coupling * coupling_fn_second * value_pe
                )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    node_precision: float,
) -> float:
    r"""Update the mean of the value level using value children's PEs (no mean-field).

    Uses the joint-Gaussian (RTS-smoother) gain rather than the canonical
    value-coupling gain. Each value child contributes

    .. math::

        \Delta \mu_b \mathrel{+}= \frac{\kappa \, g'(\hat{\mu}_b) \, g_a}
            {\pi_b} \, \delta_a, \qquad
        g_a \;=\; \frac{\hat{\pi}_a \, \pi_a}{\hat{\pi}_a + \pi_y}, \qquad
        \pi_y \;=\; \pi_a - \tilde{\pi}_a,

    where :math:`\pi_b` is the already-updated parent posterior precision (passed in
    as ``node_precision``) and the division by it happens once, after summing over
    children. This is the exact marginal mean of the joint :math:`(x_a, x_b)`
    Gaussian for any number of children — not a sum of independent single-child
    smoother gains, which would over-weight each child by
    :math:`\pi_b^{(i)} / \pi_b`. For leaves and non-Gaussian children
    :math:`\pi_y = 0` and :math:`g_a` collapses to the marginal :math:`\tilde{\pi}_a`,
    recovering the canonical gain.

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`.
    node_idx :
        Index of the volatile state node whose value-level mean is updated.
    node_precision :
        The just-updated value-level posterior precision :math:`\pi_b` of the node;
        the precision-weighted PEs are divided by this value once, after summing
        over children.

    Returns
    -------
    posterior_mean :
        The new posterior mean of the value level.
    """
    # Start with expected mean
    posterior_mean = attributes[node_idx]["expected_mean"]

    # Add contributions from value children
    value_precision_weighted_pe = 0.0

    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # Get the value prediction error
            value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

            # Get coupling function derivative
            if coupling_fn is None:
                coupling_fn_prime = 1
            else:
                coupling_fn_prime = grad(coupling_fn)(
                    attributes[node_idx]["expected_mean"]
                )

            # Coupling gain precision g_a. From the joint (x_a, x_b) Gaussian the
            # exact marginal-mean gain is
            #     g_a = π̂_a · π_a / (π̂_a + π_y),    π_y = π_a − π̃_a,
            # accumulated across children and divided once by the parent posterior
            # precision (``node_precision``) below — this is what makes the
            # multi-child mean exact rather than a sum of independent RTS gains. For
            # leaves / non-Gaussian children π_y = 0 and g_a collapses to the
            # marginal π̃_a, recovering the canonical gain.
            child_expected_precision = attributes[value_child_idx]["expected_precision"]
            child_node_type = edges[value_child_idx].node_type
            child_is_gaussian_interior = child_node_type in (2, 6) and (
                edges[value_child_idx].value_children is not None
                or edges[value_child_idx].volatility_children is not None
            )
            if child_is_gaussian_interior:
                child_precision = attributes[value_child_idx]["precision"]
                pi_y = child_precision - child_expected_precision
                child_cond = attributes[value_child_idx]["temp"][
                    "conditional_expected_precision"
                ]
                gain_precision = child_cond * child_precision / (child_cond + pi_y)
            else:
                gain_precision = child_expected_precision

            # Accumulate precision-weighted PE
            value_precision_weighted_pe += (
                (value_coupling * gain_precision * coupling_fn_prime) / node_precision
            ) * value_pe

    posterior_mean += value_precision_weighted_pe

    return posterior_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_value_level_mean_field(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> float:
    """Update the precision of the value level using value children's PEs.

    This is similar to continuous node value coupling precision update.

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.
    """
    posterior_precision = attributes[node_idx]["expected_precision"]

    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            child_expected_precision = attributes[value_child_idx]["expected_precision"]

            if coupling_fn is None:
                posterior_precision += (value_coupling**2) * child_expected_precision
            else:
                coupling_fn_prime = grad(coupling_fn)(
                    attributes[node_idx]["expected_mean"]
                )
                coupling_fn_second = grad(grad(coupling_fn))(
                    attributes[node_idx]["expected_mean"]
                )
                value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

                posterior_precision += child_expected_precision * (
                    (value_coupling**2) * (coupling_fn_prime**2)
                    - value_coupling * coupling_fn_second * value_pe
                )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_value_level_mean_field(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    node_precision: float,
) -> float:
    """Update the mean of the value level using value children's PEs.

    This is similar to continuous node value coupling mean update.

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.
    """
    posterior_mean = attributes[node_idx]["expected_mean"]
    value_precision_weighted_pe = 0.0

    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

            if coupling_fn is None:
                coupling_fn_prime = 1
            else:
                coupling_fn_prime = grad(coupling_fn)(
                    attributes[node_idx]["expected_mean"]
                )

            child_expected_precision = attributes[value_child_idx]["expected_precision"]

            value_precision_weighted_pe += (
                (value_coupling * child_expected_precision * coupling_fn_prime)
                / node_precision
            ) * value_pe

    posterior_mean += value_precision_weighted_pe

    return posterior_mean
