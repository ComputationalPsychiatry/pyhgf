# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import Array, grad, jit

from pyhgf.typing import Edges

#: Floor on the total predicted variance (carried + volatility + value-coupling).
#: At λ = 0 the carried term vanishes, so without it the precision can diverge as
#: Ω → 0; at the default λ = 1 the sum already exceeds it, so it never binds.
MIN_PREDICTED_VARIANCE = 1e-128


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_mean(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> Array:
    r"""Compute the expected mean of a continuous state node.

    The expected mean at time :math:`k` for a state node :math:`a` with optional value
    parent(s) :math:`b` is in [1]_ given by:

    .. math::

        \hat{\mu}_a^{(k)} = \lambda_a \mu_a^{(k-1)} + P_a^{(k)}

    where :math:`P_a^{(k)}` is the drift rate (the total predicted drift of the expected
    mean, which sums the tonic and - optionally - phasic drifts). The variable
    :math:`lambda_a` represents the state's autoconnection strength, with
    :math:`\lambda_a \in [0, 1]`. When :math:`\lambda_a = 1`, the node is performing a
    Gaussian Random Walk using the value :math:`P_a^{(k)}` as total drift rate. When
    :math:`\lambda_a < 1`, the state will revert back to the total mean :math:`M_a`,
    which is given by:

    .. math::
            M_a = \frac{\rho_a + f\left(\hat{\mu}_b^{(k)}\right)} {1-\lambda_a},

    If :math:`\lambda_a = 0`, the node is not influenced by its own mean anymore, but
    by the value received by the value parent.

    .. hint::

        By combining one parameter :math:`\lambda_a \in [0, 1]` and the influence of
        value parents, it is possible to implement both Gaussian Random Walks and
        Autoregressive Processes, without requiring specific coupling types.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_mean :
        The new expected mean of the state node.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1

    """
    time_step = attributes[-1]["time_step"]

    # List the node's value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Get the drift rate from the node
    driftrate = attributes[node_idx]["tonic_drift"]

    # Look at the (optional) value parents for this node
    # and update the drift rate accordingly
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            # look at each value parent
            # and get the coupling function to compute the drift
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            if coupling_fn is None:
                parent_value = attributes[value_parent_idx]["expected_mean"]
            else:
                parent_value = coupling_fn(
                    attributes[value_parent_idx]["expected_mean"]
                )

            driftrate += psi * parent_value

    # The new expected mean from the previous value
    expected_mean = (
        attributes[node_idx]["autoconnection_strength"] * attributes[node_idx]["mean"]
    ) + (time_step * driftrate)

    return expected_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision(
    attributes: dict, edges: Edges, node_idx: int
) -> tuple[Array, Array, Array]:
    r"""Compute the predicted precisions of a continuous state node (no mean-field).

    This is the *improved* (piHGF) prediction step, which marginalises over the parents'
    variational Gaussians rather than collapsing them to point estimates. Two additional
    variance terms appear relative to the canonical (g)HGF formulation [1]_:

    * an **exact** volatility-coupling term obtained from the moment-generating
      function of the Gaussian volatility parent;
    * a **first-order Laplace** value-coupling term arising from the Taylor
      expansion of the coupling function.

    Both terms vanish in the limit of perfectly known parents, recovering the canonical
    formula exactly.

    Two predicted precisions are computed and stored on the node (see
    :class:`pyhgf.typing.LayerState` for the notation table):

    .. math::

        \frac{1}{\hat{\pi}_a^{(k)}} = \frac{\lambda_a^2}{\pi_a^{(k-1)}}
            + \Omega_a^{(k)},

    .. math::

        \frac{1}{\tilde{\pi}_a^{(k)}}
            \;=\; \frac{1}{\hat{\pi}_a^{(k)}}
              + \sum_{b} \frac{ (t^{(k)} \alpha_{b,a} g'(\hat{\mu}_b))^2 }
                              { \tilde{\pi}_b },

    with the (improved) phasic-plus-tonic log-volatility

    .. math::

        \Omega_a^{(k)} = t^{(k)} \exp\!\left(
            \omega_a + \sum_{j} \left(
                \kappa_{a,\check{a}_j} \hat{\mu}_{\check{a}_j}
                + \frac{\kappa_{a,\check{a}_j}^2}{2 \hat{\pi}_{\check{a}_j}}
            \right)
        \right).

    The factor :math:`\lambda_a^2` scales the carried variance (scaling a Gaussian
    by :math:`\lambda_a` scales its variance by :math:`\lambda_a^2`); the drift is
    added after the autoregression, so the value-coupling term stays unscaled.

    .. note::

        Eq. 35 of [1]_ omits :math:`\lambda_a^2`; the two forms agree at
        :math:`\lambda_a = 1`, the value used in every simulation in the paper.

    :math:`\hat{\pi}_a^{(k)}` is the *conditional* predicted precision
    (``conditional_expected_precision``) — own variance plus volatility, without the
    parent-uncertainty bleed-through; it is the precision that enters the parent's
    Schur-complement posterior-step correction. :math:`\tilde{\pi}_a^{(k)}` is the
    *marginal* predicted precision (``expected_precision``) — the inverse of the
    marginal predictive variance, including the first-order Laplace value-coupling
    term. The bleed-through uses each parent's marginal predicted precision
    :math:`\tilde{\pi}_b`, which generalises the artefact's two-node derivation to
    deep networks. The MGF correction
    :math:`\kappa^2 / (2\, \hat{\pi}_{\check{a}_j})` inside the exponent is the
    Jensen-inequality contribution of a Gaussian volatility parent through the
    convex :math:`\exp(\cdot)` non-linearity. Both corrections use *predicted*, not
    posterior, parent precisions, keeping the prediction schedule strictly top-down.

    The *effective precision* :math:`\gamma_a^{(k)}` is defined relative to the
    volatility-driven part of the marginal variance only,

    .. math::

        \gamma_a^{(k)} = \Omega_a^{(k)} \, \tilde{\pi}_a^{(k)},

    and is consumed by the volatility-coupling posterior update at the volatility
    parent.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. For each node, the entry lists its
        value/volatility parents and children.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_precision :
        The marginal predicted precision :math:`\tilde{\pi}_a^{(k)}`.
    conditional_expected_precision :
        The conditional predicted precision :math:`\hat{\pi}_a^{(k)}`.
    effective_precision :
        The effective precision :math:`\gamma_a^{(k)}`; stored on the node for later
        use in the volatility-coupling posterior update.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    time_step = attributes[-1]["time_step"]

    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility. The piHGF improvement adds the
    # closed-form moment-generating-function correction κ²/(2 π̂) inside the
    # exponent so that the volatility parent enters as a full Gaussian rather
    # than a point estimate.
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling
                * attributes[volatility_parents_idx]["expected_mean"]
            )
            total_volatility += (volatility_coupling**2) / (
                2.0 * attributes[volatility_parents_idx]["expected_precision"]
            )

    # compute the predicted_volatility from the total volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # piHGF Laplace value-coupling correction. The conditional mean of x_a is
    # linearised around μ̂_b via a first-order Taylor expansion of the coupling
    # function g; the resulting variance contribution from each value parent is
    # (t · α · g'(μ̂_b))² / π̃_b, using the parent's marginal predicted precision
    # π̃_b (= `expected_precision`). The factor t arises because the value-parent
    # contribution to the mean is scaled by the time step in :func:`predict_mean`.
    value_parents_idxs = edges[node_idx].value_parents
    value_coupling_variance = jnp.zeros_like(predicted_volatility)
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            mu_b = attributes[value_parent_idx]["expected_mean"]
            if coupling_fn is None:
                g_prime = jnp.ones_like(mu_b)
            else:
                g_prime = grad(coupling_fn)(mu_b)

            value_coupling_variance += (time_step * psi * g_prime) ** 2 / attributes[
                value_parent_idx
            ]["expected_precision"]

    # Carried variance: input nodes carry 1/π (λ = 1 for precision, λ = 0 for
    # the mean — no tonic volatility); other nodes scale by λ².
    is_input_node = (
        edges[node_idx].value_children is None
        and edges[node_idx].volatility_children is None
    )
    if is_input_node:
        carried_variance = 1.0 / attributes[node_idx]["precision"]
    else:
        carried_variance = (
            attributes[node_idx]["autoconnection_strength"] ** 2
        ) / attributes[node_idx]["precision"]

    # Conditional π̂_a: carried variance + volatility, no parent bleed-through
    # (the parent's Schur complement acts on this; the marginal would double-count).
    conditional_expected_precision = 1 / jnp.maximum(
        carried_variance + predicted_volatility, MIN_PREDICTED_VARIANCE
    )

    # Marginal π̃_a: carried variance + volatility + Laplace value-coupling variance.
    expected_precision = 1 / jnp.maximum(
        carried_variance + predicted_volatility + value_coupling_variance,
        MIN_PREDICTED_VARIANCE,
    )

    # compute the effective precision (γ); only the volatility-driven part
    # enters γ, since γ is consumed by the volatility-coupling posterior update.
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, conditional_expected_precision, effective_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the expected mean and expected precision of a continuous node [1]_.

    The precision prediction follows the improved (piHGF) scheme, which marginalises
    over the parents' variational Gaussians instead of treating them as point estimates.
    See :func:`predict_precision` for the precise formula and its two additional
    variance terms (volatility-coupling moment-generating- function term and
    value-coupling Laplace term).

    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. The tuple has the same length as the
        node number. For each node, the index lists the value and volatility parents
        and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    # Store carried variance for the volatility-coupling posteriors (unbounded,
    # eHGF), which re-predict from it — same quantity predict_precision uses.
    # Input nodes carry 1/π (λ = 1 for precision); other nodes λ²/π.
    if (
        edges[node_idx].value_children is None
        and edges[node_idx].volatility_children is None
    ):
        attributes[node_idx]["temp"]["current_variance"] = (
            1.0 / attributes[node_idx]["precision"]
        )
    else:
        attributes[node_idx]["temp"]["current_variance"] = (
            attributes[node_idx]["autoconnection_strength"] ** 2
        ) / attributes[node_idx]["precision"]

    # Get the new expected mean
    expected_mean = predict_mean(attributes, edges, node_idx)

    # Get the new marginal (π̃) and conditional (π̂) predicted precisions and the
    # effective precision γ.
    expected_precision, conditional_expected_precision, effective_precision = (
        predict_precision(attributes, edges, node_idx)
    )

    # Update this node's parameters

    # 1. input node without volatility parent
    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]
        # A leaf has no random walk, so conditional == marginal == prior precision.
        attributes[node_idx]["temp"]["conditional_expected_precision"] = attributes[
            node_idx
        ]["precision"]

    # 2. regular continuous state node, or input with volatility parent
    else:
        attributes[node_idx]["expected_precision"] = expected_precision
        attributes[node_idx]["temp"]["conditional_expected_precision"] = (
            conditional_expected_precision
        )

    attributes[node_idx]["temp"]["effective_precision"] = effective_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision_mean_field(
    attributes: dict, edges: Edges, node_idx: int
) -> tuple[Array, Array, Array]:
    r"""Compute the expected precision of a continuous state node (mean-field).

    The expected precision at time :math:`k` for a state node :math:`a` is given by
    [1]_:

    .. math::

        \hat{\pi}_a^{(k)} = \frac{1}
            {\frac{\lambda_a^2}{\pi_a^{(k-1)}} + \Omega_a^{(k)}}

    where :math:`\lambda_a` is the autoconnection strength and
    :math:`\Omega_a^{(k)}` is the *total predicted volatility*. This term is the
    sum of the tonic (endogenous) and phasic (exogenous) volatility, such as:

    .. math::

        \Omega_a^{(k)} = t^{(k)}
        \exp{ \left( \omega_a + \sum_{j=1}^{N_{vopa}} \kappa_j \hat{\mu}_a^{(k-1)} \right) }


    with :math:`\kappa_j` the volatility coupling strength with the volatility parent
    :math:`j`. The carried variance takes :math:`\lambda_a^2` because scaling the
    belief by :math:`\lambda_a` scales its variance by :math:`\lambda_a^2`.

    The *effective precision* :math:`\gamma_a^{(k)}` is given by:

    .. math::

        \gamma_a^{(k)} = \Omega_a^{(k)} \hat{\pi}_a^{(k)}

    This value is also saved in the node for later use during the update steps.


    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_precision :
        The new expected precision of the value parent.
    effective_precision :
        The effective_precision :math:`\gamma_a^{(k)}`. This value is stored in the
        node for later use in the update steps.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized Hierarchical Gaussian Filter. eLife, 15,
       RP110174. https://doi.org/10.7554/eLife.110174.1

    """
    time_step = attributes[-1]["time_step"]

    volatility_parents_idxs = edges[node_idx].volatility_parents
    total_volatility = attributes[node_idx]["tonic_volatility"]

    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling
                * attributes[volatility_parents_idx]["expected_mean"]
            )

    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Input nodes carry their precision over (λ = 1 for precision); other nodes
    # scale by λ². See :func:`predict_precision` for the full rationale.
    is_input_node = (
        edges[node_idx].value_children is None
        and edges[node_idx].volatility_children is None
    )
    if is_input_node:
        carried_variance = 1.0 / attributes[node_idx]["precision"]
    else:
        carried_variance = (
            attributes[node_idx]["autoconnection_strength"] ** 2
        ) / attributes[node_idx]["precision"]
    expected_precision = 1 / jnp.maximum(
        carried_variance + predicted_volatility,
        MIN_PREDICTED_VARIANCE,
    )
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, expected_precision, effective_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction_mean_field(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Mean-field (v0.2.11) prediction step for a continuous node.

    Uses the simple v0.2.11 formula: no MGF correction on volatility total, no Laplace
    value-coupling variance term.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. The tuple has the same length as the
        node number. For each node, the index lists the value and volatility parents
        and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.
    """
    # Store carried variance (input nodes: 1/π; others: λ²/π) for the
    # volatility-coupling posteriors, which re-predict from it.
    if (
        edges[node_idx].value_children is None
        and edges[node_idx].volatility_children is None
    ):
        attributes[node_idx]["temp"]["current_variance"] = (
            1.0 / attributes[node_idx]["precision"]
        )
    else:
        attributes[node_idx]["temp"]["current_variance"] = (
            attributes[node_idx]["autoconnection_strength"] ** 2
        ) / attributes[node_idx]["precision"]

    expected_mean = predict_mean(attributes, edges, node_idx)

    expected_precision, conditional_expected_precision, effective_precision = (
        predict_precision_mean_field(attributes, edges, node_idx)
    )

    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]
        attributes[node_idx]["temp"]["conditional_expected_precision"] = attributes[
            node_idx
        ]["precision"]
    else:
        attributes[node_idx]["expected_precision"] = expected_precision
        attributes[node_idx]["temp"]["conditional_expected_precision"] = (
            conditional_expected_precision
        )

    attributes[node_idx]["temp"]["effective_precision"] = effective_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes
