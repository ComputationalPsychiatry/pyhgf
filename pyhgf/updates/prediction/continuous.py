# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import Array, grad, jit

from pyhgf.typing import Edges


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
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

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
) -> tuple[Array, Array]:
    r"""Compute the expected precision of a continuous state node.

    This is the *improved* (piHGF) prediction step, which marginalises over the parents'
    variational Gaussians rather than collapsing them to point estimates. Two additional
    variance terms appear relative to the canonical (g)HGF formulation [1]_:

    * an **exact** volatility-coupling term obtained from the moment-generating
      function of the Gaussian volatility parent;
    * a **first-order Laplace** value-coupling term arising from the Taylor
      expansion of the coupling function.

    Both terms vanish in the limit of perfectly known parents, recovering the canonical
    formula exactly.

    The marginal predictive variance at time :math:`k` for a state node :math:`a`
    with volatility parents :math:`\check a` and value parents :math:`b` is

    .. math::

        \mathrm{Var}\!\left[x_a^{(k)}\right] =
            \frac{1}{\pi_a^{(k-1)}}
            + t^{(k)}
              \exp\!\left( \omega_a
                + \sum_{j} \left(
                    \kappa_{a,\check a_j}\, \hat{\mu}_{\check a_j}
                    + \frac{\kappa_{a,\check a_j}^{2}}{2\, \hat{\pi}_{\check a_j}}
                  \right)
              \right)
            + \sum_{b} \frac{
                \left( t^{(k)}\, \alpha_{b,a}\, g'(\hat{\mu}_b) \right)^{2}
              }{ \hat{\pi}_b }.

    The new term :math:`\kappa_{a,\check a_j}^{2} / (2\, \hat{\pi}_{\check a_j})`
    inside the exponent is the Jensen-inequality contribution of a Gaussian-distributed
    volatility parent through the convex :math:`\exp(\cdot)` non-linearity, available in
    closed form. The new term :math:`(\alpha_{b,a}\, g'(\hat{\mu}_b))^{2} / \hat{\pi}_b`
    outside the exponent is the law-of-total-variance contribution of a
    Gaussian-distributed value parent through the linearised coupling function. Both use
    *predicted*, not posterior, parent precisions, keeping the prediction schedule
    strictly top-down.

    The expected precision is the inverse of the marginal predictive variance:

    .. math::

        \hat{\pi}_a^{(k)} = \mathrm{Var}\!\left[x_a^{(k)}\right]^{-1}.

    The *effective precision* :math:`\gamma_a^{(k)}` is defined relative to the
    volatility-driven part of the variance only,

    .. math::

        \gamma_a^{(k)} = \Omega_a^{(k)}\, \hat{\pi}_a^{(k)},

    where :math:`\Omega_a^{(k)}` denotes the (improved) phasic + tonic volatility
    contribution. This value is stored on the node for later use during the
    posterior update.

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
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

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
    # (t · α · g'(μ̂_b))² / π̂_b. The factor t arises because the value-parent
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

    # Estimate the new expected precision for the node by inverting the
    # marginal predictive variance (own variance + improved volatility term +
    # Laplace value-coupling term).
    expected_precision = 1 / (
        (1 / attributes[node_idx]["precision"])
        + predicted_volatility
        + value_coupling_variance
    )

    # compute the effective precision (γ); only the volatility-driven part
    # enters γ, since γ is consumed by the volatility-coupling posterior update.
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, effective_precision


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

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.

    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).

    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

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
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937
    """
    # if this node has volatility parent(s), store the current variance
    # to be used by the posterior update if using unbounded approximation
    attributes[node_idx]["temp"]["current_variance"] = (
        1 / attributes[node_idx]["precision"]
    )

    # Get the new expected mean
    expected_mean = predict_mean(attributes, edges, node_idx)

    # Get the new expected precision and predicted volatility (Ω)
    expected_precision, effective_precision = predict_precision(
        attributes, edges, node_idx
    )

    # Update this node's parameters

    # 1. input node without volatility parent
    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]

    # 2. regular continuous state node, or input with volatility parent
    else:
        attributes[node_idx]["expected_precision"] = expected_precision

    attributes[node_idx]["temp"]["effective_precision"] = effective_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes
