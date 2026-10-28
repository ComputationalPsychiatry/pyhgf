# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.nn import sigmoid, softmax

from pyhgf.math import lambert_w0
from pyhgf.typing import Edges

from .posterior_update_mean_continuous_node import (
    posterior_update_mean_continuous_node,
    posterior_update_mean_continuous_node_mean_field,
)
from .posterior_update_precision_continuous_node import (
    _has_observations,
    precision_update,
    precision_update_mean_field,
    precision_update_missing_values,
)


@partial(
    jit,
    static_argnames=(
        "edges",
        "node_idx",
        "max_posterior_precision",
        "mean_field_updates",
    ),
)
def continuous_node_posterior_update_unbounded(
    attributes: dict,
    node_idx: int,
    edges: Edges,
    max_posterior_precision: float = 1e10,
    mean_field_updates: bool = False,
    **args,
) -> dict:
    """Update the posterior of a continuous node with unbounded quadratic approximation.

    The posterior is the :math:`(n+1)`-expansion update of section 3.6 in [1]_
    (:py:func:`posterior_update_unbounded`), so that every volatility and value child
    contributes to the posterior. For a single volatility child and no value child,
    this is the two-expansion update of section 3.4 in [1]_. Unobserved children are
    left out and, if no child is observed, the precision ages as in
    :py:func:`pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node.precision_update_missing_values`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated.
    edges :
        The edges of the probabilistic nodes.
    max_posterior_precision :
        Upper bound applied to the posterior precision write. Default ``1e10``.
    mean_field_updates :
        Only used when the node also has value children. If ``False`` (default), the
        value-coupling terms use the relaxed updates; if ``True``, the mean-field
        updates. This mirrors the ``mean_field_updates`` argument of
        :py:class:`pyhgf.model.Network`.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Mathys, C., Legrand, N., Waade, P. T., Mikuš, N., & Weber, L. A. (2026).
       Robust volatility updates for Hierarchical Gaussian Filtering.
       arXiv:2605.00966. https://doi.org/10.48550/arXiv.2605.00966
    """
    posterior_precision, posterior_mean = posterior_update_unbounded(
        attributes=attributes,
        node_idx=node_idx,
        edges=edges,
        mean_field_updates=mean_field_updates,
    )
    # without any observed child, the precision ages as in the standard update
    posterior_precision = jnp.where(
        _has_observations(attributes, edges, node_idx),
        posterior_precision,
        precision_update_missing_values(attributes, edges, node_idx),
    )

    attributes[node_idx]["precision"] = jnp.minimum(
        posterior_precision, max_posterior_precision
    )
    attributes[node_idx]["mean"] = posterior_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx", "mean_field_updates"))
def posterior_update_unbounded(
    attributes: dict,
    node_idx: int,
    edges: Edges,
    mean_field_updates: bool = False,
    **args,
) -> tuple[Array, Array]:
    r"""Compute unbounded posterior for a continuous volatility-parent node.

    :math:`(n+1)`-expansion update of section 3.6 in [1]_, for a node with one or
    several volatility children and, optionally, value children. With :math:`n`
    volatility children :math:`a_i`, coupling strengths :math:`\kappa_i`, carried
    variances :math:`\alpha_i`, total posterior uncertainties
    :math:`\beta_i = 1/\pi_{a_i} + (\mu_{a_i} - \hat{\mu}_{a_i})^2` and effective tonic
    volatilities :math:`\tilde{\omega}_{a_i} = \omega_{a_i} + \sum_p \kappa_{p, a_i}
    \hat{\mu}_p`, which absorb the other volatility parents :math:`p` of the child, the
    variational energy of the node is, up to an additive constant:

    .. math::

        I(x) = \sum_{i=1}^{n} \left[
            -\frac{1}{2}\log\left(\alpha_i + \Omega_i(x)\right)
            -\frac{1}{2}\frac{\beta_i}{\alpha_i + \Omega_i(x)}
        \right] - \frac{\tilde{\pi}}{2}\left(x - \tilde{\mu}\right)^2,
        \qquad \Omega_i(x) = t \exp(\kappa_i x + \tilde{\omega}_{a_i}).

    :math:`(\tilde{\pi}, \tilde{\mu})` is the prediction :math:`(\hat{\pi}, \hat{\mu})`
    of the node or, when the node also has value children, the effective prior that
    absorbs the value-coupling terms. The children whose ``observed`` flag is 0 are left
    out of the sum.

    1. A first expansion at the prediction :math:`\hat{\mu}` keeps only the concave part
       of :math:`I` for the precision (eqs. 52-53 in [1]_), which is therefore positive:

       .. math::

           \pi_{L_1} = \tilde{\pi} + \sum_{i=1}^{n} \frac{\kappa_i^2}{2} w_i (1 - w_i),
           \qquad
           \mu_{L_1} = \hat{\mu} + \frac{1}{\pi_{L_1}} \left(
               \tilde{\pi} (\tilde{\mu} - \hat{\mu})
               + \sum_{i=1}^{n} \frac{\kappa_i}{2} w_i \Delta_i \right),

       with :math:`w_i = \Omega_i(\hat{\mu}) / (\alpha_i + \Omega_i(\hat{\mu}))` and
       :math:`\Delta_i` the volatility prediction error of the child :math:`a_i`.

    2. For each volatility child, a second expansion is taken at the Lambert
       :math:`W_0` approximation of the mode created by this child (eq. 54 in [1]_),
       with the curvature and gradient of the full variational energy, which includes
       all the children (eqs. 55-56 in [1]_). A non-positive curvature falls back to
       the concave curvature. An expansion that is not finite, or whose child has a
       zero coupling or is unobserved, gets a zero weight.

    3. The :math:`n + 1` expansions are weighted by a softmax of the variational energy
       at their means (eq. 57 in [1]_). The posterior is the moment-matched Gaussian of
       this mixture (eqs. 58-60 in [1]_).

    For a single volatility child and no value child, this reduces to the two-expansion
    update of section 3.4 in [1]_.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated.
    edges :
        The edges of the probabilistic nodes.
    mean_field_updates :
        Only used when the node has value children. If ``False`` (default), the
        effective prior uses the relaxed value-coupling updates; if ``True``, the
        mean-field updates.

    Returns
    -------
    posterior_precision, posterior_mean :
        Updated precision and mean of the node.

    References
    ----------
    .. [1] Mathys, C., Legrand, N., Waade, P. T., Mikuš, N., & Weber, L. A. (2026).
       Robust volatility updates for Hierarchical Gaussian Filtering.
       arXiv:2605.00966. https://doi.org/10.48550/arXiv.2605.00966
    """
    volatility_children_idxs = edges[node_idx].volatility_children
    n_children = len(volatility_children_idxs)  # type: ignore
    log_time_step = jnp.log(attributes[-1]["time_step"])
    expected_mean = attributes[node_idx]["expected_mean"]

    # Effective prior (π̃, μ̃): the prediction, combined with the value-coupling terms
    # when the node also has value children.
    prior_precision, prior_mean = _value_children_effective_prior(
        attributes, edges, node_idx, mean_field_updates
    )

    # Per-child quantities. An unobserved child (``observed`` is 0) is left out: its
    # terms are multiplied by ``observed`` and its expansion gets a zero weight.
    couplings, tonic_volatilities, log_previous_variances = [], [], []
    be_auxs, volatility_prediction_errors, observed = [], [], []
    for position, volatility_child_idx in enumerate(volatility_children_idxs):  # type: ignore
        observed.append(attributes[volatility_child_idx]["observed"])
        couplings.append(attributes[node_idx]["volatility_coupling_children"][position])
        # effective tonic volatility ω̃: the other volatility parents of the child enter
        # through their predicted means, as in the prediction of the child
        tonic_volatility = attributes[volatility_child_idx]["tonic_volatility"]
        for parent_idx, coupling in zip(
            edges[volatility_child_idx].volatility_parents,  # type: ignore
            attributes[volatility_child_idx]["volatility_coupling_parents"],
        ):
            if parent_idx != node_idx:
                tonic_volatility += coupling * attributes[parent_idx]["expected_mean"]
        tonic_volatilities.append(tonic_volatility)
        # carried variance λ²/π the child used in its prediction
        log_previous_variances.append(
            jnp.log(
                jnp.maximum(
                    attributes[volatility_child_idx]["temp"]["current_variance"],
                    1e-128,
                )
            )
        )
        be_aux = (1.0 / attributes[volatility_child_idx]["precision"]) + (
            attributes[volatility_child_idx]["mean"]
            - attributes[volatility_child_idx]["expected_mean"]
        ) ** 2
        be_auxs.append(be_aux)
        # volatility prediction error of the child, evaluated at the prediction μ̂
        volatility_prediction_errors.append(
            attributes[volatility_child_idx]["expected_precision"] * be_aux - 1.0
        )

    # Ω_i(x) is only ever used in log-space (``sigmoid``/``logaddexp``). Forming
    # Ω_i(x) = exp(...) explicitly is correct in the forward pass, but once it
    # overflows the backward pass multiplies a zero local partial by an infinite
    # dΩ/dx and returns NaN.
    def log_volatility(x, i):
        """Log of the predicted volatility Ω_i(x) of the child ``i``."""
        return log_time_step + couplings[i] * x + tonic_volatilities[i]

    def variational_energy(x):
        """Variational energy I(x), in log-space to stay gradient-safe."""
        energy = -0.5 * prior_precision * (x - prior_mean) ** 2
        for i in range(n_children):
            log_denom = jnp.logaddexp(log_previous_variances[i], log_volatility(x, i))
            energy = energy - observed[i] * (
                0.5 * log_denom + 0.5 * be_auxs[i] * jnp.exp(-log_denom)
            )
        return energy

    # ----------------------------------------------------------------------------------
    # Expansion 1: quadratic at the prediction, concave part only (eqs. 52-53)
    # ----------------------------------------------------------------------------------
    precision_1 = prior_precision
    mean_shift_1 = prior_precision * (prior_mean - expected_mean)
    for i in range(n_children):
        w_i = sigmoid(log_volatility(expected_mean, i) - log_previous_variances[i])
        precision_1 = precision_1 + observed[i] * (
            0.5 * couplings[i] ** 2 * w_i * (1.0 - w_i)
        )
        mean_shift_1 = mean_shift_1 + observed[i] * (
            0.5 * couplings[i] * w_i * volatility_prediction_errors[i]
        )
    mean_1 = expected_mean + mean_shift_1 / precision_1

    # ----------------------------------------------------------------------------------
    # Expansions 2, ..., n+1: quadratic at each child's Lambert W0 mode (eqs. 54-56)
    # ----------------------------------------------------------------------------------
    precisions, means, is_valid = [precision_1], [mean_1], []
    for i in range(n_children):
        # A zero coupling makes the child independent of the node, so its expansion is
        # dropped below. κ = 1 stands in for it here so that no 0/0 is formed, which
        # would make the gradient NaN even through a dropped expansion.
        has_coupling = couplings[i] != 0.0
        coupling = jnp.where(has_coupling, couplings[i], 1.0)
        pihat_y = prior_precision / coupling**2
        gamma_c = log_time_step + coupling * prior_mean + tonic_volatilities[i]
        # W_arg in log-space, capped at log(float max) as in the MATLAB reference:
        # "W_arg = exp(min(log_W_arg, log(realmax)))"
        log_W_arg = (
            jnp.log(be_auxs[i]) - jnp.log(2.0 * pihat_y) + 0.5 / pihat_y - gamma_c
        )
        log_float_max = jnp.log(jnp.finfo(jnp.result_type(log_W_arg)).max)
        W_arg = jnp.exp(jnp.minimum(log_W_arg, log_float_max))
        y_star = gamma_c + lambert_w0(W_arg) - 0.5 / pihat_y
        x_star = (y_star - log_time_step - tonic_volatilities[i]) / coupling

        # curvature and gradient of the full variational energy at x_star
        full_curvature = prior_precision
        concave_curvature = prior_precision
        gradient = -prior_precision * (x_star - prior_mean)
        for j in range(n_children):
            log_s = log_volatility(x_star, j)
            w_j = sigmoid(log_s - log_previous_variances[j])
            da_j = (
                be_auxs[j] * jnp.exp(-jnp.logaddexp(log_previous_variances[j], log_s))
                - 1.0
            )
            full_curvature = full_curvature + observed[j] * (
                0.5 * couplings[j] ** 2 * w_j * (w_j + (2.0 * w_j - 1.0) * da_j)
            )
            concave_curvature = concave_curvature + observed[j] * (
                0.5 * couplings[j] ** 2 * w_j * (1.0 - w_j)
            )
            gradient = gradient + observed[j] * 0.5 * couplings[j] * w_j * da_j

        precision_2 = jnp.where(
            full_curvature <= 0.0, concave_curvature, full_curvature
        )
        mean_2 = x_star + gradient / precision_2

        # Drop this expansion if the child is unobserved, has no coupling, or the
        # expansion is not finite: it is replaced by expansion 1 and gets a zero weight
        # below. For a single child, this is the MATLAB reference's
        # "if ~isfinite(pi2) || ~isfinite(mu2), pi2 = pi1; mu2 = mu1". This only
        # protects the forward pass: a non-finite expansion still has a NaN gradient.
        is_valid.append(
            (observed[i] != 0)
            & has_coupling
            & jnp.isfinite(precision_2)
            & jnp.isfinite(mean_2)
        )
        precisions.append(jnp.where(is_valid[-1], precision_2, precision_1))
        means.append(jnp.where(is_valid[-1], mean_2, mean_1))

    # ----------------------------------------------------------------------------------
    # Softmax weights (eq. 57) and Gaussian mixture moment matching (eqs. 58-60)
    # ----------------------------------------------------------------------------------
    means_array = jnp.stack(means)
    precisions_array = jnp.stack(precisions)
    energies = [variational_energy(means[0])] + [
        jnp.where(valid, variational_energy(mean), -jnp.inf)
        for mean, valid in zip(means[1:], is_valid)
    ]
    weights = softmax(jnp.stack(energies), axis=0)

    posterior_mean = jnp.sum(weights * means_array, axis=0)
    posterior_variance = jnp.sum(
        weights * (1.0 / precisions_array + (means_array - posterior_mean) ** 2),
        axis=0,
    )

    return 1.0 / posterior_variance, posterior_mean


def _value_children_effective_prior(
    attributes: dict, edges: Edges, node_idx: int, mean_field_updates: bool
) -> tuple[Array, Array]:
    r"""Absorb the value-coupling terms of a node in an effective Gaussian prior.

    Each value child adds a concave quadratic term to the variational energy of the
    node. Section 3.6 of [1]_ absorbs these terms in an effective prior
    :math:`(\tilde{\pi}, \tilde{\mu})`. Here, this prior is the posterior that the
    regular value-coupling update returns when the volatility children are left out,
    so that value children are weighted as in the standard and eHGF updates (relaxed or
    mean-field). Without value children, this is the prediction of the node.

    References
    ----------
    .. [1] Mathys, C., Legrand, N., Waade, P. T., Mikuš, N., & Weber, L. A. (2026).
       Robust volatility updates for Hierarchical Gaussian Filtering.
       arXiv:2605.00966. https://doi.org/10.48550/arXiv.2605.00966
    """
    if edges[node_idx].value_children is None:
        return (
            attributes[node_idx]["expected_precision"],
            attributes[node_idx]["expected_mean"],
        )

    # Restrict the adjacency of the node to its value children, so that the existing
    # value-coupling updates provide the effective prior unchanged.
    value_only_edges = (
        edges[:node_idx]
        + (edges[node_idx]._replace(volatility_children=None),)
        + edges[node_idx + 1 :]
    )
    if mean_field_updates:
        precision = precision_update_mean_field(attributes, value_only_edges, node_idx)
        mean = posterior_update_mean_continuous_node_mean_field(
            attributes, value_only_edges, node_idx, node_precision=precision
        )
    else:
        precision = precision_update(attributes, value_only_edges, node_idx)
        mean = posterior_update_mean_continuous_node(
            attributes, value_only_edges, node_idx, node_precision=precision
        )
    return precision, mean
