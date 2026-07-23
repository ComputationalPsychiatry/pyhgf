from functools import partial

import jax.numpy as jnp
from jax import Array, grad, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx",))
def predict_precision_volatility_level(
    attributes: dict,
    node_idx: int,
) -> tuple[Array, Array]:
    """Predict the precision of the implicit volatility level.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile-state node that will be updated.

    Returns
    -------
    expected_precision_vol :
        The expected (predicted) precision of the volatility level.
    effective_precision_vol :
        The effective precision of the volatility-level prediction.
    """
    time_step = attributes[-1]["time_step"]

    # Get volatility level parameters
    precision_vol = attributes[node_idx]["precision_vol"]
    tonic_volatility_vol = attributes[node_idx]["tonic_volatility_vol"]

    # Compute predicted volatility for the volatility level
    predicted_volatility_vol = time_step * jnp.exp(tonic_volatility_vol)
    predicted_volatility_vol = jnp.where(
        predicted_volatility_vol > 1e-128, predicted_volatility_vol, jnp.nan
    )

    # Expected precision
    expected_precision_vol = 1 / ((1 / precision_vol) + predicted_volatility_vol)

    # Effective precision
    effective_precision_vol = predicted_volatility_vol * expected_precision_vol

    return expected_precision_vol, effective_precision_vol


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_mean_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> Array:
    """Predict the mean of the value level (external facing).

    This uses value parents if they exist.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. For each node, the entry lists its
        value/volatility parents and children.
    node_idx :
        Pointer to the volatile-state node that will be updated.

    Returns
    -------
    expected_mean :
        The expected (predicted) mean of the value level.
    """
    time_step = attributes[-1]["time_step"]

    # List the node's value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Get the drift rate from the node
    driftrate = 0.0

    # Look at the (optional) value parents for this node
    if value_parents_idxs is not None:
        for value_parent_idx, value_coupling_parent in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            # Get the coupling function
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            if coupling_fn is None:
                parent_value = attributes[value_parent_idx]["expected_mean"]
            else:
                parent_value = coupling_fn(
                    attributes[value_parent_idx]["expected_mean"]
                )

            driftrate += value_coupling_parent * parent_value

    # The new expected mean from the previous value
    expected_mean = (
        attributes[node_idx]["autoconnection_strength"] * attributes[node_idx]["mean"]
    ) + (time_step * driftrate)

    return expected_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> tuple[Array, Array, Array]:
    r"""Predict the value-level precisions using the implicit volatility level.

    The implicit volatility level is treated as a full Gaussian parent: the exact
    moment-generating-function correction :math:`\kappa^2 / (2 \, \hat{\pi}_{\mathrm{vol}})`
    is added inside the log-volatility exponent. In addition, each value parent
    contributes a first-order Laplace term
    :math:`(t^{(k)} \, \alpha \, g'(\hat{\mu}_b))^2 / \tilde{\pi}_b` to the marginal
    predictive variance (using each parent's marginal predicted precision,
    ``expected_precision``).

    Two predicted precisions are returned (see
    :class:`pyhgf.typing.LayerState` for the notation table):

    .. math::

        \frac{1}{\hat{\pi}_a^{(k)}}
            = \frac{1}{\pi_a^{(k-1)}} + \Omega_a^{(k)}, \qquad
        \frac{1}{\tilde{\pi}_a^{(k)}}
            = \frac{1}{\hat{\pi}_a^{(k)}}
              + \sum_{b} \frac{ (t^{(k)} \, \alpha_b \, g'(\hat{\mu}_b))^2 }
                              { \tilde{\pi}_b }.

    :math:`\hat{\pi}_a` is the *conditional* predicted precision used by the
    parent's posterior-step Schur complement; :math:`\tilde{\pi}_a` is the *marginal*
    predicted precision consumed by downstream surprise/likelihood code.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`.
    node_idx :
        Index of the volatile state node.

    Returns
    -------
    expected_precision :
        The marginal predicted precision :math:`\tilde{\pi}_a^{(k)}`.
    conditional_expected_precision :
        The conditional predicted precision :math:`\hat{\pi}_a^{(k)}`.
    effective_precision :
        The effective precision :math:`\gamma_a^{(k)} = \Omega_a^{(k)} \tilde{\pi}_a^{(k)}`,
        consumed by the volatility-coupling posterior update.
    """
    time_step = attributes[-1]["time_step"]

    # Get value level parameters
    precision = attributes[node_idx]["precision"]

    # Get volatility level's expected mean and precision (already computed)
    expected_mean_vol = attributes[node_idx]["expected_mean_vol"]
    expected_precision_vol = attributes[node_idx]["expected_precision_vol"]

    # Total volatility = implicit volatility parent's expected mean + closed-form
    # moment-generating-function correction 1/(2 π̂_vol) that arises from
    # marginalising over the volatility parent's Gaussian. The volatility coupling
    # is fixed at 1, and the value level carries no tonic volatility of its own;
    # its diffusion is driven entirely by the volatility level.
    total_volatility = expected_mean_vol + 1.0 / (2.0 * expected_precision_vol)

    # Compute predicted volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Laplace value-coupling correction. The conditional mean of the value level
    # is linearised around μ̂_b via a first-order Taylor expansion of the coupling
    # function g; the variance contribution from each value parent is then
    # (t · α · g'(μ̂_b))² / π̃_b, using the parent's marginal predicted precision
    # π̃_b (= `expected_precision`). The factor t arises because the value-parent
    # contribution to the mean is scaled by the time step in
    # :func:`predict_mean_value_level`.
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

    # Conditional predicted precision π̂_a — the precision of x_a given its
    # value parents (own variance + volatility only), WITHOUT the parent-uncertainty
    # value-coupling term. This is the quantity the parent's posterior-step Schur
    # complement acts on; substituting the marginal there would double-count parent
    # uncertainty.
    conditional_expected_precision = 1 / ((1 / precision) + predicted_volatility)

    # Expected precision = inverse marginal predictive variance.
    expected_precision = 1 / (
        (1 / precision) + predicted_volatility + value_coupling_variance
    )

    # Effective precision (γ): only the volatility-driven part enters γ, since
    # γ is consumed by the volatility-coupling posterior update.
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, conditional_expected_precision, effective_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_prediction(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the expected mean and expected precision of a value-volatility node.

    This node has two internal levels:
    1. Volatility level (implicit, internal)
    2. Value level (external facing)

    The volatility level predicts first, then affects the value level's precision.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile-state node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. For each node, the entry lists its
        value/volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.
    """
    # Store current variance for potential unbounded updates
    attributes[node_idx]["temp"]["current_variance"] = (
        1 / attributes[node_idx]["precision"]
    )

    # 1. PREDICT VOLATILITY LEVEL (implicit internal state)
    expected_precision_vol, effective_precision_vol = (
        predict_precision_volatility_level(attributes, node_idx)
    )

    attributes[node_idx]["expected_mean_vol"] = attributes[node_idx]["mean_vol"]
    attributes[node_idx]["expected_precision_vol"] = expected_precision_vol
    attributes[node_idx]["temp"]["effective_precision_vol"] = effective_precision_vol

    # 2. PREDICT VALUE LEVEL (external facing)
    # Value level's precision depends on volatility level
    expected_precision, conditional_expected_precision, effective_precision = (
        predict_precision_value_level(attributes, edges, node_idx)
    )

    # Value level's mean
    expected_mean = predict_mean_value_level(attributes, edges, node_idx)

    attributes[node_idx]["expected_mean"] = expected_mean

    # Input/leaf override: an observed volatile node has no value children, so it
    # does not undergo a Gaussian random walk between observations. Skip the
    # volatility contribution at the value level and use the prior precision
    # directly, mirroring the continuous-node treatment in
    # :func:`continuous_node_prediction`.
    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]
        # A leaf has no volatility random walk, so the conditional and marginal
        # predicted precisions coincide with the prior precision.
        attributes[node_idx]["temp"]["conditional_expected_precision"] = attributes[
            node_idx
        ]["precision"]
        attributes[node_idx]["temp"]["effective_precision"] = jnp.zeros_like(
            effective_precision
        )
    else:
        attributes[node_idx]["expected_precision"] = expected_precision
        attributes[node_idx]["temp"]["conditional_expected_precision"] = (
            conditional_expected_precision
        )
        attributes[node_idx]["temp"]["effective_precision"] = effective_precision

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision_value_level_mean_field(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> tuple[Array, Array, Array]:
    """Predict the precision of the value level using the implicit volatility level.

    The volatility level's mean modulates the value level's precision.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. For each node, the entry lists its
        value/volatility parents and children.
    node_idx :
        Pointer to the volatile-state node that will be updated.

    Returns
    -------
    expected_precision :
        The expected (marginal) precision of the value level.
    conditional_expected_precision :
        The conditional predicted precision of the value level.
    effective_precision :
        The effective precision of the value-level prediction.
    """
    time_step = attributes[-1]["time_step"]

    precision = attributes[node_idx]["precision"]
    expected_mean_vol = attributes[node_idx]["expected_mean_vol"]

    # Volatility coupling is fixed at 1.
    total_volatility = expected_mean_vol

    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    expected_precision = 1 / ((1 / precision) + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, expected_precision, effective_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_prediction_mean_field(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the expected mean and expected precision of a value-volatility node.

    This node has two internal levels:
    1. Volatility level (implicit, internal)
    2. Value level (external facing)

    The volatility level predicts first, then affects the value level's precision.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile-state node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. For each node, the entry lists its
        value/volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.
    """
    attributes[node_idx]["temp"]["current_variance"] = (
        1 / attributes[node_idx]["precision"]
    )

    # Volatility level prediction (unchanged — no relaxed changes here)
    expected_precision_vol, effective_precision_vol = (
        predict_precision_volatility_level(attributes, node_idx)
    )

    attributes[node_idx]["expected_mean_vol"] = attributes[node_idx]["mean_vol"]
    attributes[node_idx]["expected_precision_vol"] = expected_precision_vol
    attributes[node_idx]["temp"]["effective_precision_vol"] = effective_precision_vol

    # Value level precision — mean_field: no MGF, no Laplace correction
    expected_precision, conditional_expected_precision, effective_precision = (
        predict_precision_value_level_mean_field(attributes, edges, node_idx)
    )

    expected_mean = predict_mean_value_level(attributes, edges, node_idx)
    attributes[node_idx]["expected_mean"] = expected_mean

    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]
        attributes[node_idx]["temp"]["conditional_expected_precision"] = attributes[
            node_idx
        ]["precision"]
        attributes[node_idx]["temp"]["effective_precision"] = jnp.zeros_like(
            effective_precision
        )
    else:
        attributes[node_idx]["expected_precision"] = expected_precision
        attributes[node_idx]["temp"]["conditional_expected_precision"] = (
            conditional_expected_precision
        )
        attributes[node_idx]["temp"]["effective_precision"] = effective_precision

    return attributes
