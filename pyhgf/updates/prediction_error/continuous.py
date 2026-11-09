# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import grad, jit

from pyhgf.typing import Edges
from pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node import (
    _has_observations,
)
from pyhgf.updates.prediction.continuous import is_noise_child


@partial(jit, static_argnames=("node_idx", "edges"))
def continuous_node_value_prediction_error(
    attributes: dict,
    node_idx: int,
    edges: Edges = (),
) -> dict:
    r"""Compute the value prediction error of a state node.

    The value prediction error :math:`\delta_j^{(k)}` of a continuous state node is
    given by:

    .. math::

        \delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}

    See [1]_ for more details.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the value parent node that will be updated.
    edges :
        Unused. Kept for API compatibility with callers that pass edges.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_volatility_prediction_error, continuous_node_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1

    """
    # value prediction error
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # send to the value parent node for later use in the update step
    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error

    return attributes


@partial(jit, static_argnames=("node_idx"))
def continuous_node_volatility_prediction_error(
    attributes: dict, node_idx: int
) -> dict:
    r"""Compute the volatility prediction error of a state node.

    The volatility prediction error :math:`\Delta_j^{(k)}` of a state node
    :math:`j` is given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    Every volatility parent of the node receives the full :math:`\Delta_j^{(k)}`, as
    in the unbounded update. See [1]_ for more details. An input node with noise
    parents sends the noise prediction error of
    :func:`continuous_node_noise_prediction_error` instead.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_value_prediction_error, continuous_node_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1

    """
    # compute the volatility prediction error (VOPE)
    volatility_prediction_error = (
        (attributes[node_idx]["expected_precision"] / attributes[node_idx]["precision"])
        + attributes[node_idx]["expected_precision"]
        * (attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2
        - 1
    )

    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        volatility_prediction_error
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction_error(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Store the prediction errors of a continuous state node.

    A node with value or volatility children also records whether at least one of
    them was observed at this step, in its ``observed`` flag. Its parents then treat
    it as unobserved after a step without observations: they receive no prediction
    error from it and keep their predicted precision (see
    :func:`pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node.precision_update_missing_values`).
    The flag of a node without children is the one set by the observation step.

    See [1]_ for more details.

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
        Pointer to the continuous node.
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
    continuous_node_volatility_prediction_error, continuous_node_value_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    # A node with children is observed when at least one of its children was
    # ------------------------------------------------------------------------
    if (
        edges[node_idx].value_children is not None
        or edges[node_idx].volatility_children is not None
    ):
        attributes[node_idx]["observed"] = jnp.where(
            _has_observations(attributes, edges, node_idx), 1, 0
        )

    # Store value prediction errors
    # -----------------------------
    attributes = continuous_node_value_prediction_error(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    # Store volatility prediction errors
    # ----------------------------------
    # An input node with noise parents sends them a noise prediction error, which is
    # computed after its value parents' update
    # (:func:`continuous_node_noise_prediction_error`).
    if not is_noise_child(edges, node_idx):
        attributes = continuous_node_volatility_prediction_error(
            attributes=attributes, node_idx=node_idx
        )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_noise_prediction_error(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    r"""Compute the noise prediction error of an input node.

    The noise prediction error of an input node :math:`u` with value parent
    :math:`p` is Eq. 72 of [1]_:

    .. math::

        \epsilon_u^{(k)} = \frac{\hat{\pi}_u^{(k)}}{\pi_p^{(k)}}
            + \hat{\pi}_u^{(k)} \left( u^{(k)} - \mu_p^{(k)} \right)^2 - 1,

    with :math:`\hat{\pi}_u^{(k)}` the precision of the observation noise
    (:func:`pyhgf.updates.prediction.continuous.predict_noise_precision`). Unlike
    the volatility prediction error, it does not compare :math:`u` with its
    prediction but with the posterior :math:`(\mu_p^{(k)}, \pi_p^{(k)})` of the
    value parent: :math:`1/\pi_p + (u - \mu_p)^2` is the expected squared
    observation noise :math:`(u - x_p)^2` under that posterior. This step must
    therefore run after the posterior update of every value parent of :math:`u`,
    and before the update of its noise parents, which read it with the effective
    precision :math:`\gamma_u = 1` (Eqs. 73 to 78 of [1]_).

    With several value parents :math:`b`, coupling strengths :math:`\alpha_b`,
    coupling functions :math:`g_b` and time step :math:`t`, the expected squared
    noise is evaluated to first order under the posteriors of all value parents,

    .. math::

        \epsilon_u^{(k)} = \hat{\pi}_u^{(k)} \left( \sum_b
            \frac{(t \alpha_b g_b'(\mu_b^{(k)}))^2}{\pi_b^{(k)}}
            + \left( u^{(k)} - m_u^{(k)} \right)^2 \right) - 1,

    where :math:`m_u^{(k)}` is the predicted mean of :math:`u` with every value
    parent moved from :math:`\hat{\mu}_b^{(k)}` to :math:`\mu_b^{(k)}`. With one
    linear value parent, :math:`\alpha = 1`, :math:`t = 1`, no drift and an
    autoconnection strength of 0 (the defaults of an input node), this is Eq. 72.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the input node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`.

    Returns
    -------
    attributes :
        The attributes, with the noise prediction error stored in the input node's
        ``temp["volatility_prediction_error"]``.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    time_step = attributes[-1]["time_step"]

    # Predicted mean of the input with the value parents at their posterior means,
    # and the variance their posteriors add (first order in the coupling function).
    posterior_mean = attributes[node_idx]["expected_mean"]
    posterior_variance = 0.0
    value_parents = edges[node_idx].value_parents
    if value_parents is not None:
        for value_parent_idx, value_coupling in zip(
            value_parents,
            attributes[node_idx]["value_coupling_parents"],
        ):
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            parent_mean = attributes[value_parent_idx]["mean"]
            parent_expected_mean = attributes[value_parent_idx]["expected_mean"]
            if coupling_fn is None:
                shift = parent_mean - parent_expected_mean
                coupling_fn_prime = 1.0
            else:
                shift = coupling_fn(parent_mean) - coupling_fn(parent_expected_mean)
                coupling_fn_prime = grad(coupling_fn)(parent_mean)
            posterior_mean += time_step * value_coupling * shift
            posterior_variance += (
                time_step * value_coupling * coupling_fn_prime
            ) ** 2 / attributes[value_parent_idx]["precision"]

    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        attributes[node_idx]["expected_precision"]
        * (posterior_variance + (attributes[node_idx]["mean"] - posterior_mean) ** 2)
        - 1.0
    )

    return attributes
