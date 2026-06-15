# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

from jax import grad, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_continuous_node(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    node_precision: float,
) -> float:
    r"""Continuous mean update without mean-field approximation.

    1. Mean update from value coupling
    ----------------------------------

    Each value child :math:`j` contributes

    .. math::

        \Delta \mu_b \;\mathrel{+}=\;
            \frac{\kappa_j \, g'_{j,b}(\mu_b^{(k-1)}) \, g_{a,j}}{\pi_b} \, \delta_j^{(k)},
            \qquad
        g_{a,j} \,=\, \frac{\hat{\pi}_j^{(k)} \, \pi_j^{(k)}}{\hat{\pi}_j^{(k)} + \pi_{y,j}},
            \qquad
        \pi_{y,j} \,=\, \pi_j^{(k)} - \tilde{\pi}_j^{(k)},

    where :math:`\kappa_j` is the value-coupling strength, :math:`g'_{j,b}` is the
    derivative of the coupling function (1 for linear coupling),
    :math:`\hat{\pi}_j` is the child's *conditional* predicted precision (stored as
    ``child.temp["conditional_expected_precision"]``), :math:`\tilde{\pi}_j` its
    *marginal* predicted precision (``child.expected_precision``), and
    :math:`\delta_j^{(k)}` is the value prediction error computed by
    :py:func:`pyhgf.updates.prediction_error.continuous.continuous_node_value_prediction_error`.
    The smoothing form :math:`g_{a,j}` is the joint-Gaussian (RTS-smoother) gain
    derived from the structured-Gaussian variational posterior on the value-coupling
    edge; for leaves and non-Gaussian children :math:`\pi_{y,j} = 0` and
    :math:`g_{a,j}` collapses to the marginal :math:`\tilde{\pi}_j`, recovering the
    canonical gain :math:`\hat{\pi}_j \, \pi_j / \pi_j = \tilde{\pi}_j`.

    2. Mean update from volatility coupling
    ---------------------------------------

    .. math::

        \Delta \mu_b \mathrel{+}= \frac{1}{2 \, \pi_b}
            \sum_{j} \kappa_j \, \gamma_j^{(k)} \, \Delta_j^{(k)},

    where :math:`\kappa_j` is the volatility-coupling strength between the
    volatility parent and the volatility child :math:`j`,
    :math:`\gamma_j^{(k)} = \Omega_j^{(k)} \, \tilde{\pi}_j^{(k)}` is the effective
    precision of the prediction (computed in the prediction step,
    :py:func:`pyhgf.updates.prediction.predict_precision`), and the volatility
    prediction error is

    .. math::

        \Delta_j^{(k)} = \frac{\tilde{\pi}_j^{(k)}}{\pi_j^{(k)}} +
            \tilde{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. For each node, the entry lists its
        value/volatility parents and children.
    node_idx :
        Pointer to the value parent node that will be updated.
    node_precision :
        The precision :math:`\pi_b` of the node to divide by once after summing the
        precision-weighted PEs across children. Depending on the volatility update
        schedule this is either the just-updated posterior precision (standard
        scheme) or the marginal predicted precision (eHGF scheme).

    Returns
    -------
    posterior_mean :
        The new posterior mean.

    See Also
    --------
    posterior_update_precision_continuous_node

    """
    # sum the prediction errors from both value and volatility coupling
    (
        value_precision_weigthed_prediction_error,
        volatility_precision_weigthed_prediction_error,
    ) = (0.0, 0.0)

    # Value coupling updates - update the mean of a value parent
    # ----------------------------------------------------------
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # get the value prediction error (VAPE)
            # if this is jnp.nan (no observation) set the VAPE to 0.0
            value_prediction_error = attributes[value_child_idx]["temp"][
                "value_prediction_error"
            ]

            # cancel the prediction error if the child value was not observed
            value_prediction_error *= attributes[value_child_idx]["observed"]

            # get differential of coupling function with value children
            if coupling_fn is None:  # linear coupling
                coupling_fn_prime = 1
            else:  # non-linear coupling
                # Compute the derivative of the coupling function
                coupling_fn_prime = grad(coupling_fn)(attributes[node_idx]["mean"])

            # Coupling gain precision g_a. From the joint (x_a, x_b) Gaussian the
            # exact marginal-mean gain is
            #     g_a = π̂_a · π_a / (π̂_a + π_y),    π_y = π_a − π̃_a,
            # accumulated across children and divided once by `node_precision` (π_b)
            # after the loop — this is what makes the multi-child mean exact rather
            # than a sum of independent RTS gains. For leaves and non-Gaussian
            # children π_y = 0 and g_a collapses to π̃_a, recovering the canonical
            # gain (π̃_a is what `child.expected_precision` stores).
            child_node_type = edges[value_child_idx].node_type
            child_is_gaussian_interior = child_node_type in (2, 6) and (
                edges[value_child_idx].value_children is not None
                or edges[value_child_idx].volatility_children is not None
            )
            child_expected_precision = attributes[value_child_idx]["expected_precision"]
            if child_is_gaussian_interior:
                child_precision = attributes[value_child_idx]["precision"]
                pi_y = child_precision - child_expected_precision
                child_cond = attributes[value_child_idx]["temp"][
                    "conditional_expected_precision"
                ]
                gain_precision = child_cond * child_precision / (child_cond + pi_y)
            else:
                gain_precision = child_expected_precision

            # sum the precision weigthed prediction errors over all children
            value_precision_weigthed_prediction_error += (
                (value_coupling * gain_precision * coupling_fn_prime) / node_precision
            ) * value_prediction_error

    # Volatility coupling updates - update the mean of a volatility parent
    # --------------------------------------------------------------------
    if edges[node_idx].volatility_children is not None:
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            # get the volatility prediction error (VOPE)
            volatility_prediction_error = attributes[volatility_child_idx]["temp"][
                "volatility_prediction_error"
            ]

            # retrieve the effective precision (γ)
            # computed during the prediction step
            effective_precision = attributes[volatility_child_idx]["temp"][
                "effective_precision"
            ]

            # the precision weigthed prediction error
            precision_weigthed_prediction_error = (
                volatility_coupling * effective_precision * volatility_prediction_error
            )

            # weight using the node's precision
            precision_weigthed_prediction_error *= 1 / (2 * node_precision)

            # cancel the prediction error if the child value was not observed
            precision_weigthed_prediction_error *= attributes[volatility_child_idx][
                "observed"
            ]

            # aggregate over volatility children
            volatility_precision_weigthed_prediction_error += (
                precision_weigthed_prediction_error
            )

    # Compute the new posterior mean
    # using value prediction errors from both value and volatility coupling
    posterior_mean = (
        attributes[node_idx]["expected_mean"]
        + value_precision_weigthed_prediction_error
        + volatility_precision_weigthed_prediction_error
    )

    return posterior_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_continuous_node_mean_field(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    node_precision: float,
) -> float:
    r"""Continuous mean update with mean-field approximation [1]_.

    1. Mean update from value coupling.

    The new mean of a state node :math:`b` value coupled with other input and/or state
    nodes :math:`j` at time :math:`k` is given by:

    For linear value coupling:

    .. math::
        \mu_b^{(k)} =  \hat{\mu}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \frac{\kappa_j \hat{\pi}_j^{(k)}}{\pi_b} \delta_j^{(k)}

    Where :math:`\kappa_j` is the volatility coupling strength between the child node
    and the state node and :math:`\delta_j^{(k)}` is the value prediction error that
    was computed beforehand by
    :py:func:`pyhgf.updates.prediction_errors.continuous.continuous_node_value_prediction_error`.

    For non-linear value coupling:

    .. math::
        \mu_b^{(k)} =  \hat{\mu}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \frac{\kappa_j g'_{j,b}({\mu}_b^{(k-1)}) \hat{\pi}_j^{(k)}}{\pi_b}
            \delta_j^{(k)}


    2. Mean update from volatility coupling.

    The new mean of a state node :math:`b` volatility coupled with other input and/or
    state nodes :math:`j` at time :math:`k` is given by:

    .. math::
        \mu_b^{(k)} = \hat{\mu}_b^{(k)} + \frac{1}{2\pi_b}
          \sum_{j=1}^{N_{children}} \kappa_j \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error is given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the effective precision of the prediction, given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    with :math:`\Omega_j^{(k)}` the predicted volatility computed in the prediction
    step :py:func:`pyhgf.updates.prediction.predict_precision`.

    If the child node is a continuous state node, the volatility prediction error
    :math:`\Delta_j^{(k)}` was computed by
    :py:func:`pyhgf.updates.prediction_errors.continuous.continuous_node_volatility_prediction_error`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    node_precision :
        The precision of the node. Depending on the kind of volatility update, this
        value can be the expected precision (ehgf), or the posterior from the update
        (standard).

    Returns
    -------
    posterior_mean :
        The new posterior mean.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    posterior_update_precision_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    (
        value_precision_weigthed_prediction_error,
        volatility_precision_weigthed_prediction_error,
    ) = (0.0, 0.0)

    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            value_prediction_error = attributes[value_child_idx]["temp"][
                "value_prediction_error"
            ]
            value_prediction_error *= attributes[value_child_idx]["observed"]

            if coupling_fn is None:
                coupling_fn_prime = 1
            else:
                coupling_fn_prime = grad(coupling_fn)(attributes[node_idx]["mean"])

            value_precision_weigthed_prediction_error += (
                (
                    value_coupling
                    * attributes[value_child_idx]["expected_precision"]
                    * coupling_fn_prime
                )
                / node_precision
            ) * value_prediction_error

    if edges[node_idx].volatility_children is not None:
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            volatility_prediction_error = attributes[volatility_child_idx]["temp"][
                "volatility_prediction_error"
            ]
            effective_precision = attributes[volatility_child_idx]["temp"][
                "effective_precision"
            ]
            precision_weigthed_prediction_error = (
                volatility_coupling * effective_precision * volatility_prediction_error
            )
            precision_weigthed_prediction_error *= 1 / (2 * node_precision)
            precision_weigthed_prediction_error *= attributes[volatility_child_idx][
                "observed"
            ]
            volatility_precision_weigthed_prediction_error += (
                precision_weigthed_prediction_error
            )

    posterior_mean = (
        attributes[node_idx]["expected_mean"]
        + value_precision_weigthed_prediction_error
        + volatility_precision_weigthed_prediction_error
    )

    return posterior_mean
