# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import grad, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_continuous_node(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
    node_precision: float,
) -> float:
    r"""Update the mean of a state node using the value prediction errors.

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

            # expected precisions from the value children
            # sum the precision weigthed prediction errors over all children
            value_precision_weigthed_prediction_error += (
                (
                    (
                        value_coupling
                        * attributes[value_child_idx]["expected_precision"]
                        * coupling_fn_prime
                    )
                    / node_precision
                )
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
