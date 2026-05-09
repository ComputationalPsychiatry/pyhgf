# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized posterior update for volatile node layers."""

from typing import Callable

import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing import LayerState


def vectorized_posterior_update_precision_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn_grad: Callable,
) -> jnp.ndarray:
    """Update the precision of the value level for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.posterior_update_value_level.posterior_update_precision_value_level`.

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
    coupling_fn_grad :
        Gradient of the coupling function.

    Returns
    -------
    jnp.ndarray
        Posterior precision for each node in the parent layer.
    """
    # Coupling derivatives at parent expected means
    coupling_prime = vmap(coupling_fn_grad)(layer.expected_mean)

    # Coupling second derivative (for second-order EKF correction)
    coupling_second = vmap(jgrad(coupling_fn_grad))(layer.expected_mean)

    # First-order term: sum_i(w_ij^2 * child_prec_i) * f'(m_j)^2
    precision_contrib_1 = jnp.matmul(weights.T**2, child.expected_precision) * (
        coupling_prime**2
    )

    # Second-order correction: -f''(m_j) * sum_i(w_ij * child_prec_i * vpe_i)
    sum_pi_vpe = jnp.matmul(
        weights.T, child.expected_precision * child.value_prediction_error
    )
    precision_contrib_2 = -coupling_second * sum_pi_vpe

    posterior_precision = (
        layer.expected_precision + precision_contrib_1 + precision_contrib_2
    )

    return posterior_precision


def vectorized_posterior_update_mean_value_level(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn_grad: Callable,
    posterior_precision: jnp.ndarray,
) -> jnp.ndarray:
    """Update the mean of the value level for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.posterior_update_value_level.posterior_update_mean_value_level`.

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
    coupling_fn_grad :
        Gradient of the coupling function.
    posterior_precision :
        Already-updated posterior precision for the parent layer.

    Returns
    -------
    jnp.ndarray
        Posterior mean for each node in the parent layer.
    """
    # Coupling derivatives at parent expected means
    coupling_prime = vmap(coupling_fn_grad)(layer.expected_mean)

    # Precision-weighted PE from children
    weighted_pe = (
        jnp.matmul(weights.T, child.expected_precision * child.value_prediction_error)
        * coupling_prime
        / posterior_precision
    )

    posterior_mean = layer.expected_mean + weighted_pe

    return posterior_mean


def vectorized_layer_posterior_update(
    layer: LayerState,
    child: LayerState,
    weights: jnp.ndarray,
    coupling_fn_grad: Callable,
    parent_has_constant: bool = False,
    max_posterior_precision: float = 1e10,
) -> LayerState:
    """Update the value-level posterior for all nodes in a parent layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.volatile_node_posterior_update.volatile_node_posterior_update`.
    It updates the value level precision first, then the mean.

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
    coupling_fn_grad :
        Gradient of the coupling function.
    parent_has_constant :
        If True, the last column of *weights* corresponds to the constant input node and
        is stripped before computing the posterior update.
    max_posterior_precision :
        Upper bound applied to the posterior precision. Default ``1e10``.

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
        vectorized_posterior_update_precision_value_level(
            layer, child, weights, coupling_fn_grad
        ),
        a_min=layer.expected_precision,
        a_max=max_posterior_precision,
    )

    posterior_mean = vectorized_posterior_update_mean_value_level(
        layer, child, weights, coupling_fn_grad, posterior_precision
    )

    return layer._replace(
        precision=posterior_precision,
        mean=posterior_mean,
    )
