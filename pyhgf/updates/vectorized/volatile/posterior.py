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
    child_is_input_layer: bool = False,
) -> jnp.ndarray:
    r"""Update the precision of the value level for all nodes in a layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.posterior.volatile.posterior_update_value_level.posterior_update_precision_value_level`.

    Implements the *posterior-step (smoothing) correction* of the relaxed HGF on
    value-coupling edges. Lifting the mean-field delta-collapse approximation
    :math:`q(x_a, x_b) = q(x_a)\,q(x_b)` to a structured Gaussian on the value-coupling
    edge and applying the Schur complement yields the substitution

    .. math::

        \hat\pi_a^{(k)} \;\longmapsto\;
        \hat\pi_a^{(k)} \, \frac{\pi_a^{(k)} - \hat\pi_a^{(k)}}{\pi_a^{(k)}}

    in the canonical value-coupling contribution to :math:`\pi_b^{(k)}`. The
    multiplicative factor :math:`(\pi_a - \hat\pi_a)/\pi_a \in [0, 1]` is the fraction
    of the child's posterior precision attributable to bottom-up evidence (rather than
    to its own predicted prior). The same factor scales both the :math:`g'^2` and the
    :math:`g''\,\delta_a` contributions. Reduces to the canonical formula when the child
    is fully observed (:math:`\pi_a \gg \hat\pi_a`); returns no update when the child
    gained no bottom-up information (:math:`\pi_a = \hat\pi_a`).

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
    child_is_input_layer :
        If True, the child is the clamped observation leaf (binary or
        continuous output). In that case the paper's Limit 3 applies
        (``π_a → ∞``) and the smoothing correction reduces to the canonical
        contribution ``π̂_a``. We bypass the ``(π_a − π̂_a)/π_a`` formula
        here because pyhgf's leaf convention sets ``π_a = π̂_a`` for the
        binary PE and never updates ``π_a`` for a continuous leaf — neither
        signals "info gained", so the smoothing form would incorrectly
        zero out the leaf's contribution. Defaults to ``False`` (interior
        child, full smoothing correction).

    Returns
    -------
    jnp.ndarray
        Posterior precision for each node in the parent layer.
    """
    # Coupling derivatives at parent expected means
    coupling_prime = vmap(coupling_fn_grad)(layer.expected_mean)

    # Coupling second derivative (for second-order EKF correction)
    coupling_second = vmap(jgrad(coupling_fn_grad))(layer.expected_mean)

    # Effective child precision under the smoothing correction:
    #     π̂_a · (π_a − π̂_a) / π_a
    # By the time this kernel runs for an interior child, the child layer's
    # bottom-up posterior update has already populated ``child.precision``;
    # the prediction-step value remains in ``child.expected_precision``.
    # Their difference (π_a − π̂_a) is the information actually gained at the
    # child during the bottom-up sweep — exactly the quantity the
    # structured-Gaussian + Schur derivation identifies as the fresh
    # evidence the parent should be credited with.
    #
    # For a clamped leaf (binary/continuous observation layer), pyhgf's
    # representational convention does not store the "infinite" posterior
    # precision that the observation logically carries — ``π_a`` stays
    # equal to ``π̂_a``. We therefore short-circuit to the canonical
    # contribution ``π̂_a``, matching the paper's Limit 3 (``π_a → ∞``).
    if child_is_input_layer:
        effective_child_precision = child.expected_precision
    else:
        effective_child_precision = (
            child.expected_precision
            * (child.precision - child.expected_precision)
            / child.precision
        )

    # First-order term: sum_i(w_ij^2 * effective_child_prec_i) * f'(m_j)^2
    precision_contrib_1 = jnp.matmul(weights.T**2, effective_child_precision) * (
        coupling_prime**2
    )

    # Second-order correction: -f''(m_j) * sum_i(w_ij * effective_child_prec_i * vpe_i)
    # The same multiplicative factor (π_a − π̂_a)/π_a applies.
    sum_pi_vpe = jnp.matmul(
        weights.T, effective_child_precision * child.value_prediction_error
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
    child_is_input_layer: bool = False,
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
    child_is_input_layer :
        If True, the child is the clamped observation leaf (binary or
        continuous output). The smoothing correction reduces to the
        canonical contribution (paper's Limit 3). Defaults to ``False``.

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
            layer,
            child,
            weights,
            coupling_fn_grad,
            child_is_input_layer=child_is_input_layer,
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
