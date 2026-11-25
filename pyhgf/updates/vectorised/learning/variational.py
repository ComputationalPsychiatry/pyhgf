# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""The ``"variational"`` gradient of the Kronecker-factored weight belief.

:func:`vectorised_variational_gradient` is the descent gradient of one weight matrix
under the posterior of the beliefs, the M-step of variational expectation maximisation.
It is not a rank-one product, so it is assembled per weight matrix rather than from the
factors of :mod:`~pyhgf.updates.vectorised.learning.factors`.
:func:`relaxed_effective_precision` is the precision of a child's prediction under the
observations, which the gradient shrinks the weights by and the Kronecker-factored
weight belief uses as its output increment.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing.vectorised import LayerState
from pyhgf.updates.vectorised.elements import _EPS

#: The ``gradient`` setting of ``"kronecker"`` whose gradient is not rank-one and is
#: assembled by :func:`vectorised_variational_gradient` instead of the factors of
#: :func:`~pyhgf.updates.vectorised.learning.factors.learning_weights_vectorised`.
VARIATIONAL_KIND: str = "variational"


def relaxed_effective_precision(
    child_state: LayerState,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Relaxed effective precision of a child layer, one value per unit.

    .. math::

        s_a = \frac{\bar{\pi}_a\, \pi^y_a}{\bar{\pi}_a + \pi^y_a},
        \qquad \pi^y_a = \max(\pi_a - \hat{\pi}_a, 0),

    with :math:`\bar{\pi}_a` the conditional expected precision of the child and
    :math:`\pi^y_a` the evidence it received from below. :math:`s_a` is the precision
    of the prediction of unit :math:`a` under the observations: one over the sum of the
    variance of the prediction given the parents, :math:`1 / \bar{\pi}_a`, and the
    variance the evidence leaves, :math:`1 / \pi^y_a`. It is :math:`\bar{\pi}_a` for the
    clamped observed layer (:math:`\pi^y_a \to \infty`, the observations are known
    exactly) and under ``mean_field_updates``, whose posterior has no covariance
    between layers.

    Parameters
    ----------
    child_state :
        The child layer after the update sweep, as its parent sees it.
    child_is_input_layer :
        Whether the child is the clamped observed layer.
    mean_field_updates :
        Whether the beliefs follow the mean-field updates.

    Returns
    -------
    jnp.ndarray
        :math:`s_a`, shape ``(n_child,)``.
    """
    conditional = child_state.conditional_expected_precision
    if child_is_input_layer or mean_field_updates:
        return conditional
    evidence = jnp.maximum(child_state.precision - child_state.expected_precision, 0.0)
    return conditional * evidence / jnp.maximum(conditional + evidence, _EPS)


def vectorised_variational_gradient(
    parent_state: LayerState,
    child_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Descent gradient of one weight matrix under the posterior of the beliefs.

    The weights :math:`W` predict child :math:`a` from the parents :math:`b` with
    precision :math:`\bar{\pi}_a` (the child's conditional expected precision),
    :math:`x_a \sim \mathcal{N}(\sum_b W_{ab}\, g(x_b) + c_a, 1/\bar{\pi}_a)`. This
    returns the gradient of the expected log-likelihood of that factor under the
    posterior of the beliefs (the M-step of variational expectation maximisation),
    linearising :math:`g` at the posterior means:

    .. math::

        \frac{\partial}{\partial W_{ab}} = \bar{\pi}_a \Big( \rho_a\, g(\mu_b)
            + g'(\mu_b)\, \mathrm{Cov}(x_a, x_b)
            - W_{ab}\, g'(\mu_b)^2\, \mathrm{Var}(x_b) \Big),

    with the units of the parent layer independent of each other, :math:`\rho_a` the
    posterior mean of the child's noise :math:`x_a - \sum_b W_{ab}\, g(x_b) - c_a`,
    and :math:`\mathrm{Var}(x_b) = 1/\pi_b` the parent's posterior variance. On the
    relaxed edge the posterior covariance is :math:`\mathrm{Cov}(x_a, x_b) =
    \bar{\pi}_a W_{ab}\, g'(\mu_b) / ((\bar{\pi}_a + \pi^y_a)\, \pi_b)`, with
    :math:`\pi^y_a = \pi_a - \hat{\pi}_a` the evidence the child received from below,
    so the two variance terms combine into

    .. math::

        \frac{\partial}{\partial W_{ab}} = \bar{\pi}_a\, \rho_a\, g(\mu_b)
            - W_{ab}\, s_a\, \frac{g'(\mu_b)^2}{\pi_b},
        \qquad s_a = \frac{\bar{\pi}_a\, \pi^y_a}{\bar{\pi}_a + \pi^y_a}.

    :math:`s_a` is the relaxed effective precision of the child. It is
    :math:`\bar{\pi}_a` for the clamped observed layer (:math:`\pi^y_a \to \infty`, the
    observations are known exactly) and under ``mean_field_updates``, whose posterior
    has no covariance between layers. The second term shrinks each weight in
    proportion to the posterior variance of its parent. The bias column takes
    :math:`\bar{\pi}_a\, \rho_a`.

    The noise is read at the parents' posterior means. Given them, the child's prior
    is :math:`\mathcal{N}(\hat{\mu}_a + \Delta_a, 1/\bar{\pi}_a)`, with
    :math:`\Delta_a = \sum_b W_{ab}\, (g(\mu_b) - g(\hat{\mu}_b))` the shift of its
    prediction, and the evidence from below brings the precision :math:`\pi^y_a` and
    the information :math:`\pi_a \delta_a`, with :math:`\delta_a = \mu_a -
    \hat{\mu}_a` the child's prediction error. Combining the two gives

    .. math::

        \bar{\pi}_a\, \rho_a = \frac{\bar{\pi}_a}{\bar{\pi}_a + \pi^y_a}
            \big( \pi_a \delta_a - \pi^y_a \Delta_a \big)
            = \frac{\bar{\pi}_a\, \pi_a}{\bar{\pi}_a + \pi^y_a}\, \delta_a
            - s_a\, \Delta_a,

    the prediction error the child sends its parents, weighted by the relaxed gain of
    :func:`~pyhgf.updates.vectorised.volatile.posterior.vectorised_posterior_update_mean_value_level`,
    less the part of it that the parents' moves explain. For the clamped observed
    layer and under ``mean_field_updates`` this is the difference of the posterior
    means, :math:`\rho_a = \mu_a - \sum_b W_{ab}\, g(\mu_b) - c_a`. On a hidden child
    with more than one unit the two differ: the child's mean is formed with the
    diagonal of its predicted covariance, while the parents' moves also carry the
    evidence its sibling units received, so the difference of the means keeps a share
    of those moves that does not shrink as the hidden precisions grow. With
    :math:`\rho_a` read at the parents' posterior means, under the relaxed updates,
    the gradient tends to that of the forward pass (the ``"backprop"`` gradient) as the
    conditional expected precisions of the hidden layers grow, provided the variance
    of the top layer, clamped to the predictors, stays negligible against theirs.

    Parameters
    ----------
    parent_state :
        The parent layer after the update sweep (posterior means and precisions).
    child_state :
        The child layer after the update sweep, as its parent sees it: the topmost
        slice when the child is a ``LayerStack``.
    weights :
        The weight matrix predicting the child, shape ``(n_child, n_parent)`` or
        ``(n_child, n_parent + 1)`` with the bias as the last column.
    coupling_fn :
        Coupling function applied to the parent means.
    parent_has_constant :
        Whether the last column of ``weights`` is the bias.
    child_is_input_layer :
        Whether the child is the clamped observed layer.
    mean_field_updates :
        Whether the beliefs follow the mean-field updates.

    Returns
    -------
    jnp.ndarray
        The descent gradient (the negative of the derivative above), shape of
        ``weights``, so that it composes with ``optax.apply_updates`` as the other
        learning kinds do.
    """
    n_parents = parent_state.mean.shape[-1]
    matrix = weights[:, :n_parents]
    activation = coupling_fn(parent_state.mean)
    prediction = jnp.matmul(matrix, activation)
    if parent_has_constant:
        prediction = prediction + weights[:, -1]
    conditional = child_state.conditional_expected_precision
    shrink = relaxed_effective_precision(
        child_state, child_is_input_layer, mean_field_updates
    )
    if child_is_input_layer or mean_field_updates:
        # The noise is the difference of the posterior means: the observations are
        # known exactly, and the mean-field posterior has no covariance between
        # layers.
        weighted_residual = conditional * (child_state.mean - prediction)
    else:
        # The child's noise given its parents' posterior means: the prediction error
        # it sent up, weighted by the relaxed gain, less s_a times the shift of its
        # prediction.
        evidence = jnp.maximum(
            child_state.precision - child_state.expected_precision, 0.0
        )
        gain = (
            conditional
            * child_state.precision
            / jnp.maximum(conditional + evidence, _EPS)
        )
        weighted_residual = gain * (
            child_state.mean - child_state.expected_mean
        ) - shrink * (prediction - child_state.expected_mean)

    slope = vmap(jgrad(coupling_fn))(parent_state.mean)
    parent_variance = slope**2 / jnp.maximum(parent_state.precision, _EPS)

    ascent = weighted_residual[:, None] * activation[None, :] - matrix * (
        shrink[:, None] * parent_variance[None, :]
    )
    if parent_has_constant:
        ascent = jnp.concatenate([ascent, weighted_residual[:, None]], axis=1)
    return -jnp.where(jnp.isfinite(ascent), ascent, 0.0)
