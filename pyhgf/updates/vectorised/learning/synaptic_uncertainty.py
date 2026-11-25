# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Weight-belief rule of the ``"synaptic_uncertainty"`` learning kind.

Each weight carries a Gaussian belief: its mean is the weight, and its precision
accumulates the curvature the data impose on it. The mean steps along the gradient
divided by that precision. The rule and its settings are stated in
:func:`resolve_synaptic_uncertainty_settings`.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple, Optional

import jax.numpy as jnp

from pyhgf.typing.vectorised import VectorisedNetwork
from pyhgf.updates.vectorised.elements import _EPS, _edge_mask

#: Settings of the weight-belief rule and their defaults, as
#: ``learning_kwargs`` accepts them (see :func:`resolve_synaptic_uncertainty_settings`).
SYNAPTIC_UNCERTAINTY_DEFAULTS: dict = {
    "window": None,
    "prior_variance": 1.0,
    "prior_mean": 0.0,
    "learning_rate": 1.0,
    "increment_scale": 1.0,
}


class SynapticUncertaintySettings(NamedTuple):
    r"""Validated settings of the weight-belief rule.

    Built from a ``learning_kwargs`` dictionary by
    :func:`resolve_synaptic_uncertainty_settings`; every field is documented there.
    """

    window: float
    prior_variance: float
    prior_mean: float
    learning_rate: float
    increment_scale: float
    prior_precision: float


def resolve_synaptic_uncertainty_settings(
    learning_kwargs: Optional[dict] = None,
) -> SynapticUncertaintySettings:
    r"""Validate the weight-belief rule's settings and fill in the defaults.

    Each weight carries a Gaussian belief: the weight itself is the belief's
    mean and the layer's ``weights_precision_delta`` its precision above the
    prior. One update step does two things.

    **Mean.** The update is :math:`\Delta w = -\alpha g/\pi + \text{reversion}`,
    the incoming descent gradient :math:`g` divided by the weight's precision
    and scaled by ``learning_rate`` :math:`\alpha`, plus a pull of the mean
    toward ``prior_mean``. At the start :math:`\pi = \pi_p` everywhere, so the
    rule begins as plain gradient descent at rate
    :math:`\alpha \times \texttt{prior\_variance}` and departs from it only as
    precision accumulates.

    **Precision.** The increment is the curvature the data impose on that
    weight, delivered as the importance factors of
    :func:`~pyhgf.updates.vectorised.learning.factors.learning_weights_vectorised` or
    as a full increment matrix. The child-side factor :math:`p` of the increment
    :math:`p \otimes h^2` is the evidence precision carried up the network from its
    clamped layer, so a network whose clamped layer is continuous seeds it with that
    layer's own precision, one number for every unit. The precision then relaxes
    toward the prior in precision form,
    :math:`\pi \leftarrow \pi + H - (\pi - \pi_p)/N` with
    :math:`N = \texttt{window}`. The fixed point is
    :math:`\pi^\ast = N \bar{H} + \pi_p`, linear in the evidence at every
    scale, and the mean reversion is precision-scaled,
    :math:`(\pi_p / (N\pi))(\mu_p - w)`, so a weight the data have pinned is
    also pulled back more gently.

    Both halves are mean-field: each weight's gradient is divided by its own
    precision, and the joint structure across the weights feeding one child is
    not retained.

    Parameters
    ----------
    learning_kwargs :
        The settings, as ``DeepNetwork.fit(learning_kwargs=...)`` takes them.
        Recognised keys, with the defaults of :data:`SYNAPTIC_UNCERTAINTY_DEFAULTS`:

        ``window``
            The memory window :math:`N`, in update steps. Importance decays
            toward the prior at rate :math:`1/N`, so curvature older than
            roughly :math:`N` steps no longer protects a weight. Required,
            and at least 1.
        ``prior_variance``
            Variance of the per-weight prior belief, :math:`1/\pi_p`. Also the
            effective learning rate before any importance has accumulated, so a
            value that trains the model well under plain gradient descent is
            the natural setting.
        ``prior_mean``
            Mean the weights revert toward.
        ``learning_rate``
            Multiplier :math:`\alpha` on the gradient part of the mean update.
            It exists because the rule otherwise has no step size of its own:
            the effective rate is :math:`\alpha/\pi`, and once the accumulated
            curvature dominates the prior the rate is pinned at
            :math:`\alpha/(N\bar{H})`. It sets the overall scale of every step
            but not the *ratio* between the rate before and after importance
            accumulates, which is :math:`1 + N\bar{H}/\pi_p` and is the depth
            of protection the rule applies. It does not scale the reversion,
            which stays at its :math:`1/N` rate, so ``window`` keeps meaning
            one thing only: how long importance is remembered.
        ``increment_scale``
            Multiplier on the importance increment. It deepens protection
            without touching anything else: the precision reaches
            :math:`\pi_p + c\,N\bar{H}`, so the most protected weights slow by
            :math:`c` times more, while the step size before any importance
            accumulates, the reversion rate and the window are unchanged. A
            uniform scale leaves rank orderings unchanged.

        The gradient the rule descends is not among them: it is fixed to
        ``"synaptic_uncertainty"``.

    Returns
    -------
    SynapticUncertaintySettings
        The validated settings, with the prior precision precomputed.

    Raises
    ------
    ValueError
        If a key is unrecognised, ``window`` is missing or below 1, or a
        positive quantity is not positive.
    """
    settings = dict(SYNAPTIC_UNCERTAINTY_DEFAULTS)
    given = dict(learning_kwargs or {})
    unknown = sorted(set(given) - set(SYNAPTIC_UNCERTAINTY_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown learning_kwargs for learning_kind="
            f"'synaptic_uncertainty': {unknown}. "
            f"Expected a subset of {sorted(SYNAPTIC_UNCERTAINTY_DEFAULTS)}."
        )
    settings.update(given)

    if settings["window"] is None:
        raise ValueError(
            "learning_kind='synaptic_uncertainty' requires "
            "learning_kwargs={'window': N, ...}: "
            "the memory window has no default, since it sets how long "
            "importance is remembered."
        )
    window = float(settings["window"])
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}.")
    prior_variance = float(settings["prior_variance"])
    if prior_variance <= 0:
        raise ValueError(f"prior_variance must be positive, got {prior_variance}.")
    learning_rate = float(settings["learning_rate"])
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    increment_scale = float(settings["increment_scale"])
    if increment_scale <= 0:
        raise ValueError(f"increment_scale must be positive, got {increment_scale}.")
    return SynapticUncertaintySettings(
        window=window,
        prior_variance=prior_variance,
        prior_mean=float(settings["prior_mean"]),
        learning_rate=learning_rate,
        increment_scale=increment_scale,
        prior_precision=1.0 / prior_variance,
    )


def vectorised_synaptic_uncertainty_update(
    weights: jnp.ndarray,
    precision_delta: jnp.ndarray,
    gradient: jnp.ndarray,
    importance,
    settings: SynapticUncertaintySettings,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""One weight matrix's belief update: new mean, new accumulated precision.

    The rule is stated in :func:`resolve_synaptic_uncertainty_settings`. A
    ``LayerStack`` carries a leading slice axis on every operand, which the ellipsis
    broadcasting here handles unchanged.

    Parameters
    ----------
    weights :
        The belief means, i.e. the element's ``weights_mean``.
    precision_delta :
        The accumulated precision above the prior, same shape.
    gradient :
        The descent gradient, same shape.
    importance :
        Either the factor pair ``(p, q)`` of one sample, from
        :func:`~pyhgf.updates.vectorised.learning.factors.learning_weights_vectorised`,
        whose outer product is the increment; or a full increment matrix of the
        weights' shape, the weighted batch mean of the per-sample outer products.
    settings :
        The resolved settings.

    Returns
    -------
    tuple of jnp.ndarray
        The updated weights and the updated accumulated precision.
    """
    prior_precision = settings.prior_precision
    pi = jnp.maximum(precision_delta + prior_precision, _EPS)

    # The mean update divides by the pre-update precision
    # and the reversion is precision-scaled.
    reversion = (prior_precision / (settings.window * pi)) * (
        settings.prior_mean - weights
    )
    update = -settings.learning_rate * gradient / pi + reversion

    if isinstance(importance, tuple):
        # Rank-one importance increment H[a, i] = p[a] * q[i].
        p, q = importance
        increment = p[..., :, None] * q[..., None, :]
    else:
        increment = importance
    increment = settings.increment_scale * increment
    new_delta = precision_delta + increment - precision_delta / settings.window

    return weights + update, new_delta


def _apply_synaptic_uncertainty_updates(
    network: VectorisedNetwork,
    grads: tuple,
    importance: tuple,
    settings: SynapticUncertaintySettings,
) -> VectorisedNetwork:
    r"""Advance every weight belief one step, mean and precision together.

    The rule needs no optimiser: the step size is the belief's own variance (see
    :func:`resolve_synaptic_uncertainty_settings`), so the update is applied here and
    both the mean (``weights_mean``) and the accumulated precision
    (``weights_precision_delta``) are written back to the element that carries them.
    In a one-to-one element, only the diagonal beliefs move.

    Parameters
    ----------
    network :
        The network whose beliefs are advanced.
    grads :
        One descent gradient per element, ``None`` for the bottom element.
    importance :
        One importance entry per element, aligned with ``grads``.
    settings :
        The resolved settings of the rule.

    Returns
    -------
    VectorisedNetwork
        The network with updated weights and precisions.

    Raises
    ------
    ValueError
        If an element holding weights carries no belief, which means the
        install step was skipped.
    """
    elements = list(network.layers)
    for i, elem in enumerate(elements):
        if elem.weights_mean is None or grads[i] is None:
            continue
        if elem.weights_precision_delta is None:
            raise ValueError(
                f"layers[{i}] holds weights but no weight belief. Install one "
                "with add_layer(weight_belief=True), or let "
                "learning_kind='synaptic_uncertainty' install it."
            )
        mask = _edge_mask(elem)
        new_weights, new_delta = vectorised_synaptic_uncertainty_update(
            elem.weights_mean,
            elem.weights_precision_delta,
            grads[i],
            importance[i],
            settings,
        )
        if mask is not None:
            # The reversion towards the prior mean moves every weight whatever its
            # gradient, so the masked weights and precisions keep their values.
            new_weights = jnp.where(mask, new_weights, elem.weights_mean)
            new_delta = jnp.where(mask, new_delta, elem.weights_precision_delta)
        elements[i] = dataclasses.replace(
            elem,
            weights_mean=new_weights,
            weights_precision_delta=new_delta,
        )
    return dataclasses.replace(network, layers=tuple(elements))
