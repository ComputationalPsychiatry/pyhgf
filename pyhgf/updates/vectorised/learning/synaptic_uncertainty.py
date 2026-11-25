# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Weight-belief rule of the ``"synaptic_uncertainty"`` learning kind.

Each weight carries a Gaussian belief: its mean is the weight, and its precision
accumulates the curvature the data impose on it. The mean steps along the gradient
divided by that precision. The rule and its settings are stated in
:func:`resolve_synaptic_uncertainty_settings`. The input Gram matrices a layer can
record beside its weight belief are advanced here as well.
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
    "gram_evidence": False,
}

#: The recorded fields that hold an input Gram matrix. A learning step forms the
#: batch's Gram matrix only for an element carrying at least one of them.
_GRAM_FIELDS = ("weights_input_gram", "weights_step_gram", "weights_batch_gram")


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
    gram_evidence: bool = False


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
    not retained. A layer can record that structure without using it (its input
    Gram matrices, see :class:`~pyhgf.typing.vectorised.Layer`), for a step a
    caller takes from the recorded fields.

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
        ``gram_evidence``
            Whether each sample's outer product :math:`h_t h_t^\top` enters the
            recorded Gram matrices (``weights_input_gram``, ``weights_step_gram``
            and ``weights_batch_gram``) weighted by the evidence its child layer
            received, :math:`r_t = \frac{1}{n} \sum_a p_{a,t}`: the mean over the
            child's :math:`n` units of the child-side importance factor :math:`p`,
            the evidence precision of
            :func:`~pyhgf.updates.vectorised.learning.factors.learning_weights_vectorised`.
            Within each batch the weights are rescaled so that the weighted Gram matrix
            keeps the trace of the unweighted one, so the evidence decides which
            samples' inputs shape the matrix and leaves its scale alone. Where the
            evidence is one constant across samples (a continuous clamped child)
            the matrix is unchanged. At a categorical child :math:`r_t = (1 -
            \lVert \mathbf{p}_t \rVert^2) / K`, with :math:`\mathbf{p}_t` the
            predicted probabilities over :math:`K` classes, so a sample the model
            already predicts confidently counts less. ``False`` (the default)
            weights every sample equally. A single-sample step is unaffected
            either way, since the rescaling cancels the weight of a lone sample.

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
        gram_evidence=bool(settings["gram_evidence"]),
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


def create_input_gram(weights: jnp.ndarray) -> jnp.ndarray:
    r"""Initialise the input Gram matrix of one weight matrix at zero.

    The Gram matrix spans the parent side of ``weights``, so its shape is the last
    axis of ``weights`` on both sides, ``(n_parents[+1], n_parents[+1])``, and a bias
    column is included. Zero is the state before any evidence, matching a freshly
    installed ``weights_precision_delta``.

    Parameters
    ----------
    weights :
        The weight matrix the Gram matrix accompanies, ``(n_children,
        n_parents[+1])``.

    Returns
    -------
    jnp.ndarray
        Zeros of shape ``(n_parents[+1], n_parents[+1])``, stored as a layer's
        ``weights_input_gram``.
    """
    n_inputs = weights.shape[-1]
    return jnp.zeros((n_inputs, n_inputs), dtype=weights.dtype)


def vectorised_input_gram_update(
    gram: jnp.ndarray,
    increment: jnp.ndarray,
    settings: SynapticUncertaintySettings,
) -> jnp.ndarray:
    r"""Advance the input Gram matrix of one weight matrix by one learning step.

    .. math::

        G \leftarrow G + c\,\overline{h h^\top} - \frac{G}{N}

    where :math:`\overline{h h^\top}` is the weighted batch mean of the outer
    product of the coupled parent activations :math:`h`, :math:`c` is the
    increment scale and :math:`N` the window. This is the update
    :func:`vectorised_synaptic_uncertainty_update` applies to the accumulated
    precision, :math:`\Delta \leftarrow \Delta + c\,\overline{p \otimes h^2} -
    \Delta / N`, with the child-side factor :math:`p` left out and the parent
    side kept as a full matrix rather than its diagonal.

    Two consequences follow. Where the child-side factor is one constant
    :math:`\xi` across units and samples (a continuous clamped layer), every row
    :math:`a` of the accumulated precision is :math:`\Delta_{a,\cdot} = \xi \,
    \operatorname{diag}(G)`, so :math:`G` holds the off-diagonal terms that the
    per-weight precision discards. Where it varies, :math:`\bar{p}_a\, G` is the
    Kronecker-factored approximation of the precision of row :math:`a`, which replaces
    the mean of the products :math:`p_a\, h h^\top` by the product of the means (Eq. 1
    of [1]_).

    Parameters
    ----------
    gram :
        The accumulated Gram matrix, ``(n_parents[+1], n_parents[+1])``.
    increment :
        The weighted batch mean of :math:`h h^\top`, same shape, before the
        increment scale is applied.
    settings :
        The resolved settings of the weight-belief rule.

    Returns
    -------
    jnp.ndarray
        The advanced Gram matrix.

    References
    ----------
    .. [1] Martens, J., & Grosse, R. (2015). Optimizing neural networks with
       Kronecker-factored approximate curvature. Proceedings of the 32nd International
       Conference on Machine Learning, 2408 to 2417. Equation and section numbers
       refer to the extended version, arXiv:1503.05671.
    """
    return gram + settings.increment_scale * increment - gram / settings.window


def create_step_gram(weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Initialise the running input Gram matrix of one weight matrix at zero.

    A layer carrying the running Gram matrix also needs a ``weights_step_rate``, the
    rate at which :func:`vectorised_step_gram_update` mixes each batch in.

    Parameters
    ----------
    weights :
        The weight matrix the Gram matrix accompanies, ``(n_children,
        n_parents[+1])``.

    Returns
    -------
    tuple of jnp.ndarray
        Zeros of shape ``(n_parents[+1], n_parents[+1])``, stored as
        ``weights_step_gram``, and a zero scalar, stored as
        ``weights_step_mass``.
    """
    return create_input_gram(weights), jnp.zeros((), dtype=weights.dtype)


def vectorised_step_gram_update(
    gram: jnp.ndarray,
    mass: jnp.ndarray,
    batch_gram: jnp.ndarray,
    rate: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Mix one batch into the running input Gram matrix of one weight matrix.

    .. math::

        G \leftarrow (1 - \alpha) G + \alpha\, \overline{h h^\top},
        \qquad
        m \leftarrow (1 - \alpha) m + \alpha,

    with :math:`\alpha` the layer's ``weights_step_rate``.
    Starting from :math:`G = 0` and :math:`m = 0`, :math:`G / m` is the
    weighted average of the batches seen, each batch weighted by the product
    of the :math:`1 - \alpha` factors of the steps after it, so its first value
    is the first batch's Gram matrix whatever :math:`\alpha` is. Without
    ``gram_evidence``, :math:`G / m` estimates the Gram matrix that preconditions the
    natural-gradient step of predictive coding (Eqs. 5 and 6 of [1]_), which Eq. 7 of
    [1]_ tracks as an exponential moving average.

    Parameters
    ----------
    gram, mass :
        ``weights_step_gram`` and ``weights_step_mass``.
    batch_gram :
        The weighted batch mean of :math:`h h^\top`, before any increment scale.
    rate :
        The mixing rate :math:`\alpha` of this step, a scalar in :math:`(0, 1]`.

    Returns
    -------
    tuple of jnp.ndarray
        The advanced Gram matrix and normaliser.

    References
    ----------
    .. [1] Beumer, L., de Haan, S., Pokel, N., Vilimelis Aceituno, P., & Grewe, B. F.
       (2026). Natural gradient learning in predictive coding through lateral
       circuits. arXiv:2611.16666
    """
    return (1.0 - rate) * gram + rate * batch_gram, (1.0 - rate) * mass + rate


def vectorised_step_rows_update(
    rows: jnp.ndarray,
    batch_rows: jnp.ndarray,
    rate: jnp.ndarray,
) -> jnp.ndarray:
    r"""Mix one batch's row evidence into the running row evidence.

    .. math::

        e_a \leftarrow (1 - \alpha)\, e_a + \alpha\, \sum_t \omega_t\, p_{a,t}
        \lVert h_t \rVert^2,

    with :math:`p_{a,t}` the evidence child unit :math:`a` received from sample
    :math:`t` (the child-side importance factor of
    :func:`~pyhgf.updates.vectorised.learning.factors.learning_weights_vectorised`),
    :math:`h_t` the coupled parent activations, :math:`\omega_t` the normalised sample
    weights, and :math:`\alpha` the rate :func:`vectorised_step_gram_update` uses on
    the same step. The batch
    term is the row sum of the batch's importance increment :math:`p \otimes h^2`,
    taken over the edges only: in a one-to-one element row :math:`a` has the single
    input :math:`h_a`, and its term is :math:`\sum_t \omega_t\, p_{a,t} h_{a,t}^2`.

    Divided by its mean over rows, :math:`e` is the row scale :math:`s_a` of the
    Kronecker-factored curvature :math:`s_a G` of row :math:`a` (the form of Eq. 1 of
    [1]_ with a diagonal output factor), whose input side :math:`G` is
    ``weights_step_gram`` (trace matching: :math:`s_a \operatorname{tr}
    G` equals the row's own curvature trace up to one factor common to all rows).
    Where the evidence is one constant across units, every :math:`s_a` is 1.

    Parameters
    ----------
    rows :
        ``weights_step_rows``, shape ``(n_children,)``.
    batch_rows :
        The batch's row evidence, same shape.
    rate :
        The mixing rate :math:`\alpha` of this step, a scalar in :math:`(0, 1]`.

    Returns
    -------
    jnp.ndarray
        The advanced row evidence.

    References
    ----------
    .. [1] Martens, J., & Grosse, R. (2015). Optimizing neural networks with
       Kronecker-factored approximate curvature. Proceedings of the 32nd International
       Conference on Machine Learning, 2408 to 2417. Equation and section numbers
       refer to the extended version, arXiv:1503.05671.
    """
    return (1.0 - rate) * rows + rate * batch_rows


def _apply_synaptic_uncertainty_updates(
    network: VectorisedNetwork,
    grads: tuple,
    importance: tuple,
    settings: SynapticUncertaintySettings,
    grams: Optional[tuple] = None,
) -> VectorisedNetwork:
    r"""Advance every weight belief one step, mean and precision together.

    The rule needs no optimiser: the step size is the belief's own variance (see
    :func:`resolve_synaptic_uncertainty_settings`), so the update is applied here and
    both the mean (``weights_mean``) and the accumulated precision
    (``weights_precision_delta``) are written back to the element that carries them.
    In a one-to-one element, only the diagonal beliefs move.

    The recorded fields an element carries (see
    :class:`~pyhgf.typing.vectorised.Layer`) are written here and never read by the
    update:

    - ``weights_gradient`` is overwritten with this step's gradient.
    - ``weights_input_gram`` is advanced by :func:`vectorised_input_gram_update`.
    - ``weights_step_gram`` and ``weights_step_mass`` are advanced by
      :func:`vectorised_step_gram_update`, at the element's ``weights_step_rate``, and
      a ``weights_step_rows`` is advanced at the same rate by
      :func:`vectorised_step_rows_update`, from the row sums of the importance
      increment.
    - ``weights_batch_gram`` is overwritten with this step's batch Gram matrix.

    In a one-to-one element every recorded quantity is restricted to the edges. The
    gradient is masked to the diagonal. Each Gram matrix keeps only its diagonal: row
    :math:`a` has the single input :math:`h_a`, so a product :math:`h_i h_j` with
    :math:`i \neq j` is the curvature of no weight, and a step solved against it would
    mix the gradients of weights that are not edges. The row evidence sums over the
    edges only, so there row :math:`a` sums to :math:`\sum_t \omega_t p_{a,t}
    h_{a,t}^2` rather than :math:`\sum_t \omega_t p_{a,t} \lVert h_t \rVert^2`.

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
    grams :
        One batch mean :math:`h h^\top` per element, aligned with ``grads``,
        ``None`` for an element that records no Gram matrix. ``None`` (default)
        when no element records one.

    Returns
    -------
    VectorisedNetwork
        The network with updated weights, precisions and recorded fields.

    Raises
    ------
    ValueError
        If an element holding weights carries no belief, which means the
        install step was skipped, or carries a ``weights_step_gram`` without a
        ``weights_step_rate``.
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
        fields = {}
        if getattr(elem, "weights_gradient", None) is not None:
            fields["weights_gradient"] = (
                grads[i] if mask is None else jnp.where(mask, grads[i], 0.0)
            )
        batch_gram = None if grams is None else grams[i]
        if batch_gram is not None:
            if mask is not None:
                batch_gram = batch_gram * jnp.eye(
                    batch_gram.shape[-1], dtype=batch_gram.dtype
                )
            if getattr(elem, "weights_input_gram", None) is not None:
                fields["weights_input_gram"] = vectorised_input_gram_update(
                    elem.weights_input_gram, batch_gram, settings
                )
            if getattr(elem, "weights_batch_gram", None) is not None:
                fields["weights_batch_gram"] = batch_gram
            if getattr(elem, "weights_step_gram", None) is not None:
                rate = getattr(elem, "weights_step_rate", None)
                if rate is None:
                    raise ValueError(
                        f"layers[{i}] carries a weights_step_gram but no "
                        "weights_step_rate. Install the rate at which the batches "
                        "are mixed in beside the running Gram matrix."
                    )
                gram, mass = vectorised_step_gram_update(
                    elem.weights_step_gram, elem.weights_step_mass, batch_gram, rate
                )
                fields["weights_step_gram"] = gram
                fields["weights_step_mass"] = mass
                if getattr(elem, "weights_step_rows", None) is not None:
                    # The importance is the factor pair (p, h**2) of one sample or the
                    # contracted increment matrix of a batch.
                    if isinstance(importance[i], tuple):
                        p, q = importance[i]
                        increment = p[..., :, None] * q[..., None, :]
                    else:
                        increment = importance[i]
                    if mask is not None:
                        increment = jnp.where(mask, increment, 0.0)
                    fields["weights_step_rows"] = vectorised_step_rows_update(
                        elem.weights_step_rows, increment.sum(axis=-1), rate
                    )
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
            **fields,
        )
    return dataclasses.replace(network, layers=tuple(elements))
