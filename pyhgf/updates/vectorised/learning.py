# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised weight learning for deep predictive coding networks."""

from typing import Callable, NamedTuple, Optional, Union

import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing.vectorised import LayerState

# The accepted weight-update kinds. All three share the same base term and
# factorise into a child-side and a parent-side vector (a rank-one product),
# which is why the gradient below is assembled as a single outer product.
SEPARABLE_KINDS: tuple = ("standard", "precision_weighted", "synaptic_uncertainty")

#: Floor applied to precisions before any division.
_EPS = 1e-30


def learning_weights_vectorised(
    parent_state: LayerState,
    child_state: LayerState,
    coupling_fn: Callable,
    kind: str = "precision_weighted",
    parent_has_constant: bool = False,
    child_kind: str = "continuous",
    child_evidence: Optional[jnp.ndarray] = None,
) -> tuple:
    r"""Per-layer weight-learning factors for the vectorised deep network.

    The single entry point of this module. One weight update has two halves and
    this returns the pieces of both: the descent gradient that moves the
    weight's mean, and the importance increment that raises its precision. They
    are the first and second derivative of one layer-local energy, computed in
    one pass rather than selected between.

    Both are rank-one and **share their parent-side factor**. The gradient is
    :math:`u \otimes h` and the increment is :math:`p \otimes h^2`, with the
    same coupled parent activation :math:`h_i = g(\mu_i)`, so three vectors
    carry both quantities. Returning factors rather than assembled matrices
    lets a batched caller average over samples and contract once
    (``einsum('bi,bj->ij') / batch``): the same arithmetic, without
    materialising one weight-matrix-sized array per sample.

    Both are strictly *local*: what one weight gets reads only the prediction
    error and precision at its child and the activation at its parent, never a
    reduction over the other parents or children in the layer.

    **The child-side gradient factor** starts from the value prediction error
    :math:`\delta = \mu_\text{child} - \hat{\mu}_\text{child}`, and *kind*
    decides what scales it:

    - ``"standard"`` takes :math:`\delta` alone, the raw gradient of an
      *unweighted* squared error. This coincides with the free-energy gradient
      only where the child precision is one (the unit-precision output or
      categorical convention).
    - ``"precision_weighted"`` scales by the child's *posterior* precision,
      :math:`\delta\,\pi_a`. This is the **backprop-parity** mode: a moved
      interior belief shifts by (routed error) / posterior precision, so
      weighting by that same posterior precision cancels the division and
      reproduces the backpropagated gradient node for node at *any* precision
      setting.
    - ``"synaptic_uncertainty"`` charges the process noise between the weight and
      its child on top, :math:`\delta\,\pi_a / (1 + \Omega_a \xi_a)`, so the
      gradient and the increment become the two derivatives of one energy.
      This is the kind the weight-belief rule descends, and the only one it
      can: dividing it by the weight's own precision is the Gaussian
      posterior step, so both halves of that update read the same evidence.
      It is *not* backprop parity.

    A binary child drops the precision factor either way: its ``precision``
    field holds the Bernoulli variance :math:`p(1-p)`, which cancels through
    the sigmoid in the gradient, so keeping it would count the same term
    twice. That cancellation belongs to the *first* derivative only, and the
    curvature below keeps it.

    **The child-side importance factor** is the exact Hessian diagonal of the
    layer-local variational energy: that energy is exactly quadratic in the
    weights for any coupling function, because the coupling acts on the
    activation while the map :math:`w_{ai} \mapsto w_{ai} h_i` stays linear, so
    one observation adds :math:`\tilde{\xi}_a h_i^2` with no approximation and
    no sampling. Where the evidence comes from depends on the child:

    - A **binary** or **categorical** child is where the data are clamped, so
      the evidence is the curvature of its own likelihood, per unit
      :math:`p_a(1 - p_a)` with :math:`p_a` the child's expected mean. For the
      categorical case the Hessian of the log-likelihood with respect to the
      logits is :math:`\operatorname{diag}(\mathbf{p}) -
      \mathbf{p}\mathbf{p}^{\top}`, whose diagonal is that expression. No
      label enters it, since averaging the outer product of the residual over
      labels drawn from the model returns the same matrix, so this is the
      model's own expected curvature rather than an estimate built from
      observed errors.
    - A **continuous** child passes on the evidence it received from below,
      softened by the process noise it had to cross,
      :math:`\tilde\xi_a = (1/\xi_a + \Omega_a)^{-1}`, with
      :math:`\Omega_a = \gamma_a / \tilde\pi_a` recovered from the effective
      precision the prediction sweep writes as
      :math:`\gamma_a = \Omega_a \tilde\pi_a`. The evidence :math:`\xi_a`
      itself is supplied by the caller through *child_evidence*, carried up from
      the clamped layer by :func:`evidence_pullback`, and the weight-belief rule
      always supplies it.

      Without it, :math:`\xi_a` falls back to :math:`\pi_a - \tilde\pi_a` from
      the cache, which is the same quantity in exact arithmetic *only* where the
      filter's own precision chain was seeded with the clamped layer's likelihood
      curvature. It is not: a clamped categorical layer enters that chain at unit
      precision, because that convention is what makes the message it routes exact
      cross-entropy backpropagation. Reading the difference also subtracts two
      large nearly-equal precisions, which loses every significant digit once a
      layer's precision has accumulated. The fallback is kept for direct callers
      reading the layer-local quantity; it is not what the rule descends.

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations), after the update
        sweep has written its posterior.
    coupling_fn :
        Coupling function applied to parent means.
    kind :
        The metric the gradient is expressed in, one of
        :data:`SEPARABLE_KINDS`. It also sets the importance convention:
        ``"standard"`` uses unit observation precision on both halves.
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0,
        precision = 1.0) appended to its activations after coupling.
    child_kind :
        The child layer's node kind, ``"binary"``, ``"categorical"`` or
        anything else for a continuous one.
    child_evidence :
        The child's evidence precision :math:`\xi_a`, supplied by a caller that
        carries it up the stack itself (:func:`evidence_pullback`) rather than
        letting it be recovered here as :math:`\pi_a - \tilde\pi_a`. When given,
        it replaces that difference in the importance factor and the fallback
        below never applies, since a carried evidence is non-negative by
        construction. The gradient factor is unaffected: it reads the child's
        posterior precision either way. ``None`` (default) recovers the evidence
        from the cache.

    Returns
    -------
    factors :
        The triple ``(u, h, p)``. The child-side gradient factor :math:`u` and
        the child-side importance factor
        :math:`p` have shape ``(n_children,)``; the shared parent-side factor
        :math:`h` has shape ``(n_parents[+1],)``. The gradient is
        ``u[:, None] * h[None, :]`` and the increment
        ``p[:, None] * (h ** 2)[None, :]``. Non-finite entries are zeroed, so
        optax never propagates a NaN or an inf through its moment
        accumulators.

    Raises
    ------
    ValueError
        If *kind* is unrecognised.
    """
    if kind not in SEPARABLE_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {SEPARABLE_KINDS}.")

    # The shared parent-side factor, h_i = g(mu_i). A constant input node
    # contributes a fixed activation of 1.0, so its weight counts as usage 1.
    h = coupling_fn(parent_state.mean)
    if parent_has_constant:
        h = jnp.concatenate([h, jnp.ones(1)])
    h = jnp.where(jnp.isfinite(h), h, 0.0)

    # The evidence the child received from below, and the factor by which the
    # process noise between the weight and the child attenuates it. Both halves
    # of the update read these, so they are formed once. The evidence is
    # floored at zero: clipping can drive a posterior precision below its
    # prediction, which must not turn the increment into a subtraction.
    if child_evidence is None:
        evidence = child_state.precision - child_state.expected_precision
    else:
        evidence = child_evidence
    floored = jnp.maximum(evidence, 0.0)
    tonic = child_state.effective_precision / jnp.maximum(
        child_state.expected_precision, _EPS
    )
    softening = 1.0 / (1.0 + tonic * floored)

    # Child-side gradient factor, in descent form: sign-flipped from the
    # natural "ascent" formulation so it composes with standard optax
    # (apply_updates performs weights + updates, and sgd(lr).update returns
    # -lr * grad).
    u = child_state.mean - child_state.expected_mean
    if kind != "standard" and child_kind != "binary":
        u = u * child_state.precision  # posterior precision
        if kind == "synaptic_uncertainty":
            u = u * softening
    u = -jnp.where(jnp.isfinite(u), u, 0.0)

    # Child-side importance factor. A caller advancing only the means ignores
    # it; the two array operations below are removed with it by dead-code
    # elimination, so gating them on a flag would buy nothing.
    if kind == "standard":
        p = jnp.ones_like(child_state.mean)
    elif child_evidence is not None:
        # A carried evidence is already the right quantity for any child: the
        # walk seeds a clamped discrete layer with its own p(1-p) and pulls that
        # up, so no per-kind branch and no fallback are needed here.
        p = floored * softening
    elif child_kind in ("binary", "categorical"):
        # Clamped discrete child: the curvature of its own likelihood.
        prob = child_state.expected_mean
        p = prob * (1.0 - prob)
    else:
        # A clamped continuous layer receives no message from below, so no
        # evidence arrives and its own precision stands in. That fallback is
        # the uniform clamp-precision convention and carries no per-unit
        # curvature; seeding such a layer with the true curvature of its
        # likelihood is separate work.
        p = jnp.where(evidence > 0.0, floored * softening, child_state.precision)
    p = jnp.where(jnp.isfinite(p), p, 0.0)

    return u, h, p


# ------------------------------------------------------ synaptic uncertainty

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
    :func:`learning_weights_vectorised` or as a full increment matrix. It then
    relaxes toward the prior in precision form,
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


def clamped_layer_evidence(child_state: LayerState, child_kind: str) -> jnp.ndarray:
    r"""Seed the evidence walk at the layer the data are clamped to.

    A **binary or categorical** layer contributes the exact, label-free curvature
    of its own likelihood, :math:`\xi_a = \hat{p}_a(1 - \hat{p}_a)`: the diagonal
    of :math:`\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top`, which
    is the Hessian of the categorical log-likelihood with respect to the logits.
    No observed label enters it, so it is the model's own expected curvature
    rather than one built from the errors that happened to occur.

    A **continuous** layer contributes its own precision, which for a Gaussian
    likelihood is that same curvature: the second derivative of
    :math:`-\log \mathcal{N}(y; \mu, 1/\pi)` with respect to :math:`\mu` is
    :math:`\pi`, whatever the residual. Nothing is approximated. What differs is
    that the value is a *constant*, and two things follow from that. It varies
    neither across units nor across samples, so unlike the discrete seed it says
    nothing about which outputs the data constrain hardest, and every per-weight
    difference the walk produces above such a layer comes from the activations
    and the weight pullback alone. And wherever the precision expresses "this
    layer is observed" rather than a modelled observation noise, its value is
    arbitrary; since the seed multiplies through the whole chain, it then fixes
    the scale of every accumulated weight precision above it, which trades
    against the prior precision.

    Parameters
    ----------
    child_state :
        State of the clamped layer, after the update sweep.
    child_kind :
        ``"binary"``, ``"categorical"``, or anything else for a continuous layer.

    Returns
    -------
    jnp.ndarray
        The evidence precision at each node of the clamped layer.
    """
    if child_kind in ("binary", "categorical"):
        prob = child_state.expected_mean
        evidence = prob * (1.0 - prob)
    else:
        evidence = child_state.precision
    return jnp.where(jnp.isfinite(evidence), evidence, 0.0)


def evidence_pullback(
    parent_state: LayerState,
    child_evidence: jnp.ndarray,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
) -> jnp.ndarray:
    r"""Carry the evidence precision up one layer, by the squared-coupling recursion.

    A Gaussian likelihood pulled back through a linear map of coefficient
    :math:`c` has its precision multiplied by :math:`c^2`; children are
    conditionally independent given the parent, so their pulled-back precisions
    add:

    .. math::

        \xi_i = g'(\hat\mu_i)^2 \sum_a W_{ai}^2\, \tilde\xi_a

    This is the upward mirror of the downward variance bleed-through, and
    computationally it is curvature backpropagation: per sample,
    :math:`\tilde\xi_a h_i^2` is the Gauss-Newton diagonal of the global loss
    with respect to :math:`W_{ai}`, under the same diagonal treatment (cross
    terms between different parents of one child are dropped).

    Computed from the evidence carried in the walk rather than recovered as
    :math:`\pi_a - \tilde\pi_a` from the filter's cache. The two agree in exact
    arithmetic, but the difference of two large nearly-equal precisions loses
    every significant digit once a layer's precision has accumulated, whereas a
    carried quantity does not.

    Parameters
    ----------
    parent_state :
        State of the layer the evidence is being carried up to.
    child_evidence :
        Evidence precision at each node of the layer below, shape
        ``(n_children,)``.
    weights :
        The matrix connecting them, shape ``(n_children, n_parents[+1])``. A bias
        column is dropped: a constant input node is not a parent whose belief the
        evidence can be about.
    coupling_fn :
        Coupling applied to parent means, differentiated at the parent's expected
        mean.
    parent_has_constant :
        Whether *weights* carries that trailing bias column.

    Returns
    -------
    jnp.ndarray
        The evidence precision at each node of the parent layer, shape
        ``(n_parents,)``.
    """
    if parent_has_constant:
        weights = weights[..., :-1]
    coupling_prime = vmap(jgrad(coupling_fn))(parent_state.expected_mean)
    pulled = jnp.matmul(weights.T**2, child_evidence) * coupling_prime**2
    return jnp.where(jnp.isfinite(pulled), jnp.maximum(pulled, 0.0), 0.0)


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
        Either the factor pair ``(p, q)`` of
        :func:`learning_weights_vectorised`, batch-averaged, whose outer product
        is the increment; or a full increment matrix of the weights' shape,
        which is the exact batch contraction the evidence pass delivers.
    settings :
        The resolved settings.

    Returns
    -------
    tuple of jnp.ndarray
        The updated weights and the updated accumulated precision.
    """
    prior_precision = settings.prior_precision
    pi = jnp.maximum(precision_delta + prior_precision, _EPS)

    # The mean update divides by the pre-update precision and the reversion is
    # precision-scaled, both as in the published rule.
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
