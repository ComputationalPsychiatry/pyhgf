# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Rank-one factors of the ``"standard"`` and ``"precision_weighted"`` learning kinds.

:func:`learning_weights_vectorised` returns, for one weight matrix, the vectors whose
outer products are the descent gradient and the importance increment. The
``"synaptic_uncertainty"`` kind reads both: its gradient is one more kind of these
factors, and its importance factor reads the evidence precision that
:func:`clamped_layer_evidence` seeds at the clamped layer and :func:`evidence_pullback`
carries up the network. The private helpers compute the factors of every element after
the update sweep, and assemble or contract them over a batch.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing.vectorised import Layer, LayerStack, LayerState, VectorisedNetwork
from pyhgf.updates.vectorised.elements import _EPS, _child_view

# The learning kinds whose factors :func:`learning_weights_vectorised` returns. All
# three share the same base term and factorise into a child-side and a parent-side
# vector (a rank-one product), which is why the gradient is assembled as a single outer
# product.
SEPARABLE_KINDS: tuple = ("standard", "precision_weighted", "synaptic_uncertainty")


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

    One weight update has two halves and
    this returns the pieces of both: the descent gradient that moves the
    weight's mean, and the importance increment that raises its precision. They
    are the first and second derivative of one layer-local energy, computed in
    one pass.

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
      :math:`\Omega_a = \gamma_a / \hat\pi_a` recovered from the effective
      precision the prediction sweep writes as
      :math:`\gamma_a = \Omega_a \hat\pi_a`. The evidence :math:`\xi_a`
      itself is supplied by the caller through *child_evidence*, carried up from
      the clamped layer by :func:`evidence_pullback`, and the weight-belief rule
      always supplies it.

      Without it, :math:`\xi_a` falls back to :math:`\pi_a - \hat\pi_a` from
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
        letting it be recovered here as :math:`\pi_a - \hat\pi_a`. When given,
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
    # prediction, which must not turn the increment into a subtraction, nor the
    # softening below into an amplification.
    if child_evidence is None:
        evidence = child_state.precision - child_state.expected_precision
    else:
        evidence = child_evidence
    floored = jnp.maximum(evidence, 0.0)

    # The child's process noise. The prediction sweep caches the effective
    # precision gamma = Omega * pi_hat, so Omega = gamma / pi_hat. It is
    # zero at a clamped layer, which undergoes no random walk.
    process_noise = child_state.effective_precision / jnp.maximum(
        child_state.expected_precision, _EPS
    )

    # A message crossing that noise has the two variances convolved, so an
    # evidence precision xi arrives as xi_tilde = (1/xi + Omega)^-1
    # = xi / (1 + Omega xi). The denominator is kept on its own because both
    # halves need it: the increment is xi_tilde h^2, while the gradient factor
    # divides the child's displacement by the same quantity.
    softening = 1.0 / (1.0 + process_noise * floored)

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
    child_process_noise: jnp.ndarray,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
) -> jnp.ndarray:
    r"""Carry the evidence precision up one layer, by the squared-coupling recursion.

    The message leaving a child first crosses that child's own process noise, which
    convolves the two variances and softens the evidence to :math:`\tilde\xi_a`.
    What is left is then pulled back through the linear map the child depends on the
    parent by: a Gaussian likelihood pulled back through a coefficient :math:`c` has
    its precision multiplied by :math:`c^2`, and here
    :math:`c = W_{ai}\,g'(\hat\mu_i)`. Children are conditionally independent given
    the parent, so their pulled-back precisions add:

    .. math::

        \xi_i = g'(\hat\mu_i)^2 \sum_a W_{ai}^2\, \tilde\xi_a,
        \qquad
        \tilde\xi_a = \left(\frac{1}{\xi_a} + \Omega_a\right)^{-1}

    The softening applies once per level crossed, so it is part of the recursion and
    not only of the increment the evidence eventually feeds. It is exactly one at a
    clamped layer, which has no process noise, so it changes nothing at the seed.

    This is the upward mirror of the downward variance bleed-through, and
    computationally it is curvature backpropagation: per sample,
    :math:`\tilde\xi_a h_i^2` is the Gauss-Newton diagonal of the global loss
    with respect to :math:`W_{ai}`, under the same diagonal treatment (cross
    terms between different parents of one child are dropped).

    Computed from the evidence carried in the walk rather than recovered as
    :math:`\pi_a - \hat\pi_a` from the filter's cache. The two agree in exact
    arithmetic, but the difference of two large nearly-equal precisions loses
    every significant digit once a layer's precision has accumulated, whereas a
    carried quantity does not.

    Parameters
    ----------
    parent_state :
        State of the layer the evidence is being carried up to.
    child_evidence :
        Evidence precision at each node of the layer below, shape
        ``(n_children,)``, before the child's own process noise is charged.
    child_process_noise :
        That layer's process noise :math:`\Omega_a`, same shape, recovered by its
        caller from the cached effective precision as
        :math:`\Omega_a = \gamma_a / \hat\pi_a`.
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

    # Charge the child's process noise, xi_tilde = xi / (1 + Omega xi). Once per
    # level, so it compounds with depth, and exactly the identity at a clamped
    # layer, whose process noise is zero. The floor keeps a negative evidence,
    # which precision clipping elsewhere in the sweep can produce, from
    # amplifying the message instead of attenuating it.
    floored = jnp.maximum(child_evidence, 0.0)
    softened = floored / (1.0 + child_process_noise * floored)
    coupling_prime = vmap(jgrad(coupling_fn))(parent_state.expected_mean)
    pulled = jnp.matmul(weights.T**2, softened) * coupling_prime**2
    return jnp.where(jnp.isfinite(pulled), jnp.maximum(pulled, 0.0), 0.0)


def _layer_weight_op(
    parent: Layer, child_elem, learning_kind: str, child_evidence=None
):
    """Learning factors for a ``Layer`` parent and its child.

    *child_evidence* is passed only by the evidence walk; ``None`` leaves
    ``learning_weights_vectorised`` to recover the evidence from the cache.
    """
    child_state, child_kind, _ = _child_view(child_elem)
    return learning_weights_vectorised(
        parent_state=parent.state,
        child_state=child_state,
        coupling_fn=parent.coupling_fn,
        kind=learning_kind,
        parent_has_constant=parent.add_constant_input,
        child_kind=child_kind,
        child_evidence=child_evidence,
    )


def _stack_weight_op(stack: LayerStack, child_elem, learning_kind: str, evidence=None):
    """Learning factors for every slice of a ``LayerStack``, and the evidence above it.

    The child of slice 0 is the layer below the stack (``child_elem``); the child of
    slice k>0 is slice k-1 within the stack. Slice 0 is the boundary and is computed
    on its own: its child keeps its actual kind, since it may be the clamped binary
    or categorical observation layer, whose gradient and importance factors differ
    from a continuous child's. The interior slices ``1 .. N-1`` all have a stack
    slice as their child and are ``vmap``-ed together.

    The evidence walk cannot be vectorised the same way, because slice k's evidence is
    a function of slice k-1's: it is a recursion, not a map. It is carried by a
    ``scan`` that emits, per slice, the evidence *arriving* at that slice from below,
    which is exactly what the slice's own increment needs; the scan's final carry is
    the evidence leaving the top of the stack, for the element above it.

    *evidence* is ``None`` for the learning kinds that never read the importance
    factor; the walk is then skipped and the evidence above the stack is ``None`` too.

    Returns
    -------
    tuple
        The stacked factor triple, and the evidence at the top of the stack.
    """
    child_state, child_kind, _ = _child_view(child_elem)

    def carry(below, slice_data):
        evidence_below, noise_below = below
        slice_state, slice_weights = slice_data
        above = evidence_pullback(
            parent_state=slice_state,
            child_evidence=evidence_below,
            child_process_noise=noise_below,
            weights=slice_weights,
            coupling_fn=stack.coupling_fn,
            parent_has_constant=stack.add_constant_input,
        )
        noise_here = slice_state.effective_precision / jnp.maximum(
            slice_state.expected_precision, _EPS
        )
        return (above, noise_here), evidence_below

    if evidence is None:
        evidence_out = None
        per_slice_evidence = jnp.zeros(stack.state.mean.shape)
    else:
        # The carry holds the evidence arriving at a slice and the process noise of
        # the layer it left, since the softening is charged at the level it crosses.
        # The sweep caches the effective precision gamma = Omega * pi_hat, so the
        # process noise is gamma / pi_hat. It is carried as an array rather than as
        # the layer's state, whose pytree structure differs between the external
        # child and a stack slice when one of them has no volatility parent.
        (evidence_out, _), per_slice_evidence = jax.lax.scan(
            carry,
            (
                evidence,
                child_state.effective_precision
                / jnp.maximum(child_state.expected_precision, _EPS),
            ),
            (stack.state, stack.weights_mean),
        )

    walking = evidence is not None

    def slice_factors(parent_state, child_state_for_slice, child_evidence, kind):
        return learning_weights_vectorised(
            parent_state=parent_state,
            child_state=child_state_for_slice,
            coupling_fn=stack.coupling_fn,
            kind=learning_kind,
            parent_has_constant=stack.add_constant_input,
            child_kind=kind,
            child_evidence=child_evidence if walking else None,
        )

    # Boundary: slice 0 from the external child, with the child's own kind.
    first = slice_factors(
        jax.tree_util.tree_map(lambda x: x[0], stack.state),
        child_state,
        per_slice_evidence[0],
        child_kind,
    )
    # Interior slices 1 .. N-1, whose children are the stack's own slices. A
    # single-slice stack maps over a zero-length axis and the concatenation
    # below just wraps the boundary factors.
    rest = jax.vmap(lambda p, c, e: slice_factors(p, c, e, "continuous"))(
        jax.tree_util.tree_map(lambda x: x[1:], stack.state),
        jax.tree_util.tree_map(lambda x: x[:-1], stack.state),
        per_slice_evidence[1:],
    )
    factors = jax.tree_util.tree_map(
        lambda f, r: jnp.concatenate([f[None, ...], r], axis=0), first, rest
    )
    return (factors, evidence_out) if walking else factors


def _weight_op(parent_elem, child_elem, learning_kind: str):
    """Dispatch the learning factors on ``Layer`` vs ``LayerStack``."""
    if isinstance(parent_elem, LayerStack):
        return _stack_weight_op(parent_elem, child_elem, learning_kind)
    return _layer_weight_op(parent_elem, child_elem, learning_kind)


def _weight_quantities(network: VectorisedNetwork, learning_kind: str) -> tuple:
    r"""Per-element weight-learning factors, without applying them.

    Must run *after* :func:`~pyhgf.utils.vectorised_belief_propagation._update_sweep`,
    so the per-layer states already carry their prediction errors / posteriors.
    Returns one entry per element, matched 1:1 to
    ``network.layers`` (``None`` for the bottom element, which has no incoming
    weights); each entry is the factor tuple of
    :func:`learning_weights_vectorised`.
    Assemble them with :func:`_gradient_matrix` and :func:`_importance_pair`.

    Under ``learning_kind="synaptic_uncertainty"`` the importance increment's
    child-side factor is the evidence precision, and this walks it up the stack in its
    own quantity: seeded at the clamped layer by
    :func:`clamped_layer_evidence` and raised one
    element at a time by
    :func:`evidence_pullback`. This loop runs bottom
    to top already, which is the order the recursion needs, so the walk costs one
    matrix product per element and no extra sweep.

    The evidence is *carried* rather than recovered from the filter's cache as
    :math:`\pi_a - \hat\pi_a`. The two agree in exact arithmetic only where the
    filter's own chain is seeded with the clamped layer's likelihood curvature, which
    it is not: a clamped categorical layer enters that chain at unit precision, because
    that convention is what makes the message it routes exact cross-entropy
    backpropagation. One variable cannot serve both roles, so the curvature gets its
    own. Carrying it also removes a subtraction of two large nearly-equal precisions,
    which loses every significant digit once a layer's precision has accumulated.

    The other learning kinds never read the importance factor, so they skip the walk.
    """
    elements = network.layers
    if learning_kind != "synaptic_uncertainty":
        return (None,) + tuple(
            _weight_op(elements[i], elements[i - 1], learning_kind)
            for i in range(1, len(elements))
        )

    child_state, child_kind, _ = _child_view(elements[0])
    evidence = clamped_layer_evidence(child_state, child_kind)

    factors: list = [None]
    for i in range(1, len(elements)):
        parent = elements[i]
        if isinstance(parent, LayerStack):
            stack_factors, evidence = _stack_weight_op(
                parent, elements[i - 1], learning_kind, evidence
            )
            factors.append(stack_factors)
            continue
        factors.append(
            _layer_weight_op(parent, elements[i - 1], learning_kind, evidence)
        )
        if i + 1 < len(elements):
            # The evidence is softened by the process noise of the layer it leaves,
            # which the sweep caches as the effective precision gamma = Omega *
            # pi_hat, so the noise itself is gamma / pi_hat.
            below, _, _ = _child_view(elements[i - 1])
            evidence = evidence_pullback(
                parent_state=parent.state,
                child_evidence=evidence,
                child_process_noise=below.effective_precision
                / jnp.maximum(below.expected_precision, _EPS),
                weights=parent.weights_mean,
                coupling_fn=parent.coupling_fn,
                parent_has_constant=parent.add_constant_input,
            )
    return tuple(factors)


def _gradient_matrix(factors) -> Optional[jnp.ndarray]:
    """Assemble the descent gradient from one element's factors."""
    if factors is None:
        return None
    u, h = factors[0], factors[1]
    return u[..., :, None] * h[..., None, :]


def _importance_pair(factors) -> Optional[tuple]:
    """Build the importance factor pair ``(p, h**2)`` from one element's factors.

    The parent side is squared here rather than at the source, since both quantities
    read the same activation. Squaring can overflow a finite activation, so the non-
    finite guard is applied after it, matching the child side.
    """
    if factors is None:
        return None
    squared = factors[1] ** 2
    return factors[2], jnp.where(jnp.isfinite(squared), squared, 0.0)


def _contract_factors(factors, weights: jnp.ndarray) -> Optional[jnp.ndarray]:
    r"""Weighted sum over the batch of per-sample outer products :math:`u \otimes v`.

    ``factors`` is ``None`` for the bottom element, or a tuple whose first two entries
    ``(u, v)`` carry a leading batch axis: ``(batch, n_children)`` and ``(batch,
    n_parents)`` for a ``Layer``, ``(batch, n_slices, ...)`` for a ``LayerStack``.
    ``weights`` has shape ``(batch,)``; with the normalised weights of
    :func:`~pyhgf.utils.vectorised_belief_propagation._batch_step` the result is the
    weighted batch mean, with the weights' shape.

    Each sample's two factors are multiplied before the sum, so their covariance over
    the batch is kept. For the importance pair of :func:`_importance_pair` this gives
    the batch mean of the per-sample curvatures, since discrete likelihood curvature and
    pulled-back evidence vary with the sample.
    """
    if factors is None:
        return None
    u, v = factors[:2]
    return jnp.einsum("b,b...i,b...j->...ij", weights, u, v)
