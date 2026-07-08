# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized weight learning for deep predictive coding networks."""

from typing import Callable

import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerState

# Default Tikhonov damping for the categorical (softmax) natural gradient. The
# multinomial Fisher diag(p) - p pᵀ is singular and its inverse amplifies the
# error by 1/p, which diverges for a class the model deems near-impossible.
# Adding lambda to the diagonal caps that amplification at ~1/lambda. Because p
# is a probability, this bound is scale-free: lambda = 1e-2 treats classes below
# ~1% probability as ~1% for curvature purposes.
CATEGORICAL_NATURAL_DAMPING: float = 1e-2


def _softmax_fisher_inverse(
    p: jnp.ndarray, pe: jnp.ndarray, damping: float
) -> jnp.ndarray:
    r"""Apply the damped inverse multinomial Fisher to a residual.

    Solves :math:`(\mathrm{diag}(p) + \lambda I - p p^\top)\,x = \delta` for
    ``x``, the softmax (categorical) natural gradient's output-side
    preconditioning of the value prediction error :math:`\delta` (``pe``). The
    rank-one term :math:`p p^\top` is inverted in closed form by Sherman-Morrison
    on the damped diagonal :math:`D = \mathrm{diag}(p) + \lambda I`:

    .. math::

        x = D^{-1} \delta + \frac{D^{-1} p \; (p^\top D^{-1} \delta)}{1 - p^\top D^{-1} p}.

    With ``damping`` :math:`> 0` the denominator lies in ``(0, 1)`` (because
    :math:`p^\top D^{-1} p = \sum_k p_k^2 / (p_k + \lambda) < \sum_k p_k = 1`),
    so the solve is always well-conditioned.

    Parameters
    ----------
    p :
        Predicted class probabilities (softmax output), shape ``(n_classes,)``.
    pe :
        Prediction error, the zero-sum residual ``one_hot - p``.
    damping :
        Tikhonov damping added to the Fisher diagonal.

    Returns
    -------
    x :
        The preconditioned residual, shape ``(n_classes,)``.
    """
    d = 1.0 / (p + damping)  # diagonal of D^{-1}
    dinv_pe = d * pe
    dinv_p = d * p
    denom = 1.0 - jnp.sum(p * dinv_p)  # 1 - pᵀ D^{-1} p
    coef = jnp.sum(p * dinv_pe) / denom  # (pᵀ D^{-1} pe) / denom
    return dinv_pe + dinv_p * coef


def vectorized_weight_gradient(
    parent_state: LayerState,
    child_state: LayerState,
    coupling_fn: Callable,
    kind: str = "precision_weighted",
    parent_has_constant: bool = False,
    child_is_binary: bool = False,
    child_is_categorical: bool = False,
    categorical_damping: float = CATEGORICAL_NATURAL_DAMPING,
) -> jnp.ndarray:
    r"""Per-layer weight gradient for the vectorised deep network.

    Returns the *descent* gradient for the weight matrix. Sign-flipped from the natural
    "ascent" formulation so it composes with standard optax (`apply_updates(weights,
    updates)` performs ``weights + updates``; `optax.sgd(lr).update(grad, state, w)`
    returns ``-lr * grad``; together they reproduce the legacy ``weights + lr * g``
    rule with ``g = -grad``).

    The two *kind* values share the same base term, the value prediction error
    :math:`\delta = \mu_\text{child} - \hat{\mu}_\text{child}` times the coupled
    parent activation :math:`a = g(\mu_\text{parent})`; they differ only in what
    scales it. Both are strictly *local*: the update for one weight only reads
    the prediction error and precision at its child and the activation at its
    parent. No reduction over other parents or children in the layer.

    - **standard** (``kind="standard"``) — the raw gradient of an *unweighted*
      squared error, no metric:
      :math:`g = \delta \otimes a`.
      This coincides with the free-energy gradient only when the child precision
      is one (e.g. the unit-precision output / categorical convention).

    - **precision_weighted** (``kind="precision_weighted"``) — the free-energy
      gradient weighted by the child's *posterior* precision:
      :math:`g = \delta \otimes a \cdot \pi_\text{child}`.
      This is the **backprop-parity** mode. A moved interior belief shifts by
      (routed error) / posterior precision, so weighting by that same posterior
      precision cancels the division and reproduces the backpropagated gradient
      node-for-node at *any* precision setting, not only the pinned recipe.

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations).
    coupling_fn :
        Coupling function applied to parent means.
    kind :
        Gradient computation mode.
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0,
        precision = 1.0) appended to its activations after coupling.
    child_is_binary :
        If True, the child layer is a binary node drops the redundant
        precision factor in ``precision_weighted`` (the Bernoulli variance
        cancels through the sigmoid in the gradient).

    Returns
    -------
    grad :
        Descent gradient, same shape as ``weights``. NaN / inf entries are
        zeroed out so optax does not propagate them through its moment
        accumulators.

    Raises
    ------
    ValueError
        If *kind* is unrecognised.
    """
    # Every kind is a rank-one product of a child-side and a parent-side factor
    # (see :func:`vectorized_weight_gradient_factors`), so the full matrix is
    # their outer product. The factor kernel is the single home of the per-kind
    # geometry, the descent-sign flip, and the ``kind`` validation. NaN/inf are
    # zeroed on the two factors there rather than on the assembled matrix; the
    # two agree except for a finite×finite overflow to inf, which neither path
    # special-cases.
    u, v = vectorized_weight_gradient_factors(
        parent_state,
        child_state,
        coupling_fn,
        kind=kind,
        parent_has_constant=parent_has_constant,
        child_is_binary=child_is_binary,
    )
    return u[:, None] * v[None, :]


# Both weight-update kinds factorise into a child-side and a parent-side
# vector (a rank-one product): pe (times, for precision_weighted, the child
# posterior precision) on one side, the coupled parent activation on the
# other. No reduction across other parents or children. Returning the two
# factors lets a batched caller average over samples with one contraction
# instead of materialising a weight-matrix-sized gradient per sample.
SEPARABLE_KINDS: tuple = ("standard", "precision_weighted")


def vectorized_weight_gradient_factors(
    parent_state: LayerState,
    child_state: LayerState,
    coupling_fn: Callable,
    kind: str = "precision_weighted",
    parent_has_constant: bool = False,
    child_is_binary: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Child- and parent-side factors of the weight gradient.

    For the separable gradient modes (:data:`SEPARABLE_KINDS`) the descent
    gradient of :func:`vectorized_weight_gradient` is a rank-one product,
    ``grad = u[:, None] * v[None, :]``. Returning the two vectors instead of
    their product lets a batched caller average gradients over many samples
    with a single contraction (``einsum('bi,bj->ij') / batch``). The same
    arithmetic, but without materialising one weight-matrix-sized gradient
    per sample, which is what dominates memory traffic at scale.

    Non-finite entries are zeroed on the factors, matching the per-entry
    zeroing of :func:`vectorized_weight_gradient` for vector-borne NaN/inf.

    Parameters
    ----------
    parent_state, child_state, coupling_fn, kind, parent_has_constant, child_is_binary :
        As in :func:`vectorized_weight_gradient`. *kind* must be one of
        :data:`SEPARABLE_KINDS`.

    Returns
    -------
    (u, v) :
        Child-side factor, shape ``(n_children,)``, and parent-side factor,
        shape ``(n_parents[+1],)``, such that ``u[:, None] * v[None, :]`` equals the
        descent gradient.

    Raises
    ------
    ValueError
        If *kind* is unrecognised.
    """
    if kind not in SEPARABLE_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {SEPARABLE_KINDS}.")

    pe = child_state.mean - child_state.expected_mean
    coupled_parent = coupling_fn(parent_state.mean)
    if parent_has_constant:
        coupled_parent = jnp.concatenate([coupled_parent, jnp.ones(1)])

    u = pe
    v = coupled_parent
    if kind == "precision_weighted" and not child_is_binary:
        u = u * child_state.precision  # posterior precision (backprop-parity gradient)

    u = jnp.where(jnp.isfinite(u), u, 0.0)
    v = jnp.where(jnp.isfinite(v), v, 0.0)

    # Descent sign, folded into the child-side factor.
    return -u, v
