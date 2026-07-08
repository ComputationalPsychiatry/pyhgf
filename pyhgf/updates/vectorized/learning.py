# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized weight learning for deep predictive coding networks."""

from typing import Callable

import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerState

# The accepted weight-update kinds. Both share the same base term and factorise
# into a child-side and a parent-side vector (a rank-one product), which is why
# the gradient below is assembled as a single outer product.
SEPARABLE_KINDS: tuple = ("standard", "precision_weighted")


def vectorized_weight_gradient(
    parent_state: LayerState,
    child_state: LayerState,
    coupling_fn: Callable,
    kind: str = "precision_weighted",
    parent_has_constant: bool = False,
    child_is_binary: bool = False,
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
        If True, the child layer is a binary node, so the redundant precision
        factor is dropped in ``precision_weighted`` (the Bernoulli variance
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
    if kind not in SEPARABLE_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {SEPARABLE_KINDS}.")

    pe = child_state.mean - child_state.expected_mean
    coupled_parent = coupling_fn(parent_state.mean)
    if parent_has_constant:
        coupled_parent = jnp.concatenate([coupled_parent, jnp.ones(1)])

    # The gradient is the outer product of a child-side factor (the prediction
    # error, scaled by the child's posterior precision in ``precision_weighted``)
    # and a parent-side factor (the coupled parent activation). NaN / inf are
    # zeroed on each factor so optax does not propagate them.
    u = pe
    v = coupled_parent
    if kind == "precision_weighted" and not child_is_binary:
        u = u * child_state.precision  # posterior precision (backprop-parity gradient)

    u = jnp.where(jnp.isfinite(u), u, 0.0)
    v = jnp.where(jnp.isfinite(v), v, 0.0)

    # Descent sign, folded into the child-side factor.
    return -u[:, None] * v[None, :]
