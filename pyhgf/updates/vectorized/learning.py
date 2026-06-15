# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized weight learning for deep predictive coding networks."""

from typing import Callable

import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerState


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

    The gradient is computed according to *kind*:

    - **standard** (``kind="standard"``):
      :math:`g = \text{PE} \otimes g(\text{parent})`
    - **precision_weighted** (``kind="precision_weighted"``):
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot \pi_\text{child}`
    - **precision_ratio** (``kind="precision_ratio"``): Kalman-gain-style gain
      using the parent's expected precision in the numerator.
      :math:`K = \pi_\text{parent} / (\pi_\text{parent} + \pi_\text{child})`,
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot K`
    - **map_natural** (``kind="map_natural"``): MAP weight update from the
      predictive-coding free energy with a Gaussian weight prior whose
      precision is the parent's expected precision; combines child precision
      (numerator) with parent prior plus per-weight Fisher curvature
      :math:`g(\text{parent})^2` (denominator).
      Gaussian child:
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot \pi_\text{child} / (\pi_\text{parent} + \pi_\text{child} \cdot g(\text{parent})^2)`.
      Binary child:
      :math:`g = \text{PE} \otimes g(\text{parent}) / (\pi_\text{parent} + g(\text{parent})^2)`.
    - **pure_natural** (``kind="pure_natural"``): Riemannian natural gradient
      under the parent's precision metric.
      Gaussian child:
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot \pi_\text{child} / \pi_\text{parent}`.
      Binary child:
      :math:`g = \text{PE} \otimes g(\text{parent}) / \pi_\text{parent}`.

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
        If True, the child layer is a binary node — drops the redundant
        precision factor in ``precision_weighted`` / ``map_natural`` /
        ``pure_natural`` modes (Bernoulli Fisher cancels through sigmoid).

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
    _valid_kinds = (
        "standard",
        "precision_weighted",
        "precision_ratio",
        "map_natural",
        "pure_natural",
    )
    if kind not in _valid_kinds:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {_valid_kinds}.")

    # Prediction error at the child layer.
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation. The constant bias node is wired in linearly
    # (g(1) = 1) regardless of coupling_fn, so the bias entry is appended
    # after the coupling has been applied to the activations.
    coupled_parent = coupling_fn(parent_state.mean)
    if parent_has_constant:
        coupled_parent = jnp.concatenate([coupled_parent, jnp.ones(1)])

    # Base outer product: PE ⊗ g(parent)
    gradient = pe[:, None] * coupled_parent[None, :]

    # Specialise by kind.
    if kind in ("precision_ratio", "map_natural", "pure_natural"):
        parent_precision = parent_state.expected_precision
        if parent_has_constant:
            # Constant state nodes have precision = 1.0 (fully known bias).
            parent_precision = jnp.concatenate([parent_precision, jnp.ones(1)])

    if kind == "precision_ratio" and not child_is_binary:
        kalman_gain = parent_precision[None, :] / (
            parent_precision[None, :] + child_state.expected_precision[:, None]
        )
        gradient = gradient * kalman_gain
    elif kind == "map_natural":
        if child_is_binary:
            gain = 1.0 / (parent_precision[None, :] + coupled_parent[None, :] ** 2)
        else:
            gain = child_state.expected_precision[:, None] / (
                parent_precision[None, :]
                + child_state.expected_precision[:, None] * coupled_parent[None, :] ** 2
            )
        gradient = gradient * gain
    elif kind == "pure_natural":
        if child_is_binary:
            gain = 1.0 / parent_precision[None, :]
        else:
            gain = child_state.expected_precision[:, None] / parent_precision[None, :]
        gradient = gradient * gain
    elif kind == "precision_weighted" and not child_is_binary:
        gradient = gradient * child_state.precision[:, None]

    # Zero out NaN/inf so optax moments stay finite.
    gradient = jnp.where(jnp.isnan(gradient) | jnp.isinf(gradient), 0.0, gradient)

    # Negate: HGF's natural formulation produces an ascent gradient; optax
    # expects descent. The combination ``apply_updates(w, sgd(lr).update(grad))``
    # reproduces ``w + lr * (-(-grad)) = w + lr * g`` matching legacy.
    return -gradient
