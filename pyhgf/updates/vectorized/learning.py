# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized weight learning for deep predictive coding networks."""

from typing import Callable, Optional, Union

import jax.numpy as jnp

from pyhgf.typing import LayerState


def vectorized_weight_update(
    parent_state: LayerState,
    child_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    kind: str = "precision_weighted",
    lr: Union[float, str] = 0.0,
    parent_has_constant: bool = False,
    child_is_binary: bool = False,
    adam_m: Optional[jnp.ndarray] = None,
    adam_v: Optional[jnp.ndarray] = None,
    adam_t: int = 0,
    adam_lr: float = 1e-3,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    r"""Unified weight update for vectorized layers.

    The gradient is first computed according to *kind*, then scaled by *lr*
    (uniformly across all modes):

    - **standard** (``kind="standard"``):
      :math:`g = \text{PE} \otimes g(\text{parent})`
    - **precision_weighted** (``kind="precision_weighted"``):
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot \pi_\text{child}`
    - **precision_ratio** (``kind="precision_ratio"``): Kalman-gain-style gain
      using the parent's expected precision in the numerator.
      :math:`K = \pi_\text{parent} / (\pi_\text{parent} + \pi_\text{child})`
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot K`
    - **map_natural** (``kind="map_natural"``): MAP weight update derived from
      the predictive-coding free energy with a Gaussian weight prior whose
      precision is the parent layer's expected precision. Combines child
      precision (numerator) with the parent prior plus per-weight Fisher
      curvature :math:`g(\text{parent})^2` (denominator), giving a bounded,
      curvature-aware update that uses both precisions.
      Gaussian child:
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot \pi_\text{child} / (\pi_\text{parent} + \pi_\text{child} \cdot g(\text{parent})^2)`.
      Binary child (drop the redundant :math:`\pi_\text{child}` factor since
      the Bernoulli Fisher cancels through the sigmoid):
      :math:`g = \text{PE} \otimes g(\text{parent}) / (\pi_\text{parent} + g(\text{parent})^2)`.
    - **pure_natural** (``kind="pure_natural"``): Riemannian natural gradient
      under the parent's precision metric — uses both precisions with no
      curvature term, no extra hyperparameter.
      Gaussian child:
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot \pi_\text{child} / \pi_\text{parent}`.
      Binary child:
      :math:`g = \text{PE} \otimes g(\text{parent}) / \pi_\text{parent}`.
      Not bounded — risks blowing up when :math:`\pi_\text{parent}` is small.

    *lr* controls how the gradient is applied (same semantics for all five
    kinds):

    - **float ≥ 0**: :math:`\Delta w = g \cdot \text{lr}`
    - ``"adam"``: gradient filtered through the Adam optimiser
      (Kingma & Ba, 2015); step size controlled by *adam_lr*.

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations).
    weights :
        Current weight matrix, shape ``(n_children, n_parents)`` or
        ``(n_children, n_parents + 1)`` when the parent layer includes
        a constant input node.
    coupling_fn :
        Coupling function applied to parent means.
    kind :
        Gradient computation mode: ``"standard"``, ``"precision_weighted"``,
        ``"precision_ratio"``, ``"map_natural"``, or ``"pure_natural"``.
    lr :
        How the gradient is applied: a non-negative float for direct scaling,
        or ``"adam"`` for the Adam optimiser.  Applied uniformly across all
        *kind* values, including ``"precision_ratio"``.
    parent_has_constant :
        If True, the parent layer has a constant input node. Constant nodes are assumed
        to have mean = 1.0 and precision = 1.0 (fully known bias), and are concatenated
        to the coupled parent vector after ``coupling_fn`` is applied so the bias entry
        is unconditionally linear regardless of the coupling function.
    child_is_binary :
        If True, the child layer is a binary node.  In ``"precision_weighted"``
        mode the precision multiplication is skipped because the Bernoulli
        variance is already embedded in the binary prediction-error formula.
    adam_m :
        First moment estimate for Adam, same shape as *weights*.
        Required when ``lr="adam"``.
    adam_v :
        Second moment estimate for Adam, same shape as *weights*.
        Required when ``lr="adam"``.
    adam_t :
        Global Adam timestep (already incremented for this step).
    adam_lr :
        Adam step size.  Only used when ``lr="adam"``.
    adam_beta1 :
        Exponential decay rate for the first moment.
    adam_beta2 :
        Exponential decay rate for the second moment.
    adam_epsilon :
        Small constant for numerical stability.

    Returns
    -------
    new_weights :
        Updated weight matrix.
    new_adam_m :
        Updated first moment (or *None* when Adam is not used).
    new_adam_v :
        Updated second moment (or *None* when Adam is not used).

    Raises
    ------
    ValueError
        If *kind* is not one of ``"standard"``, ``"precision_weighted"``,
        ``"precision_ratio"``, ``"map_natural"``, or ``"pure_natural"``.
    ValueError
        If *lr* is a string other than ``"adam"``.
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
    if isinstance(lr, str) and lr != "adam":
        raise ValueError(
            f"Unknown lr value '{lr}'. Expected a non-negative float or 'adam'."
        )

    # Prediction error at child layer
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation. The coupling is applied to the parent's actual
    # activations only; the constant bias node is always wired in linearly
    # (g(1) = 1) regardless of coupling_fn, so the bias entry is appended to the
    # coupled vector untouched.
    coupled_parent = coupling_fn(parent_state.mean)
    if parent_has_constant:
        coupled_parent = jnp.concatenate([coupled_parent, jnp.ones(1)])

    # Base outer product: PE ⊗ g(parent)
    gradient = pe[:, None] * coupled_parent[None, :]

    # Compute the gradient according to *kind*
    if kind in ("precision_ratio", "map_natural", "pure_natural"):
        # All three rules need the parent's expected precision, optionally
        # extended with a 1.0 entry for the constant bias node.
        parent_precision = parent_state.expected_precision
        if parent_has_constant:
            # Constant state nodes are assumed to have mean = 1.0 and
            # precision = 1.0 (fully known bias).
            parent_precision = jnp.concatenate([parent_precision, jnp.ones(1)])

    if kind == "precision_ratio" and not child_is_binary:
        kalman_gain = parent_precision[None, :] / (
            parent_precision[None, :] + child_state.expected_precision[:, None]
        )
        gradient *= kalman_gain
    elif kind == "map_natural":
        # MAP weight update: free-energy gradient with a Gaussian weight prior
        # whose precision equals the parent's expected precision.
        # Denominator includes the per-weight Fisher curvature g(parent)**2.
        if child_is_binary:
            # Bernoulli Fisher cancels through sigmoid → drop child precision.
            gain = 1.0 / (parent_precision[None, :] + coupled_parent[None, :] ** 2)
        else:
            gain = child_state.expected_precision[:, None] / (
                parent_precision[None, :]
                + child_state.expected_precision[:, None] * coupled_parent[None, :] ** 2
            )
        gradient *= gain
    elif kind == "pure_natural":
        # Riemannian natural gradient under the parent's precision metric.
        # No bounding term — can blow up if parent_precision is small.
        if child_is_binary:
            gain = 1.0 / parent_precision[None, :]
        else:
            gain = child_state.expected_precision[:, None] / parent_precision[None, :]
        gradient *= gain
    elif kind == "precision_weighted" and not child_is_binary:
        gradient *= child_state.precision[:, None]

    # Apply *lr* uniformly across all kinds
    if lr == "adam":
        assert adam_m is not None and adam_v is not None, (
            "adam_m and adam_v must be provided when lr='adam'"
        )
        new_m = adam_beta1 * adam_m + (1.0 - adam_beta1) * gradient
        new_v = adam_beta2 * adam_v + (1.0 - adam_beta2) * gradient**2
        m_hat = new_m / (1.0 - adam_beta1**adam_t)
        v_hat = new_v / (1.0 - adam_beta2**adam_t)
        coupling_delta = adam_lr * m_hat / (jnp.sqrt(v_hat) + adam_epsilon)
    else:
        coupling_delta = gradient * float(lr)
        new_m = None
        new_v = None

    # Guard against NaN / inf
    coupling_delta = jnp.where(
        jnp.isnan(coupling_delta) | jnp.isinf(coupling_delta), 0.0, coupling_delta
    )

    new_weights = weights + coupling_delta
    new_weights = jnp.where(jnp.isinf(new_weights), weights, new_weights)

    return new_weights, new_m, new_v
