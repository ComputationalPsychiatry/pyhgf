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
    - **dynamic** (``kind="dynamic"``): Kalman-gain-weighted PE.
      :math:`K = \pi_\text{child} / (\pi_\text{parent} + \pi_\text{child})`
      :math:`g = \text{PE} \otimes g(\text{parent}) \cdot K`

    *lr* controls how the gradient is applied (same semantics for all three
    kinds):

    - **float ≥ 0**: :math:`\Delta w = g \cdot \text{lr}`
    - ``"adam"``: gradient filtered through the Adam optimiser
      (Kingma & Ba, 2015); step size controlled by *adam_lr*.

    To recover the old "full Kalman step" behaviour for ``kind="dynamic"``,
    pass ``lr=1.0``.

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
        or ``"dynamic"``.
    lr :
        How the gradient is applied: a non-negative float for direct scaling,
        or ``"adam"`` for the Adam optimiser.  Applied uniformly across all
        *kind* values, including ``"dynamic"``.
    parent_has_constant :
        If True, the parent layer has a constant input node.  The parent
        mean is augmented with 1.0 and the precision with the parent
        precision mean so the bias column of *weights* is updated.
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
        or ``"dynamic"``.
    ValueError
        If *lr* is a string other than ``"adam"``.
    """
    if kind not in ("standard", "precision_weighted", "dynamic"):
        raise ValueError(
            f"Unknown kind '{kind}'. Expected 'standard', 'precision_weighted', "
            "or 'dynamic'."
        )
    if isinstance(lr, str) and lr != "adam":
        raise ValueError(
            f"Unknown lr value '{lr}'. Expected a non-negative float or 'adam'."
        )

    # Prediction error at child layer
    pe = child_state.mean - child_state.expected_mean

    # Coupled parent activation
    parent_mean = parent_state.mean
    parent_precision = parent_state.precision
    if parent_has_constant:
        parent_mean = jnp.concatenate([parent_mean, jnp.ones(1)])
        parent_precision = jnp.concatenate([
            parent_precision,
            jnp.array([jnp.mean(parent_precision)]),
        ])
    coupled_parent = coupling_fn(parent_mean)

    # Base outer product: PE ⊗ g(parent)
    base_delta = pe[:, None] * coupled_parent[None, :]

    # Compute the gradient according to *kind*
    if kind == "dynamic":
        kalman_gain = child_state.precision[:, None] / (
            parent_precision[None, :] + child_state.precision[:, None]
        )
        gradient = base_delta * kalman_gain
    elif kind == "precision_weighted" and not child_is_binary:
        gradient = base_delta * child_state.precision[:, None]
    else:  # "standard", or binary child where Bernoulli variance must not be doubled
        gradient = base_delta

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
