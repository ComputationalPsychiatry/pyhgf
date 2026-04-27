# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid as jax_sigmoid

from pyhgf.typing import Attributes, Edges
from pyhgf.utils import set_coupling


@partial(jit, static_argnames=("node_idx", "edges", "kind"))
def learning_weights(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
    kind: str = "precision_weighted",
    lr: Optional[float] = None,
    adam_beta1: Optional[float] = None,
    adam_beta2: Optional[float] = None,
    adam_epsilon: Optional[float] = None,
) -> Attributes:
    r"""Unified weights update.

    The gradient is first computed according to *kind*, then scaled by *lr*:

    - **standard** (``kind="standard"``): raw prediction-error outer product, no
    precision weighting.
      :math:`g_i = \text{PE} \cdot g(\text{parent}_i)`
    - **precision_weighted** (``kind="precision_weighted"``): gradient weighted
      by the child posterior precision.
      :math:`g_i = \text{PE} \cdot \pi_\text{child} \cdot g(\text{parent}_i)`
    - **precision_ratio** (``kind="precision_ratio"``): Kalman-gain-weighted PE
      using the posterior precisions of child and parent.
      :math:`K_i = \pi_\text{child} / (\pi_{\text{parent}_i} + \pi_\text{child})`
      :math:`g_i = K_i \cdot \text{PE} \cdot g(\text{parent}_i)`

    *lr* controls how the gradient is applied (same semantics for all three kinds):

    - **Adam** (``adam_beta1`` is a float): gradient filtered through Adam; step size
    controlled by *lr*.
    - **Fixed** (``adam_beta1`` is None): :math:`\Delta w_i = \text{lr} \cdot g_i`.

    To recover the old "full Kalman step" behaviour for ``kind="precision_ratio"``,
    pass ``lr=1.0``.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network.
    node_idx :
        Pointer to the input node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    kind :
        Gradient computation mode: ``"standard"``, ``"precision_weighted"`` (default),
        or ``"precision_ratio"``.
    lr :
        Fixed learning rate or Adam step size.  Applied uniformly across all *kind*
        values, including ``"precision_ratio"``.
    adam_beta1 :
        Adam first moment decay rate.  When ``None`` (default) Adam is not used.
    adam_beta2 :
        Adam second moment decay rate.
    adam_epsilon :
        Adam numerical stability constant.

    Returns
    -------
    attributes :
        The attributes of the probabilistic network.

    """
    # 1. get the prospective activation vector from the upper layer with coupling
    means = [
        attributes[parent_idx]["mean"]
        for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
    ]
    couplings = attributes[node_idx]["value_coupling_parents"]

    child_precision = attributes[node_idx].get("precision", 1.0)

    # 2. find the coupling function for each parent → child pair
    if edges[node_idx].node_type == 1:  # binary-state
        coupling_fns = [jax_sigmoid for _ in edges[node_idx].value_parents]  # type: ignore[union-attr]
    else:
        coupling_fns = [
            edges[parent_idx].coupling_fn[  # type: ignore[index]
                edges[parent_idx].value_children.index(node_idx)  # type: ignore[union-attr]
            ]
            if edges[parent_idx].coupling_fn is not None
            else None
            for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
        ]

    # 3. prospective activation for each parent
    prospective_activation = jnp.array([
        fn(mean) if fn is not None else mean for mean, fn in zip(means, coupling_fns)
    ])

    # 4. compute the gradient according to *kind*
    pe = attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    is_binary = edges[node_idx].node_type == 1

    if kind == "precision_ratio":
        parent_precisions = jnp.array([
            attributes[parent_idx].get("precision", 1.0)
            for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
        ])

        kalman_gain = child_precision / (parent_precisions + child_precision)
        gradient = kalman_gain * pe * prospective_activation
    elif kind == "precision_weighted" and not is_binary:
        gradient = pe * child_precision * prospective_activation
    else:  # "standard", or binary child where Bernoulli variance must not be doubled
        gradient = pe * prospective_activation

    # 5. apply *lr* uniformly across all kinds (direct scaling or Adam)
    if adam_beta1 is not None:
        assert adam_beta2 is not None
        assert adam_epsilon is not None
        adam_m = (
            adam_beta1 * attributes[node_idx]["adam_m"] + (1 - adam_beta1) * gradient
        )
        adam_v = (
            adam_beta2 * attributes[node_idx]["adam_v"] + (1 - adam_beta2) * gradient**2
        )
        adam_t = attributes[-1]["adam_t"]
        m_hat = adam_m / (1 - adam_beta1**adam_t)
        v_hat = adam_v / (1 - adam_beta2**adam_t)
        new_value_couplings = couplings + lr * m_hat / (jnp.sqrt(v_hat) + adam_epsilon)
        attributes[node_idx]["adam_m"] = adam_m
        attributes[node_idx]["adam_v"] = adam_v
    else:
        new_value_couplings = couplings + lr * gradient

    # guard against inf/nan coupling values
    new_value_couplings = jnp.where(
        jnp.isinf(new_value_couplings) | jnp.isnan(new_value_couplings),
        couplings,
        new_value_couplings,
    )

    for value_parent_idx, new_value_coupling in zip(
        edges[node_idx].value_parents,  # type: ignore[arg-type]
        new_value_couplings,
    ):
        attributes = set_coupling(
            parent_idx=value_parent_idx,
            child_idx=node_idx,
            coupling=new_value_coupling,
            edges=edges,
            attributes=attributes,
        )

    return attributes
