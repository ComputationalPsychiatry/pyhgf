# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid as jax_sigmoid

from pyhgf.typing import Attributes, Edges
from pyhgf.utils import set_coupling


@partial(jit, static_argnames=("node_idx", "edges"))
def learning_weights(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
    lr: Optional[float] = None,
    adam_beta1: Optional[float] = None,
    adam_beta2: Optional[float] = None,
    adam_epsilon: Optional[float] = None,
) -> Attributes:
    r"""Unified weights update.

    Branches on the ``lr`` and ``adam_beta1`` parameters:

    - **Adam** (``adam_beta1`` is a float): uses the Adam optimiser.
    - **Fixed** (``lr`` is a float, no Adam): uses a fixed learning rate.
      :math:`\Delta w_i = \text{lr} \cdot (\text{PE} \cdot \pi_\text{child})
      \cdot g(\text{parent}_i)`
    - **Dynamic** (``lr is None``, no Adam): uses a precision-based learning rate
      (Kalman gain).
      :math:`K_i = \pi_{\text{parent}_i} / (\pi_{\text{parent}_i} + \pi_\text{child})`

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
    lr :
        Fixed learning rate. When ``None`` (default) the dynamic
        precision-weighted rule is used instead. When Adam is active, this is
        the Adam step size.
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
    # ---------------------------------------------------------------------------
    means = [
        attributes[parent_idx]["mean"]
        for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
    ]
    couplings = attributes[node_idx]["value_coupling_parents"]

    # constant-state parents (node_type 0) may lack a "precision" attribute;
    # default to 1.0 so the dynamic path treats them identically.
    child_precision = attributes[node_idx].get("precision", 1.0)

    if lr is None and adam_beta1 is None:
        parent_precisions = jnp.array([
            attributes[parent_idx].get("precision", 1.0)
            for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
        ])

    # 2. find the coupling function for each parent → child pair
    # Binary nodes always use sigmoid coupling in the weight update.
    # -----------------------------------------------------------
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

    # 3. compute the prospective activation for each parent
    # -------------------------------------------------------
    prospective_activation = jnp.array([
        fn(mean) if fn is not None else mean for mean, fn in zip(means, coupling_fns)
    ])

    # 4. update the coupling strength between this node and its value parents
    # -----------------------------------------------------------------------
    pe = attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]

    if adam_beta1 is not None:
        # Adam optimiser
        assert adam_beta2 is not None
        assert adam_epsilon is not None
        gradient = pe * child_precision * prospective_activation
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
    elif lr is not None:
        # Fixed learning rate:
        # Δw_i = lr · (PE · π_child) · g(parent_i)
        new_value_couplings = (
            couplings + lr * pe * child_precision * prospective_activation
        )
    else:
        # Dynamic (precision-weighted Kalman gain):
        # Δw_i = K_i · PE · g(parent_i)
        precision_weighting = parent_precisions / (parent_precisions + child_precision)
        new_value_couplings = (
            couplings + precision_weighting * pe * prospective_activation
        )

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
        # update the coupling strength in the attributes dictionary for both nodes
        attributes = set_coupling(
            parent_idx=value_parent_idx,
            child_idx=node_idx,
            coupling=new_value_coupling,
            edges=edges,
            attributes=attributes,
        )

    return attributes
