# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Attributes, Edges
from pyhgf.utils import set_coupling


@partial(jit, static_argnames=("node_idx", "edges"))
def learning_weights(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
    lr: Optional[float] = None,
) -> Attributes:
    r"""Unified weights update.

    Branches on the ``lr`` parameter:

    - **Fixed** (``lr`` is a float): uses a fixed learning rate.
      :math:`\Delta w_i = \text{lr} \cdot (\text{PE} \cdot \pi_\text{child})
      \cdot g(\text{parent}_i)`
    - **Dynamic** (``lr is None``): uses a precision-based learning rate (Kalman gain).
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
        precision-weighted rule is used instead.

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

    if lr is None:
        parent_precisions = jnp.array([
            attributes[parent_idx].get("precision", 1.0)
            for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
        ])

    # 2. find the coupling function for each parent → child pair
    # -----------------------------------------------------------
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

    if lr is not None:
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
