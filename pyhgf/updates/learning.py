# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Attributes, Edges
from pyhgf.utils import set_coupling


@partial(jit, static_argnames=("node_idx", "edges"))
def learning_weights_fixed(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
    lr: float = 0.01,
) -> Attributes:
    r"""Weights update using a fixed learning rate.

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

    # 2. find the coupling function for each parent → child pair
    # -----------------------------------------------------------
    # coupling_fn is stored on the *parent* node, indexed by the child's
    # position in the parent's value_children list
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
    # Backprop-style delta rule: Δw_i = lr · PE · g(parent_i)
    # Each weight is updated proportionally to the parent's activation,
    # matching the gradient of 0.5·PE² w.r.t. w_i.
    pe = attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    new_value_couplings = couplings + lr * pe * prospective_activation

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


@partial(jit, static_argnames=("node_idx", "edges"))
def learning_weights_dynamic(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
) -> Attributes:
    r"""Dynamic weights update.

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
    # constant-state parents (node_type 0) have no "precision" attribute;
    # use the same default as regular nodes (1.0) so that precision-weighted
    # learning treats them identically.
    precisions = [
        attributes[parent_idx].get("precision", 1.0)
        for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
    ]
    couplings = attributes[node_idx]["value_coupling_parents"]

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
    # Backprop-style delta rule with precision-based learning rate:
    # Δw_i = precision_weight_i · PE · g(parent_i)
    precision_weighting = attributes[node_idx]["precision"] / (
        jnp.array(precisions) + attributes[node_idx]["precision"]
    )
    pe = attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    new_value_couplings = couplings + precision_weighting * pe * prospective_activation

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
