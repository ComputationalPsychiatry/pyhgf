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
    expected_means = [
        attributes[parent_idx]["expected_mean"]
        for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
    ]
    couplings = attributes[node_idx]["value_coupling_parents"]

    # 2. compute the expected activation for the current node
    # -------------------------------------------------------
    prospective_activation = jnp.array([
        coupling_fn(mean) if coupling_fn is not None else mean
        for mean, coupling_fn in zip(means, edges[node_idx].coupling_fn)  # type: ignore[union-attr]
    ])
    current_activation = jnp.array([
        coupling_fn(mean) if coupling_fn is not None else mean
        for mean, coupling_fn in zip(expected_means, edges[node_idx].coupling_fn)  # type: ignore[union-attr]
    ])

    # target coupling per parent: solve for w_i given all other parents' contributions
    # (child_mean - contribution_of_others) / g_i(parent_mean)
    expected_couplings = (
        attributes[node_idx]["mean"]
        - (attributes[node_idx]["expected_mean"] - current_activation * couplings)
    ) / prospective_activation

    # guard against NaN/inf when activation ≈ 0
    expected_couplings = jnp.where(
        jnp.isnan(expected_couplings) | jnp.isinf(expected_couplings),
        couplings,
        expected_couplings,
    )

    # 3. update the coupling strength between this node and its value parents
    # -----------------------------------------------------------------------
    weighting = 1.0 / len(edges[node_idx].value_parents)  # type: ignore[operator,arg-type]
    new_value_couplings = couplings + (expected_couplings - couplings) * lr * weighting

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
    precisions = [
        attributes[parent_idx]["precision"]
        for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
    ]
    expected_means = [
        attributes[parent_idx]["expected_mean"]
        for parent_idx in edges[node_idx].value_parents  # type: ignore[union-attr]
    ]
    couplings = attributes[node_idx]["value_coupling_parents"]

    # 2. compute the expected activation for the current node
    # -------------------------------------------------------
    prospective_activation = jnp.array([
        coupling_fn(mean) if coupling_fn is not None else mean
        for mean, coupling_fn in zip(means, edges[node_idx].coupling_fn)  # type: ignore[union-attr]
    ])
    current_activation = jnp.array([
        coupling_fn(mean) if coupling_fn is not None else mean
        for mean, coupling_fn in zip(expected_means, edges[node_idx].coupling_fn)  # type: ignore[union-attr]
    ])

    # target coupling per parent: solve for w_i given all other parents' contributions
    # (child_mean - contribution_of_others) / g_i(parent_mean)
    expected_couplings = (
        attributes[node_idx]["mean"]
        - (attributes[node_idx]["expected_mean"] - current_activation * couplings)
    ) / prospective_activation

    # guard against NaN/inf when activation ≈ 0
    expected_couplings = jnp.where(
        jnp.isnan(expected_couplings) | jnp.isinf(expected_couplings),
        couplings,
        expected_couplings,
    )

    # 3. update the coupling strength between this node and its value parents
    # -----------------------------------------------------------------------

    # use expected_precision (prediction-time) for both child and parent
    # to avoid asymmetry from update ordering (child already posterior-updated,
    # parent not yet)
    precision_weighting = attributes[node_idx]["precision"] / (
        precisions + attributes[node_idx]["precision"]
    )

    weighting = 1.0 / len(edges[node_idx].value_parents)  # type: ignore[operator,arg-type]
    new_value_couplings = (
        couplings + (expected_couplings - couplings) * precision_weighting * weighting
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
