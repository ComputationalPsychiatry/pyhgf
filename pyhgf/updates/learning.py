# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Attributes, Edges
from pyhgf.updates.posterior.volatile.volatile_node_posterior_update import (
    posterior_update_mean_value_level,
    posterior_update_precision_value_level,
)
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
    # 1. update the coupling strength between the child and value parent
    # ------------------------------------------------------------------
    pe = jnp.array(attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"])
    pe = jnp.where(pe == 0.0, 1e-8, pe)
    weighting = 1.0 / len(edges[node_idx].value_parents)  # type: ignore

    for value_parent_idx, value_coupling in zip(
        edges[node_idx].value_parents,  # type: ignore
        attributes[node_idx]["value_coupling_parents"],
    ):
        # prospective reconfiguration step:
        # infer the latent state that explain the outcome
        prospective_precision = posterior_update_precision_value_level(
            attributes, edges, value_parent_idx
        )
        prospective_mean = posterior_update_mean_value_level(
            attributes, edges, value_parent_idx, prospective_precision
        )

        # find the coupling function for this node
        coupling_fn = edges[value_parent_idx].coupling_fn[
            edges[value_parent_idx].value_children.index(node_idx)
        ]

        expected_coupling = attributes[node_idx]["mean"] / (
            coupling_fn(prospective_mean)
        )
        expected_coupling = jnp.where(
            jnp.isnan(expected_coupling), 0.0, expected_coupling
        )
        new_value_coupling = (
            value_coupling + (expected_coupling - value_coupling) * lr * weighting
        )

        # add a check to avoid inf coupling values
        new_value_coupling = jnp.where(
            jnp.isinf(new_value_coupling), value_coupling, new_value_coupling
        )

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
    # 1. update the coupling strength between the child and value parent
    # ------------------------------------------------------------------
    pe = jnp.array(attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"])
    pe = jnp.where(pe == 0.0, 1e-8, pe)
    weighting = 1.0 / len(edges[node_idx].value_parents)  # type: ignore

    for value_parent_idx, value_coupling in zip(
        edges[node_idx].value_parents,  # type: ignore
        attributes[node_idx]["value_coupling_parents"],
    ):
        # prospective reconfiguration step:
        # infer the latent state that explain the outcome
        prospective_precision = posterior_update_precision_value_level(
            attributes, edges, value_parent_idx
        )
        prospective_mean = posterior_update_mean_value_level(
            attributes, edges, value_parent_idx, prospective_precision
        )

        # find the coupling function for this node
        coupling_fn = edges[value_parent_idx].coupling_fn[
            edges[value_parent_idx].value_children.index(node_idx)
        ]

        expected_coupling = attributes[node_idx]["mean"] / coupling_fn(prospective_mean)
        expected_coupling = jnp.where(
            jnp.isnan(expected_coupling), 0.0, expected_coupling
        )

        precision_weighting = (
            attributes[node_idx]["precision"]
            / attributes[value_parent_idx]["precision"]
        )

        new_value_coupling = (
            value_coupling
            + (expected_coupling - value_coupling) * precision_weighting * weighting
        )

        # update the coupling strength in the attributes dictionary for both nodes
        attributes = set_coupling(
            parent_idx=value_parent_idx,
            child_idx=node_idx,
            coupling=new_value_coupling,
            edges=edges,
            attributes=attributes,
        )

    return attributes
