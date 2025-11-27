# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from pyhgf.typing import Attributes, Edges
from pyhgf.updates.prediction.continuous import predict_mean
from pyhgf.utils import set_coupling


@partial(jit, static_argnames=("node_idx", "edges"))
def learning_weights_fixed(
    attributes: Attributes,
    node_idx: int,
    edges: Edges,
    lr: float = 0.01,
) -> Attributes:
    r"""Update the coupling strengths with the value parents.

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
    # 1. update the weights of the connections to the value children
    # --------------------------------------------------------------

    # get the sum of parents' precision
    if edges[node_idx].value_parents is not None:
        pe = attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
        pe = jnp.where(pe == 0.0, 1e-8, pe)
        weighting = 1.0 / len(edges[node_idx].value_parents)  # type: ignore

        for value_parent_idx, value_coupling in zip(
            edges[node_idx].value_parents,  # type: ignore
            attributes[node_idx]["value_coupling_parents"],
        ):
            # find the coupling function for this node
            coupling_fn = edges[value_parent_idx].coupling_fn[
                edges[value_parent_idx].value_children.index(node_idx)
            ]

            # the new mean observed by the child that this parent seek to explain
            # here prediction error are equally shared among the parents
            observed = attributes[node_idx]["expected_mean"] + weighting * pe

            expected_coupling = observed / (
                coupling_fn(attributes[value_parent_idx]["mean"])
            )
            expected_coupling = jnp.where(
                jnp.isnan(expected_coupling), 0.0, expected_coupling
            )
            new_value_coupling = (
                value_coupling + (expected_coupling - value_coupling) * lr
            )

            # add a check to avoid inf coupling values
            new_value_coupling = jnp.where(jnp.isinf(new_value_coupling), value_coupling, new_value_coupling)

            # update the coupling strength in the attributes dictionary for both nodes
            set_coupling(
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
    r"""Update the coupling strengths with the value parents.

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
    # 1. update the weights of the connections to the value children
    # --------------------------------------------------------------

    # get the sum of parents' precision
    if edges[node_idx].value_parents is not None:
        sum_parents_precision = jnp.array(
            [
                attributes[parent_idx]["precision"]
                for parent_idx in edges[node_idx].value_parents  # type: ignore
            ]
        ).sum()

        pe = attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]

        for value_parent_idx, value_coupling in zip(
            edges[node_idx].value_parents,  # type: ignore
            attributes[node_idx]["value_coupling_parents"],
        ):
            # get the sum of squared prediction errors that was received by this parent
            sum_children_pe = jnp.array(
                [
                    attributes[child_idx]["temp"]["value_prediction_error"] ** 2
                    for child_idx in edges[value_parent_idx].value_children  # type: ignore
                ]
            ).sum()

            pe_weighting = (
                attributes[node_idx]["temp"]["value_prediction_error"] ** 2
                / sum_children_pe
            )

            # find the coupling function for this node
            coupling_fn = edges[value_parent_idx].coupling_fn[
                edges[value_parent_idx].value_children.index(node_idx)
            ]

            # get the precision weigthing for the coupling with this parent
            # the precision weighting represents how much of the new observed value
            # we want to explain with this parent node
            weighting = (
                attributes[value_parent_idx]["precision"] / sum_parents_precision
            )

            # the new mean observed by the child that this parent seek to explain
            observed = attributes[node_idx]["expected_mean"] + weighting * pe

            expected_coupling = observed / coupling_fn(
                attributes[value_parent_idx]["expected_mean"]
            )
            expected_coupling = jnp.where(
                jnp.isnan(expected_coupling), 0.0, expected_coupling
            )

            precision_weighting = (
                attributes[node_idx]["precision"]
                / (attributes[value_parent_idx]["precision"])
            )

            new_value_coupling = (
                value_coupling
                + (expected_coupling - value_coupling)
                * precision_weighting
                * pe_weighting
            )

            # update the coupling strength in the attributes dictionary for both nodes
            set_coupling(
                parent_idx=value_parent_idx,
                child_idx=node_idx,
                coupling=new_value_coupling,
                edges=edges,
                attributes=attributes,
            )

    # 2. call a new prediction step to update the node's mean and variance
    # --------------------------------------------------------------------
    attributes[node_idx]["expected_mean"] = predict_mean(
        attributes=attributes, node_idx=node_idx, edges=edges
    )

    return attributes
