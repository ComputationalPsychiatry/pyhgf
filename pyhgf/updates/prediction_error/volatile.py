from functools import partial

from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.posterior.volatile import (
    volatile_node_posterior_update_ehgf,
    volatile_node_posterior_update_unbounded,
    volatile_node_volatility_posterior_update_standard,
)


@partial(jit, static_argnames=("node_idx", "edges"))
def volatile_node_value_prediction_error(
    attributes: dict, node_idx: int, edges: "Edges | None" = None
) -> dict:
    """Compute the value prediction error of the value level.

    This is used by external value parents (if any).
    """
    # Value PE for the value level
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # Divide by number of non-constant value parents (if any)
    if attributes[node_idx]["value_coupling_parents"] is not None:
        if edges is not None and edges[node_idx].value_parents is not None:
            n_non_const = sum(
                edges[p].node_type != 0
                for p in edges[node_idx].value_parents  # type: ignore[union-attr]
            )
            if n_non_const > 0:
                value_prediction_error /= n_non_const
        else:
            value_prediction_error /= len(
                attributes[node_idx]["value_coupling_parents"]
            )

    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error

    return attributes


@partial(jit, static_argnames=("node_idx",))
def volatile_node_volatility_prediction_error(attributes: dict, node_idx: int) -> dict:
    """Compute the volatility prediction error for the implicit volatility level.

    This is computed from the value level's precision surprise.
    """
    # Get value level parameters
    expected_precision = attributes[node_idx]["expected_precision"]
    precision = attributes[node_idx]["precision"]

    # Volatility PE from value level
    volatility_prediction_error = (
        (expected_precision / precision)
        + expected_precision
        * ((attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]) ** 2)
        - 1
    )

    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        volatility_prediction_error
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx", "update_type"))
def volatile_node_prediction_error(
    attributes: dict, node_idx: int, edges: Edges, update_type: str, **args
) -> dict:
    """Apply prediction errors and posterior updates to the volatility parent.

    - Value PE: for external value parents (if any)
    - Volatility PE: for the implicit internal volatility level
    """
    # 1. Prediction errors -------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # value prediction error
    attributes = volatile_node_value_prediction_error(attributes, node_idx, edges)

    # volatility prediction error
    attributes = volatile_node_volatility_prediction_error(attributes, node_idx)

    # 2. Posterior updates for the volatility parent -----------------------------------
    # ----------------------------------------------------------------------------------
    if update_type == "unbounded":
        attributes = volatile_node_posterior_update_unbounded(
            attributes=attributes, node_idx=node_idx
        )
    elif update_type == "eHGF":
        attributes = volatile_node_posterior_update_ehgf(
            attributes=attributes, edges=edges, node_idx=node_idx
        )
    elif update_type == "standard":
        attributes = volatile_node_volatility_posterior_update_standard(
            attributes=attributes, node_idx=node_idx
        )

    return attributes
