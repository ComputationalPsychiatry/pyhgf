from functools import partial

from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx",))
def volatile_node_value_prediction_error(attributes: dict, node_idx: int) -> dict:
    """Compute the value prediction error of the value level.

    This is used by external value parents (if any).
    """
    # Value PE for the value level
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # Divide by number of value parents (if any)
    if attributes[node_idx]["value_coupling_parents"] is not None:
        value_prediction_error /= len(attributes[node_idx]["value_coupling_parents"])

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
    value_pe = attributes[node_idx]["temp"]["value_prediction_error"]

    # Volatility PE from value level
    volatility_prediction_error = (
        (expected_precision / precision) + expected_precision * (value_pe**2) - 1
    )

    # This is internal coupling (always 1 volatility "parent")
    # No division needed

    attributes[node_idx]["temp"]["volatility_prediction_error"] = (
        volatility_prediction_error
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def volatile_node_prediction_error(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Compute both value and volatility prediction errors.

    - Value PE: for external value parents (if any)
    - Volatility PE: for the implicit internal volatility level
    """
    # Compute value prediction error
    attributes = volatile_node_value_prediction_error(attributes, node_idx)

    # Compute volatility prediction error (internal)
    attributes = volatile_node_volatility_prediction_error(attributes, node_idx)

    return attributes
