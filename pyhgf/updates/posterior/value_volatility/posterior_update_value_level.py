from functools import partial
from jax import grad, jit
import jax.numpy as jnp
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> float:
    """Update the precision of the value level using value children's PEs.
    
    This is similar to continuous node value coupling precision update.
    """
    # Start with expected precision
    posterior_precision = attributes[node_idx]["expected_precision"]

    # Add contributions from value children
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # Get child's expected precision
            child_expected_precision = attributes[value_child_idx]["expected_precision"]

            # Linear coupling
            if coupling_fn is None:
                posterior_precision += (value_coupling ** 2) * child_expected_precision
            else:
                # Non-linear coupling (with gradient)
                coupling_fn_prime = grad(coupling_fn)(attributes[node_idx]["mean"])
                coupling_fn_second = grad(grad(coupling_fn))(attributes[node_idx]["mean"])
                value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

                posterior_precision += child_expected_precision * (
                    (value_coupling ** 2) * (coupling_fn_prime ** 2)
                    - coupling_fn_second * value_pe
                )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    node_precision: float,
) -> float:
    """Update the mean of the value level using value children's PEs.
    
    This is similar to continuous node value coupling mean update.
    """
    # Start with expected mean
    posterior_mean = attributes[node_idx]["expected_mean"]

    # Add contributions from value children
    value_precision_weighted_pe = 0.0

    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # Get the value prediction error
            value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

            # Cancel if child not observed
            value_pe *= attributes[value_child_idx]["observed"]

            # Get coupling function derivative
            if coupling_fn is None:
                coupling_fn_prime = 1
            else:
                coupling_fn_prime = grad(coupling_fn)(attributes[node_idx]["mean"])

            # Expected precision from child
            child_expected_precision = attributes[value_child_idx]["expected_precision"]

            # Accumulate precision-weighted PE
            value_precision_weighted_pe += (
                (value_coupling * child_expected_precision * coupling_fn_prime)
                / node_precision
            ) * value_pe

    posterior_mean += value_precision_weighted_pe

    return posterior_mean