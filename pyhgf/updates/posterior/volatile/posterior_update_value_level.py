# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial

from jax import grad, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_value_level(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> float:
    """Update the precision of the value level using value children's PEs.

    Implements the posterior-step (smoothing) correction of the relaxed HGF: the
    canonical child-precision factor ``π̂_a`` is replaced by
    ``π̂_a · (π_a − π̂_a) / π_a`` — the predicted precision scaled by the child's
    bottom-up information ratio. The same factor applies to both the ``g'²`` and the
    ``g''·δ_a`` contributions. Reduces to the canonical formula when the child is fully
    observed; returns no contribution when the child gained no bottom-up information.

    A child with no children of its own is an input: pyhgf's convention keeps
    ``precision = expected_precision`` for such nodes, so the smoothing form would zero
    out their contribution. Fall back to the canonical ``π̂_a``.

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.
    """
    # Start with expected precision
    posterior_precision = attributes[node_idx]["expected_precision"]

    # Add contributions from value children
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # Effective child precision under the smoothing correction. The Schur
            # derivation assumes a Gaussian-Gaussian value-coupling edge, so the
            # correction applies only when the child carries a Gaussian belief
            # (continuous-state type 2 or volatile-state type 6) AND is interior
            # (has children of its own). Binary, categorical, input/constant,
            # exponential family, and Dirichlet children, plus any Gaussian leaf,
            # fall back to the canonical ``π̂_a``.
            child_node_type = edges[value_child_idx].node_type
            child_is_gaussian_interior = child_node_type in (2, 6) and (
                edges[value_child_idx].value_children is not None
                or edges[value_child_idx].volatility_children is not None
            )
            child_expected_precision = attributes[value_child_idx]["expected_precision"]
            if child_is_gaussian_interior:
                child_precision = attributes[value_child_idx]["precision"]
                effective_child_precision = (
                    child_expected_precision
                    * (child_precision - child_expected_precision)
                    / child_precision
                )
            else:
                effective_child_precision = child_expected_precision

            # Linear coupling
            if coupling_fn is None:
                posterior_precision += (value_coupling**2) * effective_child_precision
            else:
                # Non-linear coupling (with gradient)
                coupling_fn_prime = grad(coupling_fn)(
                    attributes[node_idx]["expected_mean"]
                )
                coupling_fn_second = grad(grad(coupling_fn))(
                    attributes[node_idx]["expected_mean"]
                )
                value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

                posterior_precision += effective_child_precision * (
                    (value_coupling**2) * (coupling_fn_prime**2)
                    - value_coupling * coupling_fn_second * value_pe
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

    .. note::

        Unlike the standard continuous-state posterior updates elsewhere in the
        toolbox, the volatile-state updates evaluate coupling function derivatives
        at the *expected* mean (i.e. the prediction) rather than the posterior
        mean. This choice is made to better suit deep learning networks where the
        prediction serves as the natural reference point for computing updates.
    """
    # Start with expected mean
    posterior_mean = attributes[node_idx]["expected_mean"]

    # Add contributions from value children
    value_precision_weighted_pe = 0.0

    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling, coupling_fn in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
            edges[node_idx].coupling_fn,
        ):
            # Get the value prediction error
            value_pe = attributes[value_child_idx]["temp"]["value_prediction_error"]

            # Get coupling function derivative
            if coupling_fn is None:
                coupling_fn_prime = 1
            else:
                coupling_fn_prime = grad(coupling_fn)(
                    attributes[node_idx]["expected_mean"]
                )

            # Expected precision from child
            child_expected_precision = attributes[value_child_idx]["expected_precision"]

            # Accumulate precision-weighted PE
            value_precision_weighted_pe += (
                (value_coupling * child_expected_precision * coupling_fn_prime)
                / node_precision
            ) * value_pe

    posterior_mean += value_precision_weighted_pe

    return posterior_mean
