# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges

from .posterior_update_value_level import (
    posterior_update_mean_value_level,
    posterior_update_precision_value_level,
)
from .posterior_update_volatility_level import (
    posterior_update_mean_volatility_level,
    posterior_update_precision_volatility_level,
)


@partial(jit, static_argnames=("edges", "node_idx", "max_posterior_precision"))
def volatile_node_posterior_update(
    attributes: dict,
    edges: Edges,
    node_idx: int,
    max_posterior_precision: float = 1e10,
) -> dict:
    """Update a volatile node and the implied volatility parent.

    Unlike the standard continuous-state posterior updates elsewhere in the toolbox,
    the volatile-state updates use the *expected* mean (i.e. the prediction) as the
    reference point rather than the posterior mean. This choice is made to better suit
    deep learning networks where the prediction serves as the natural reference for
    computing updates.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`.
    node_idx :
        Pointer to the volatile node that needs to be updated.
    max_posterior_precision :
        Upper bound applied to the value-level posterior precision write.
        Default ``1e10``.
    """
    # Update precision first. Clip both ends:
    #   * upper bound at ``max_posterior_precision`` (precision blow-up guard);
    #   * lower bound at the *expected* precision — the nonlinear g'' term in
    #     the value-coupling contribution can swing strongly negative when a
    #     child's PE flips sign.
    precision_value = posterior_update_precision_value_level(
        attributes, edges, node_idx
    )
    precision_value = jnp.clip(
        precision_value,
        a_min=attributes[node_idx]["expected_precision"],
        a_max=max_posterior_precision,
    )
    attributes[node_idx]["precision"] = precision_value

    # Update mean using new precision
    mean_value = posterior_update_mean_value_level(
        attributes, edges, node_idx, precision_value
    )
    attributes[node_idx]["mean"] = mean_value

    return attributes


@partial(jit, static_argnames=("node_idx", "max_posterior_precision"))
def volatile_node_volatility_posterior_update_standard(
    attributes: dict,
    node_idx: int,
    max_posterior_precision: float = 1e10,
) -> dict:
    """Update the volatility level using the standard ordering.

    This updates the implicit volatility parent's mean and precision using the
    standard ordering: precision first, then mean using the updated precision.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the volatile node that needs to be updated.
    max_posterior_precision :
        Upper bound applied to the volatility-level posterior precision write.
        Default ``1e10``.
    """
    # Update precision first
    precision_vol = posterior_update_precision_volatility_level(attributes, node_idx)
    precision_vol = jnp.minimum(precision_vol, max_posterior_precision)
    attributes[node_idx]["precision_vol"] = precision_vol

    # Update mean using the new precision
    mean_vol = posterior_update_mean_volatility_level(
        attributes, node_idx, precision_vol
    )
    attributes[node_idx]["mean_vol"] = mean_vol

    return attributes
