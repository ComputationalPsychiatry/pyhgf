# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction error computation for volatile node layers."""

import jax.numpy as jnp

from pyhgf.typing import LayerState


def vectorized_layer_prediction_error(
    layer: LayerState,
    n_parents: int,
    sqrt_normalization: bool = False,
    no_normalization: bool = False,
) -> LayerState:
    """Compute prediction errors for all nodes in a layer.

    This computes both value prediction error (for value parents)
    and volatility prediction error (for internal volatility level).

    Parameters
    ----------
    layer :
        Current layer with mean and expected_mean set.
    n_parents :
        Number of value parents for this layer (for normalization).
    sqrt_normalization :
        If True, normalize value PE by ``sqrt(n_parents)`` instead of
        ``n_parents``. Ignored when ``no_normalization=True``.
    no_normalization :
        If True, use the raw PE with no division. Overrides
        ``sqrt_normalization``. Default is False.

    Returns
    -------
    LayerState
        Updated layer state with prediction errors computed.
    """
    # Value prediction error
    raw_pe = layer.mean - layer.expected_mean
    if no_normalization:
        value_pe = raw_pe
    elif sqrt_normalization:
        value_pe = raw_pe / jnp.sqrt(n_parents)
    else:
        value_pe = raw_pe / n_parents

    # Volatility prediction error (from value level precision surprise)
    # This is computed using the pre-update values
    volatility_pe = (
        (layer.expected_precision / layer.precision)
        + layer.expected_precision * (value_pe**2)
        - 1.0
    )

    return layer._replace(
        value_prediction_error=value_pe,
        volatility_prediction_error=volatility_pe,
    )
