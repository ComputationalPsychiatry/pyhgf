# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorized prediction errors for layers of regular continuous nodes."""

import dataclasses

from pyhgf.typing.vectorised import LayerState


def vectorized_continuous_value_prediction_error(layer: LayerState) -> LayerState:
    r"""Compute the value prediction error for all nodes in a continuous layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.continuous.continuous_node_value_prediction_error`:

    .. math::

        \delta_a^{(k)} = \mu_a^{(k)} - \hat{\mu}_a^{(k)}.

    Parameters
    ----------
    layer :
        Current layer with ``mean`` and ``expected_mean`` set.

    Returns
    -------
    LayerState
        Updated layer state with ``value_prediction_error`` set.
    """
    return dataclasses.replace(
        layer, value_prediction_error=layer.mean - layer.expected_mean
    )


def vectorized_continuous_volatility_prediction_error(layer: LayerState) -> LayerState:
    r"""Compute the volatility prediction error for all nodes in a continuous layer.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.continuous.continuous_node_volatility_prediction_error`:

    .. math::

        \Delta_a^{(k)} = \frac{\tilde{\pi}_a^{(k)}}{\pi_a^{(k)}}
            + \tilde{\pi}_a^{(k)} \left( \delta_a^{(k)} \right)^2 - 1.

    The nodalised backend divides by the number of volatility parents; the
    vectorized layer topology allows at most one volatility parent per layer, so
    no division is applied.

    Parameters
    ----------
    layer :
        Current layer with ``mean``, ``expected_mean``, ``precision`` and
        ``expected_precision`` set.

    Returns
    -------
    LayerState
        Updated layer state with ``volatility_prediction_error`` set.
    """
    volatility_pe = (
        (layer.expected_precision / layer.precision)
        + layer.expected_precision * (layer.mean - layer.expected_mean) ** 2
        - 1.0
    )
    return dataclasses.replace(layer, volatility_prediction_error=volatility_pe)


def vectorized_continuous_prediction_error(
    layer: LayerState,
    has_volatility_parent: bool = False,
) -> LayerState:
    """Compute the prediction errors a continuous layer sends to its parents.

    This is the vectorized equivalent of
    :func:`pyhgf.updates.prediction_error.continuous.continuous_node_prediction_error`.
    The value prediction error is always computed; the volatility prediction
    error only when there is a volatility parent to consume it.

    Parameters
    ----------
    layer :
        Current layer with posterior ``mean`` and ``precision`` set.
    has_volatility_parent :
        Whether the layer has a volatility-parent layer.

    Returns
    -------
    LayerState
        Updated layer state with the prediction errors set.
    """
    layer = vectorized_continuous_value_prediction_error(layer)
    if has_volatility_parent:
        layer = vectorized_continuous_volatility_prediction_error(layer)
    return layer
