# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorized prediction error for categorical state node layers."""

import dataclasses

from pyhgf.typing.vectorised import LayerState


def vectorized_categorical_prediction_error(
    layer: LayerState,
) -> LayerState:
    r"""Compute the prediction error of a categorical state node layer.

    The value prediction error is the raw residual between the clamped observation (a
    one-hot pattern) and the softmax prediction:

    .. math::

        \delta = \mu - \hat{\mu} = \mathrm{one\_hot} - \mathrm{softmax}

    which is the cross-entropy gradient with respect to the logits. No scaling is needed
    (the binary layer, by contrast, divides by its Bernoulli variance). The posterior
    precision is set equal to the expected precision (both one), so the parent's
    smoothing gain is exactly one and the routed message is the residual multiplied back
    through the weights. Categorical leaves carry no volatility level: the volatility
    fields are left untouched.

    Parameters
    ----------
    layer :
        Current categorical layer state with ``mean`` (the clamped one-hot observation)
        and ``expected_mean`` set by the prediction step.

    Returns
    -------
    LayerState
        Updated layer state with ``value_prediction_error`` and ``precision`` set.
    """
    return dataclasses.replace(
        layer,
        value_prediction_error=layer.mean - layer.expected_mean,
        precision=layer.expected_precision,
    )
