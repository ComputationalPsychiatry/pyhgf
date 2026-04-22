# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized prediction error for binary state node layers."""

from pyhgf.typing import LayerState


def vectorized_binary_prediction_error(
    layer: LayerState,
    n_parents: int = 1,
) -> LayerState:
    r"""Compute prediction errors for a binary state node layer.

    The value prediction error for binary nodes is scaled by the inverse of the
    expected precision (Bernoulli variance) so it can be consumed directly by the
    continuous parent's posterior update without requiring a different update step,
    then divided by the number of value parents to distribute the signal:

    .. math::

        \\delta_b = \\frac{\\mu_b - \\hat{\\mu}_b}{\\hat{\\pi}_b \cdot N_{\\text{parents}}}

    The posterior precision of a binary node is set equal to the expected precision
    (there is no precision update for binary nodes).

    Parameters
    ----------
    layer :
        Current binary layer state with ``mean`` (observation) and
        ``expected_mean``/``expected_precision`` set by the prediction step.
    n_parents :
        Number of value parents for this layer (default: 1).  The PE is divided
        by this count so each parent receives its proportional share of the signal.

    Returns
    -------
    LayerState
        Updated layer state with ``value_prediction_error`` and ``precision``.
    """
    # Raw prediction error
    value_pe = layer.mean - layer.expected_mean

    # Scale by inverse expected precision (eq. 98 in Weber et al., v1)
    value_pe = value_pe / layer.expected_precision

    # Distribute across value parents
    value_pe = value_pe / n_parents

    # Posterior precision = expected precision for binary nodes
    precision = layer.expected_precision

    return layer._replace(
        value_prediction_error=value_pe,
        precision=precision,
    )
