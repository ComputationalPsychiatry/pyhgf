# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorised update functions for layers of regular continuous nodes.

Layer-wise vectorised implementations of the continuous HGF update equations (standard
drift semantics), with the value and volatility parents generalised to fully connected
layers. Mirrors :mod:`pyhgf.updates.prediction.continuous`,
:mod:`pyhgf.updates.prediction_error.continuous`, and
:mod:`pyhgf.updates.posterior.continuous`.
"""

from .posterior import (
    ValueChild,
    VolatilityChild,
    vectorised_continuous_posterior_update,
    vectorised_continuous_posterior_update_ehgf,
    vectorised_continuous_posterior_update_standard,
    vectorised_continuous_posterior_update_unbounded,
)
from .prediction import vectorised_continuous_prediction
from .prediction_error import (
    vectorised_continuous_prediction_error,
    vectorised_continuous_value_prediction_error,
    vectorised_continuous_volatility_prediction_error,
)

__all__ = [
    "ValueChild",
    "VolatilityChild",
    "vectorised_continuous_posterior_update",
    "vectorised_continuous_posterior_update_ehgf",
    "vectorised_continuous_posterior_update_standard",
    "vectorised_continuous_posterior_update_unbounded",
    "vectorised_continuous_prediction",
    "vectorised_continuous_prediction_error",
    "vectorised_continuous_value_prediction_error",
    "vectorised_continuous_volatility_prediction_error",
]
