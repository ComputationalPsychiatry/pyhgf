# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorized update functions for layers of regular continuous nodes.

Layer-wise vectorized implementations of the continuous HGF update equations (standard
drift semantics), with the value and volatility parents generalised to fully connected
layers. Mirrors :mod:`pyhgf.updates.prediction.continuous`,
:mod:`pyhgf.updates.prediction_error.continuous`, and
:mod:`pyhgf.updates.posterior.continuous`.
"""

from .posterior import (
    ValueChild,
    VolatilityChild,
    vectorized_continuous_posterior_update,
    vectorized_continuous_posterior_update_ehgf,
    vectorized_continuous_posterior_update_standard,
    vectorized_continuous_posterior_update_unbounded,
)
from .prediction import vectorized_continuous_prediction
from .prediction_error import (
    vectorized_continuous_prediction_error,
    vectorized_continuous_value_prediction_error,
    vectorized_continuous_volatility_prediction_error,
)

__all__ = [
    "ValueChild",
    "VolatilityChild",
    "vectorized_continuous_posterior_update",
    "vectorized_continuous_posterior_update_ehgf",
    "vectorized_continuous_posterior_update_standard",
    "vectorized_continuous_posterior_update_unbounded",
    "vectorized_continuous_prediction",
    "vectorized_continuous_prediction_error",
    "vectorized_continuous_value_prediction_error",
    "vectorized_continuous_volatility_prediction_error",
]
