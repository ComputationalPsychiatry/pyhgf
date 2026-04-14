# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized update functions for deep predictive coding networks.

This module provides layer-wise vectorized implementations of HGF
update equations that operate on entire layers instead of individual nodes.
"""

from .posterior import (
    vectorized_layer_posterior_update,
    vectorized_posterior_update_mean_value_level,
    vectorized_posterior_update_precision_value_level,
)
from .prediction import vectorized_layer_prediction
from .prediction_error import (
    vectorized_layer_prediction_error,
    vectorized_layer_value_prediction_error,
    vectorized_layer_volatility_posterior_ehgf,
    vectorized_layer_volatility_posterior_standard,
    vectorized_layer_volatility_posterior_unbounded,
    vectorized_layer_volatility_prediction_error,
)

__all__ = [
    "vectorized_layer_prediction",
    "vectorized_layer_prediction_error",
    "vectorized_layer_posterior_update",
    "vectorized_layer_value_prediction_error",
    "vectorized_layer_volatility_prediction_error",
    "vectorized_layer_volatility_posterior_ehgf",
    "vectorized_layer_volatility_posterior_standard",
    "vectorized_layer_volatility_posterior_unbounded",
    "vectorized_posterior_update_mean_value_level",
    "vectorized_posterior_update_precision_value_level",
]
