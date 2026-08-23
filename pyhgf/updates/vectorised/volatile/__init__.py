# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised update functions for deep predictive coding networks.

This module provides layer-wise vectorised implementations of HGF update equations that
operate on entire layers instead of individual nodes.
"""

from .posterior import (
    vectorised_layer_posterior_update,
    vectorised_posterior_update_mean_value_level,
    vectorised_posterior_update_precision_value_level,
)
from .prediction import vectorised_layer_prediction, vectorised_root_prediction
from .prediction_error import (
    vectorised_layer_prediction_error,
    vectorised_layer_value_prediction_error,
    vectorised_layer_volatility_posterior_ehgf,
    vectorised_layer_volatility_posterior_standard,
    vectorised_layer_volatility_posterior_unbounded,
    vectorised_layer_volatility_prediction_error,
)

__all__ = [
    "vectorised_layer_prediction",
    "vectorised_layer_prediction_error",
    "vectorised_layer_posterior_update",
    "vectorised_layer_value_prediction_error",
    "vectorised_layer_volatility_prediction_error",
    "vectorised_layer_volatility_posterior_ehgf",
    "vectorised_layer_volatility_posterior_standard",
    "vectorised_layer_volatility_posterior_unbounded",
    "vectorised_posterior_update_mean_value_level",
    "vectorised_posterior_update_precision_value_level",
    "vectorised_root_prediction",
]
