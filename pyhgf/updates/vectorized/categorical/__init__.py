# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorized update functions for categorical state node layers."""

from .prediction import vectorized_categorical_prediction
from .prediction_error import vectorized_categorical_prediction_error

__all__ = [
    "vectorized_categorical_prediction",
    "vectorized_categorical_prediction_error",
]
