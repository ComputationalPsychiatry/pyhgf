# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Vectorised update functions for categorical state node layers."""

from .prediction import vectorised_categorical_prediction
from .prediction_error import vectorised_categorical_prediction_error

__all__ = [
    "vectorised_categorical_prediction",
    "vectorised_categorical_prediction_error",
]
