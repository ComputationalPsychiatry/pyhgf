# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized update functions for binary state node layers."""

from .prediction import vectorized_binary_prediction
from .prediction_error import vectorized_binary_prediction_error

__all__ = [
    "vectorized_binary_prediction",
    "vectorized_binary_prediction_error",
]
