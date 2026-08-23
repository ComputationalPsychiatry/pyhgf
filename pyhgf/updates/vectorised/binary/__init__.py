# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised update functions for binary state node layers."""

from .prediction import vectorised_binary_prediction
from .prediction_error import vectorised_binary_prediction_error

__all__ = [
    "vectorised_binary_prediction",
    "vectorised_binary_prediction_error",
]
