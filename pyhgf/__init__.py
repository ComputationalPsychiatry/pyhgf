# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pkgutil
from io import BytesIO
from typing import Tuple, Union

import numpy as np
import pandas as pd

__version__ = "0.2.2"


def load_data(dataset: str) -> Union[Tuple[np.ndarray, ...], np.ndarray]:
    """Load dataset for continuous or binary HGF.

    Parameters
    ----------
    dataset : str
        The type of data to load. Can be `"continous"` or `"binary"`.

    Returns
    -------
    data : np.ndarray
        The data (a 1d timeseries).

    Notes
    -----
    The continuous time series is the standard USD-CHF conversion rates over time used
    in the Matlab examples.

    The binary dataset is from Iglesias et al. (2013) [#] (see the full dataset
    `here <https://www.research-collection.ethz.ch/handle/20.500.11850/454711)>`_. The
    binary set consist of one vector *u*, the observations, and one vector *y*, the
    decisions.

    References
    ----------
    .. [#] Iglesias, S., Kasper, L., Harrison, S. J., Manka, R., Mathys, C., & Stephan,
      K. E. (2021). Cholinergic and dopaminergic effects on prediction error and
      uncertainty responses during sensory associative learning. In NeuroImage (Vol.
      226, p. 117590). Elsevier BV. https://doi.org/10.1016/j.neuroimage.2020.117590

    """
    if dataset == "continuous":
        data = pd.read_csv(
            BytesIO(pkgutil.get_data(__name__, "data/usdchf.txt")),  # type: ignore
            names=["x"],
        ).x.to_numpy()
    elif dataset == "binary":
        u = pd.read_csv(
            BytesIO(
                pkgutil.get_data(__name__, "data/binary_input.txt")  # type: ignore
            ),
            names=["x"],
        ).x.to_numpy()
        y = pd.read_csv(
            BytesIO(
                pkgutil.get_data(__name__, "data/binary_response.txt")  # type: ignore
            ),
            names=["x"],
        ).x.to_numpy()
        data = (u, y)
    else:
        raise ValueError("Invalid dataset argument. Should be 'continous' or 'binary'.")

    return data
