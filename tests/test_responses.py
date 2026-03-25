# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np

from pyhgf import load_data
from pyhgf.model import HGF
from pyhgf.response import binary_softmax, binary_softmax_inverse_temperature


def test_binary_responses():
    """Test the binary responses."""
    u, y = load_data("binary")

    # two-level binary HGF
    # --------------------
    two_level_binary_hgf = HGF(
        n_levels=2,
        model_type="binary",
        initial_mean={"1": 0.5, "2": 0.0},
        initial_precision={"1": 0.0, "2": 1.0},
        tonic_volatility={"2": -6.0},
    ).input_data(input_data=u)

    # binary sofmax
    # -------------
    surprise = two_level_binary_hgf.surprise(
        response_function=binary_softmax, response_function_inputs=y
    )
    assert np.isclose(surprise.sum(), 195.81573)

    # binary sofmax with inverse temperature
    # --------------------------------------
    surprise = two_level_binary_hgf.surprise(
        response_function=binary_softmax_inverse_temperature,
        response_function_inputs=y,
        response_function_parameters=2.0,
    )
    assert np.isclose(surprise.sum(), 188.77818)
