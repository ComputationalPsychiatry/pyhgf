# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.response import binary_softmax, binary_softmax_inverse_temperature


def test_binary_responses():
    """Test the binary responses."""
    u, y = load_data("binary")

    # two-level binary HGF
    # --------------------
    two_level_binary_hgf = (
        Network()
        .add_nodes(
            kind="binary-state",
            node_parameters={"mean": 0.5, "precision": 0.0},
        )
        .add_nodes(
            kind="continuous-state",
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 0.0,
                "precision": 1.0,
                "tonic_volatility": -6.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
        .input_data(input_data=u)
    )

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
