# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

NETWORK_CLASSES = [PyNetwork, RsNetwork]


def test_continuous_2_levels():
    """Test the 2-level continuous HGF: input node → value parent."""
    timeseries = load_data("continuous")

    results = {}
    for cls in NETWORK_CLASSES:
        results[cls.__name__] = (
            cls()
            .add_nodes()
            .add_nodes(value_children=0)
            .input_data(input_data=timeseries)
        )

    # Ensure identical results across implementations
    ref = results[NETWORK_CLASSES[0].__name__]
    for name, net in results.items():
        if net is ref:
            continue
        for node_idx in range(2):
            for key in ["mean", "expected_mean", "precision", "expected_precision"]:
                assert np.isclose(
                    ref.node_trajectories[node_idx][key],
                    net.node_trajectories[node_idx][key],
                ).all(), f"{name} node {node_idx}, key '{key}' mismatch"


def test_continuous_3_levels():
    """Test the 3-level continuous HGF: input → value parent + volatility parent."""
    timeseries = load_data("continuous")

    results = {}
    for cls in NETWORK_CLASSES:
        results[cls.__name__] = (
            cls()
            .add_nodes()
            .add_nodes(value_children=0)
            .add_nodes(volatility_children=0)
            .input_data(input_data=timeseries)
        )

    # Ensure identical results across implementations
    ref = results[NETWORK_CLASSES[0].__name__]
    for name, net in results.items():
        if net is ref:
            continue
        for node_idx in range(3):
            for key in ["mean", "expected_mean", "precision", "expected_precision"]:
                assert np.isclose(
                    ref.node_trajectories[node_idx][key],
                    net.node_trajectories[node_idx][key],
                ).all(), f"{name} node {node_idx}, key '{key}' mismatch"
