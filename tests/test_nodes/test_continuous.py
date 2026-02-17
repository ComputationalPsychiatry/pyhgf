# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork


def test_continuous_2_levels():
    """Test the 2-level continuous HGF: input node → value parent."""
    timeseries = load_data("continuous")

    # Rust -----------------------------------------------------------------------------
    rs_network = RsNetwork()
    rs_network.add_nodes()
    rs_network.add_nodes(value_children=0)
    rs_network.set_update_sequence()
    rs_network.input_data(timeseries)

    # Python ---------------------------------------------------------------------------
    py_network = (
        PyNetwork()
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data(input_data=timeseries)
    )

    # Ensure identical results for all nodes
    for node_idx in range(2):
        for key in ["mean", "expected_mean", "precision", "expected_precision"]:
            assert np.isclose(
                py_network.node_trajectories[node_idx][key],
                rs_network.node_trajectories[node_idx][key],
            ).all(), f"Node {node_idx}, key '{key}' mismatch"


def test_continuous_3_levels():
    """Test the 3-level continuous HGF: input → value parent + volatility parent."""
    timeseries = load_data("continuous")

    # Rust -----------------------------------------------------------------------------
    rs_network = RsNetwork()
    rs_network.add_nodes()
    rs_network.add_nodes(value_children=0)
    rs_network.add_nodes(volatility_children=0)
    rs_network.set_update_sequence()
    rs_network.input_data(timeseries)

    # Python ---------------------------------------------------------------------------
    py_network = (
        PyNetwork()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
        .input_data(input_data=timeseries)
    )

    # Ensure identical results for all nodes
    for node_idx in range(3):
        for key in ["mean", "expected_mean", "precision", "expected_precision"]:
            assert np.isclose(
                py_network.node_trajectories[node_idx][key],
                rs_network.node_trajectories[node_idx][key],
            ).all(), f"Node {node_idx}, key '{key}' mismatch"
