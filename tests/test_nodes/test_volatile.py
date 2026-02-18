# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network


def _assert_value_level_match(net_a, node_a, net_b, node_b, label=""):
    """Assert value-level trajectories match between two networks."""
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.isclose(
            net_a.node_trajectories[node_a][key],
            net_b.node_trajectories[node_b][key],
        ).all(), f"{label}: Value-level key '{key}' mismatch"


def _assert_vol_level_match(volatile_net, vol_node, explicit_net, exp_node, label=""):
    """Assert volatility-level trajectories match (volatile _vol vs explicit node)."""
    vol_key_map = {
        "mean_vol": "mean",
        "expected_mean_vol": "expected_mean",
        "precision_vol": "precision",
        "expected_precision_vol": "expected_precision",
    }
    for vol_key, explicit_key in vol_key_map.items():
        assert np.isclose(
            volatile_net.node_trajectories[vol_node][vol_key],
            explicit_net.node_trajectories[exp_node][explicit_key],
        ).all(), (
            f"{label}: Volatility-level key '{vol_key}' vs '{explicit_key}' mismatch"
        )


def test_volatile_node_matches_explicit_volatility_parent():
    """Test volatile nodes against continuous pairs (standard update).

    Four networks are compared:
    - Python volatile vs Python explicit
    - Rust volatile vs Rust explicit
    - Python volatile vs Rust volatile
    """
    timeseries = load_data("continuous")

    # Python volatile ------------------------------------------------------------------
    py_volatile = (
        Network(update_type="standard")
        .add_nodes()
        .add_nodes(kind="volatile-node", value_children=0)
        .input_data(input_data=timeseries)
    )

    # Python explicit ------------------------------------------------------------------
    py_explicit = (
        Network(update_type="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )

    # Rust volatile --------------------------------------------------------------------
    rs_volatile = RsNetwork("standard")
    rs_volatile.add_nodes()
    rs_volatile.add_nodes(kind="volatile-state", value_children=0)
    rs_volatile.set_update_sequence()
    rs_volatile.input_data(timeseries.tolist())

    # Rust explicit --------------------------------------------------------------------
    rs_explicit = RsNetwork("standard")
    rs_explicit.add_nodes()
    rs_explicit.add_nodes(value_children=0)
    rs_explicit.add_nodes(volatility_children=1)
    rs_explicit.set_update_sequence()
    rs_explicit.input_data(timeseries.tolist())

    # Python volatile vs Python explicit
    _assert_value_level_match(py_volatile, 1, py_explicit, 1, "Py vol vs Py exp")
    _assert_vol_level_match(py_volatile, 1, py_explicit, 2, "Py vol vs Py exp")
    _assert_value_level_match(py_volatile, 0, py_explicit, 0, "Py input")

    # Rust volatile vs Rust explicit
    _assert_value_level_match(rs_volatile, 1, rs_explicit, 1, "Rs vol vs Rs exp")
    _assert_vol_level_match(rs_volatile, 1, rs_explicit, 2, "Rs vol vs Rs exp")
    _assert_value_level_match(rs_volatile, 0, rs_explicit, 0, "Rs input")

    # Python volatile vs Rust volatile
    _assert_value_level_match(py_volatile, 1, rs_volatile, 1, "Py vol vs Rs vol")
    _assert_value_level_match(py_volatile, 0, rs_volatile, 0, "Py vs Rs input")


def test_volatile_node_ehgf_matches_explicit():
    """Test volatile node with eHGF update matches explicit network with eHGF.

    Four networks are compared:
    - Python volatile vs Python explicit
    - Rust volatile vs Rust explicit
    - Python volatile vs Rust volatile
    """
    timeseries = load_data("continuous")

    # Python volatile ------------------------------------------------------------------
    py_volatile = (
        Network(update_type="eHGF")
        .add_nodes()
        .add_nodes(kind="volatile-node", value_children=0)
        .input_data(input_data=timeseries)
    )

    # Python explicit ------------------------------------------------------------------
    py_explicit = (
        Network(update_type="eHGF")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )

    # Rust volatile --------------------------------------------------------------------
    rs_volatile = RsNetwork("eHGF")
    rs_volatile.add_nodes()
    rs_volatile.add_nodes(kind="volatile-state", value_children=0)
    rs_volatile.set_update_sequence()
    rs_volatile.input_data(timeseries.tolist())

    # Rust explicit --------------------------------------------------------------------
    rs_explicit = RsNetwork("eHGF")
    rs_explicit.add_nodes()
    rs_explicit.add_nodes(value_children=0)
    rs_explicit.add_nodes(volatility_children=1)
    rs_explicit.set_update_sequence()
    rs_explicit.input_data(timeseries.tolist())

    # Python volatile vs Python explicit
    _assert_value_level_match(py_volatile, 1, py_explicit, 1, "eHGF Py vol vs Py exp")
    _assert_vol_level_match(py_volatile, 1, py_explicit, 2, "eHGF Py vol vs Py exp")

    # Rust volatile vs Rust explicit
    _assert_value_level_match(rs_volatile, 1, rs_explicit, 1, "eHGF Rs vol vs Rs exp")
    _assert_vol_level_match(rs_volatile, 1, rs_explicit, 2, "eHGF Rs vol vs Rs exp")

    # Python volatile vs Rust volatile
    _assert_value_level_match(py_volatile, 1, rs_volatile, 1, "eHGF Py vol vs Rs vol")
    _assert_value_level_match(py_volatile, 0, rs_volatile, 0, "eHGF Py vs Rs input")


def test_volatile_node_unbounded_matches_explicit():
    """Test volatile node with unbounded update matches explicit network.

    Four networks are compared:
    - Python volatile vs Python explicit
    - Rust volatile vs Rust explicit
    - Python volatile vs Rust volatile
    """
    timeseries = load_data("continuous")

    # Python volatile ------------------------------------------------------------------
    py_volatile = (
        Network(update_type="unbounded")
        .add_nodes()
        .add_nodes(kind="volatile-node", value_children=0)
        .input_data(input_data=timeseries)
    )

    # Python explicit ------------------------------------------------------------------
    py_explicit = (
        Network(update_type="unbounded")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )

    # Rust volatile --------------------------------------------------------------------
    rs_volatile = RsNetwork("unbounded")
    rs_volatile.add_nodes()
    rs_volatile.add_nodes(kind="volatile-state", value_children=0)
    rs_volatile.set_update_sequence()
    rs_volatile.input_data(timeseries.tolist())

    # Rust explicit --------------------------------------------------------------------
    rs_explicit = RsNetwork("unbounded")
    rs_explicit.add_nodes()
    rs_explicit.add_nodes(value_children=0)
    rs_explicit.add_nodes(volatility_children=1)
    rs_explicit.set_update_sequence()
    rs_explicit.input_data(timeseries.tolist())

    # Python volatile vs Python explicit
    _assert_value_level_match(
        py_volatile, 1, py_explicit, 1, "Unbounded Py vol vs Py exp"
    )
    _assert_vol_level_match(
        py_volatile, 1, py_explicit, 2, "Unbounded Py vol vs Py exp"
    )

    # Rust volatile vs Rust explicit
    _assert_value_level_match(
        rs_volatile, 1, rs_explicit, 1, "Unbounded Rs vol vs Rs exp"
    )
    _assert_vol_level_match(
        rs_volatile, 1, rs_explicit, 2, "Unbounded Rs vol vs Rs exp"
    )

    # Python volatile vs Rust volatile
    _assert_value_level_match(
        py_volatile, 1, rs_volatile, 1, "Unbounded Py vol vs Rs vol"
    )
    _assert_value_level_match(
        py_volatile, 0, rs_volatile, 0, "Unbounded Py vs Rs input"
    )
