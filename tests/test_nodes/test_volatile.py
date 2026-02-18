# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np

from pyhgf import load_data
from pyhgf.model import Network


def test_volatile_node_matches_explicit_volatility_parent():
    """Test volatile nodes agains continuous pairs.

    Network A (volatile):
        node 0 (input) → node 1 (volatile-node, value parent of 0)

    Network B (explicit):
        node 0 (input) → node 1 (continuous, value parent of 0)
                          node 2 (continuous, volatility parent of 1)

    The volatile node's value level corresponds to node 1, and its implicit
    volatility level corresponds to node 2.
    """
    timeseries = load_data("continuous")

    # Network A: volatile node ---------------------------------------------------------
    # Use update_type="standard" so the volatile node's internal vol level uses
    # the same precision-first update order as the explicit volatility parent.
    volatile_network = (
        Network(update_type="standard")
        .add_nodes()  # node 0: input
        .add_nodes(kind="volatile-node", value_children=0)  # node 1: volatile parent
        .input_data(input_data=timeseries)
    )

    # Network B: explicit value parent + volatility parent -----------------------------
    # Match the volatile node defaults:
    #   value level:  tonic_volatility=-4.0 (default)
    #   vol level:    tonic_volatility_vol=-4.0 → node 2 tonic_volatility=-4.0 (default)
    # Use update_type="standard" so the volatility parent uses the same
    # precision-first update order as the volatile node's internal vol level.
    explicit_network = (
        Network(update_type="standard")
        .add_nodes()  # node 0: input
        .add_nodes(value_children=0)  # node 1: value parent
        .add_nodes(volatility_children=1)  # node 2: volatility parent of node 1
        .input_data(input_data=timeseries)
    )

    # Compare value-level trajectories (volatile node 1 vs explicit node 1) -----------
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.isclose(
            volatile_network.node_trajectories[1][key],
            explicit_network.node_trajectories[1][key],
        ).all(), (
            f"Value-level key '{key}' mismatch between volatile and explicit networks"
        )

    # Compare volatility-level trajectories
    # volatile node stores vol-level as *_vol; explicit network stores them in node 2
    vol_key_map = {
        "mean_vol": "mean",
        "expected_mean_vol": "expected_mean",
        "precision_vol": "precision",
        "expected_precision_vol": "expected_precision",
    }
    for vol_key, explicit_key in vol_key_map.items():
        assert np.isclose(
            volatile_network.node_trajectories[1][vol_key],
            explicit_network.node_trajectories[2][explicit_key],
        ).all(), (
            f"Volatility-level key '{vol_key}' vs '{explicit_key}' mismatch "
            f"between volatile and explicit networks"
        )

    # Compare input node trajectories (should also match) -----------------------------
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.isclose(
            volatile_network.node_trajectories[0][key],
            explicit_network.node_trajectories[0][key],
        ).all(), f"Input node key '{key}' mismatch"


def test_volatile_node_ehgf_matches_explicit():
    """Test volatile node with eHGF update matches explicit network with eHGF.

    Both networks should use update_type="eHGF" and produce matching trajectories.
    """
    timeseries = load_data("continuous")

    # Network A: volatile node with eHGF
    volatile_network = (
        Network(update_type="eHGF")
        .add_nodes()
        .add_nodes(kind="volatile-node", value_children=0)
        .input_data(input_data=timeseries)
    )

    # Network B: explicit with eHGF
    explicit_network = (
        Network(update_type="eHGF")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )

    # Compare value-level trajectories
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.isclose(
            volatile_network.node_trajectories[1][key],
            explicit_network.node_trajectories[1][key],
        ).all(), f"eHGF: Value-level key '{key}' mismatch"

    # Compare volatility-level trajectories
    vol_key_map = {
        "mean_vol": "mean",
        "expected_mean_vol": "expected_mean",
        "precision_vol": "precision",
        "expected_precision_vol": "expected_precision",
    }
    for vol_key, explicit_key in vol_key_map.items():
        assert np.isclose(
            volatile_network.node_trajectories[1][vol_key],
            explicit_network.node_trajectories[2][explicit_key],
        ).all(), f"eHGF: Volatility-level key '{vol_key}' vs '{explicit_key}' mismatch"


def test_volatile_node_unbounded_matches_explicit():
    """Test volatile node with unbounded update matches explicit network.

    Both networks should use update_type="unbounded" and produce matching trajectories.
    """
    timeseries = load_data("continuous")

    # Network A: volatile node with unbounded
    volatile_network = (
        Network(update_type="unbounded")
        .add_nodes()
        .add_nodes(kind="volatile-node", value_children=0)
        .input_data(input_data=timeseries)
    )

    # Network B: explicit with unbounded
    explicit_network = (
        Network(update_type="unbounded")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )

    # Compare value-level trajectories
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.isclose(
            volatile_network.node_trajectories[1][key],
            explicit_network.node_trajectories[1][key],
        ).all(), f"Unbounded: Value-level key '{key}' mismatch"

    # Compare volatility-level trajectories
    vol_key_map = {
        "mean_vol": "mean",
        "expected_mean_vol": "expected_mean",
        "precision_vol": "precision",
        "expected_precision_vol": "expected_precision",
    }
    for vol_key, explicit_key in vol_key_map.items():
        assert np.isclose(
            volatile_network.node_trajectories[1][vol_key],
            explicit_network.node_trajectories[2][explicit_key],
        ).all(), (
            f"Unbounded: Volatility-level key '{vol_key}' vs '{explicit_key}' mismatch"
        )
