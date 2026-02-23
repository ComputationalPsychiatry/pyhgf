# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

NETWORK_CLASSES = [PyNetwork, RsNetwork]


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


def _run_volatile_vs_explicit(update_type):
    """Build volatile and explicit networks for each class, then cross-compare."""
    timeseries = load_data("continuous")

    volatile = {}
    explicit = {}
    for cls in NETWORK_CLASSES:
        name = cls.__name__
        volatile[name] = (
            cls(update_type=update_type)
            .add_nodes()
            .add_nodes(kind="volatile-state", value_children=0)
            .input_data(input_data=timeseries)
        )
        explicit[name] = (
            cls(update_type=update_type)
            .add_nodes()
            .add_nodes(value_children=0)
            .add_nodes(volatility_children=1)
            .input_data(input_data=timeseries)
        )

    # For each implementation: volatile must match explicit
    for name in volatile:
        label = f"{update_type} {name}"
        _assert_value_level_match(
            volatile[name], 0, explicit[name], 0, f"{label} input"
        )
        _assert_value_level_match(volatile[name], 1, explicit[name], 1, label)
        _assert_vol_level_match(volatile[name], 1, explicit[name], 2, label)

    # Cross-implementation: volatile networks must agree
    ref_name = NETWORK_CLASSES[0].__name__
    for name in volatile:
        if name == ref_name:
            continue
        label = f"{update_type} {ref_name} vs {name}"
        _assert_value_level_match(
            volatile[ref_name], 0, volatile[name], 0, f"{label} input"
        )
        _assert_value_level_match(volatile[ref_name], 1, volatile[name], 1, label)


def test_volatile_node_matches_explicit_volatility_parent():
    """Test volatile nodes against continuous pairs (standard update)."""
    _run_volatile_vs_explicit("standard")


def test_volatile_node_ehgf_matches_explicit():
    """Test volatile node with eHGF update matches explicit network."""
    _run_volatile_vs_explicit("eHGF")


def test_volatile_node_unbounded_matches_explicit():
    """Test volatile node with unbounded update matches explicit network."""
    _run_volatile_vs_explicit("unbounded")
