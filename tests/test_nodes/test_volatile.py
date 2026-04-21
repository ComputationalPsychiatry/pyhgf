# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork


def _assert_value_level_match(net_a, node_a, net_b, node_b, label="", rtol=1e-5):
    """Assert value-level trajectories match between two networks."""
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.allclose(
            net_a.node_trajectories[node_a][key],
            net_b.node_trajectories[node_b][key],
            rtol=rtol,
        ), f"{label}: Value-level key '{key}' mismatch"


def _assert_vol_level_match(volatile_net, vol_node, explicit_net, exp_node, label=""):
    """Assert volatility-level trajectories match (volatile _vol vs explicit node)."""
    vol_key_map = {
        "mean_vol": "mean",
        "expected_mean_vol": "expected_mean",
        "precision_vol": "precision",
        "expected_precision_vol": "expected_precision",
    }
    for vol_key, explicit_key in vol_key_map.items():
        assert np.allclose(
            volatile_net.node_trajectories[vol_node][vol_key],
            explicit_net.node_trajectories[exp_node][explicit_key],
        ), f"{label}: Volatility-level key '{vol_key}' vs '{explicit_key}' mismatch"


def _build_volatile(cls, update_type, timeseries):
    """Build a volatile-state network with autoconnection=1.0 for equivalence testing."""
    return (
        cls(update_type=update_type)
        .add_nodes()
        .add_nodes(
            kind="volatile-state",
            value_children=0,
            autoconnection_strength=1.0,
        )
        .input_data(input_data=timeseries)
    )


def _build_explicit(cls, update_type, timeseries):
    """Build explicit continuous + volatility-parent network."""
    return (
        cls(update_type=update_type)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )


def _run_volatile_vs_explicit(update_type):
    """Test that volatile-state is equivalent to explicit continuous+vol-parent pair.

    Both the Python and Rust backends satisfy this equivalence: the value-level
    posterior update runs before the volatility-level prediction-error step in both.
    """
    timeseries = load_data("continuous")

    for cls, label in [
        (PyNetwork, f"{update_type} py"),
        (RsNetwork, f"{update_type} rs"),
    ]:
        vol = _build_volatile(cls, update_type, timeseries)
        exp = _build_explicit(cls, update_type, timeseries)

        _assert_value_level_match(vol, 0, exp, 0, f"{label} input")
        _assert_value_level_match(vol, 1, exp, 1, label)
        _assert_vol_level_match(vol, 1, exp, 2, label)


def _run_explicit_cross_backend(update_type):
    """Test that Python and Rust produce the same trajectories for explicit networks."""
    timeseries = load_data("continuous")

    exp_py = _build_explicit(PyNetwork, update_type, timeseries)
    exp_rs = _build_explicit(RsNetwork, update_type, timeseries)

    label = f"{update_type} py vs rs"
    _assert_value_level_match(exp_py, 0, exp_rs, 0, f"{label} input", rtol=1e-4)
    _assert_value_level_match(exp_py, 1, exp_rs, 1, label, rtol=1e-4)


def test_volatile_node_matches_explicit_volatility_parent():
    """Test Rust volatile-state against explicit continuous+vol-parent (standard)."""
    _run_volatile_vs_explicit("standard")


def test_volatile_node_ehgf_matches_explicit():
    """Test Rust volatile-state against explicit continuous+vol-parent (eHGF)."""
    _run_volatile_vs_explicit("eHGF")


def test_volatile_node_unbounded_matches_explicit():
    """Test Rust volatile-state against explicit continuous+vol-parent (unbounded)."""
    _run_volatile_vs_explicit("unbounded")


def test_explicit_cross_backend_standard():
    """Test Python and Rust explicit networks agree (standard update)."""
    _run_explicit_cross_backend("standard")


def test_explicit_cross_backend_ehgf():
    """Test Python and Rust explicit networks agree (eHGF update)."""
    _run_explicit_cross_backend("eHGF")


def test_explicit_cross_backend_unbounded():
    """Test Python and Rust explicit networks agree (unbounded update)."""
    _run_explicit_cross_backend("unbounded")
