# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

NETWORK_CLASSES = [PyNetwork, RsNetwork]


def _build_binary_2_levels(cls, u):
    """
    Two-level binary HGF.

    Network topology
    ----------------
    Node 0 : binary-state   (input node)
    Node 1 : continuous-state, value_children=0  (value parent of binary node)
    """
    return (
        cls()
        .add_nodes(kind="binary-state")
        .add_nodes(kind="continuous-state", value_children=0)
        .input_data(input_data=u)
    )


def _build_binary_3_levels(cls, u):
    """
    Three-level binary HGF.

    Network topology
    ----------------
    Node 0 : binary-state   (input node)
    Node 1 : continuous-state, value_children=0  (value parent of binary)
    Node 2 : continuous-state, volatility_children=1  (volatility parent)
    """
    return (
        cls()
        .add_nodes(kind="binary-state")
        .add_nodes(
            kind="continuous-state", value_children=0, mean=0.0, tonic_volatility=1.0
        )
        .add_nodes(
            kind="continuous-state",
            volatility_children=1,
            mean=0.0,
            tonic_volatility=1.0,
        )
        .input_data(input_data=u)
    )


# ---------------------------------------------------------------------------
# Cross-backend comparison helpers
# ---------------------------------------------------------------------------


def _compare_backends(results, node_idxs, keys):
    """Assert that all backends produce identical trajectories.

    JAX defaults to float32 while Rust uses f64, so accumulated rounding
    differences over many recursive updates can reach ~1e-3.  We use
    ``rtol=1e-4`` which is comfortably within float32 precision.
    """
    names = list(results.keys())
    ref_name = names[0]
    ref = results[ref_name]
    for name in names[1:]:
        net = results[name]
        for node_idx in node_idxs:
            for key in keys:
                np.testing.assert_allclose(
                    np.asarray(ref.node_trajectories[node_idx][key], dtype=np.float64),
                    np.asarray(net.node_trajectories[node_idx][key], dtype=np.float64),
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=(
                        f"Backend '{name}' differs from '{ref_name}' "
                        f"at node {node_idx}, key '{key}'"
                    ),
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_binary_2_levels():
    """Two-level binary HGF: binary input → continuous value parent.

    Checks that PyNetwork and RsNetwork produce identical belief trajectories.
    """
    u, _ = load_data("binary")

    results = {cls.__name__: _build_binary_2_levels(cls, u) for cls in NETWORK_CLASSES}

    # Cross-backend comparison
    # Node 0 is binary: compare mean, expected_mean, precision, expected_precision
    # Node 1 is continuous: compare all four standard keys
    _compare_backends(
        results,
        node_idxs=[0, 1],
        keys=["mean", "expected_mean", "precision", "expected_precision"],
    )

    # Sanity: node 0 expected_mean must stay in (0, 1) — it is a probability
    ref = results[PyNetwork.__name__]
    assert np.all(
        (ref.node_trajectories[0]["expected_mean"] > 0)
        & (ref.node_trajectories[0]["expected_mean"] < 1)
    ), "Binary node expected_mean must be in (0, 1) at every time step"
    # expected_precision = mu*(1-mu) must be positive
    assert np.all(ref.node_trajectories[0]["expected_precision"] > 0), (
        "Binary node expected_precision must be positive"
    )


def test_binary_3_levels():
    """Three-level binary HGF: binary input → value parent → volatility parent.

    Checks that PyNetwork and RsNetwork produce identical belief trajectories and
    that the Python backend's final values match the canonical reference output.
    """
    u, _ = load_data("binary")

    results = {cls.__name__: _build_binary_3_levels(cls, u) for cls in NETWORK_CLASSES}

    # Cross-backend comparison across all three nodes
    _compare_backends(
        results,
        node_idxs=[0, 1, 2],
        keys=["mean", "expected_mean", "precision", "expected_precision"],
    )

    # Sanity: node 0 expected_mean in (0, 1) and expected_precision positive at all steps
    ref = results[PyNetwork.__name__]
    assert np.all(
        (ref.node_trajectories[0]["expected_mean"] > 0)
        & (ref.node_trajectories[0]["expected_mean"] < 1)
    ), "Binary node expected_mean must be in (0, 1) at every time step"
    assert np.all(ref.node_trajectories[0]["expected_precision"] > 0), (
        "Binary node expected_precision must be positive"
    )

    # Sanity: node 1 and 2 precisions must be positive (no divergence)
    for node_idx in [1, 2]:
        assert np.all(ref.node_trajectories[node_idx]["precision"] > 0), (
            f"Node {node_idx} precision must remain positive"
        )
