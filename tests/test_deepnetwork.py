# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf.model import DeepNetwork

NETWORK_CLASSES = [DeepNetwork, RsNetwork]


def test_fit():
    """Test that Python and Rust backends produce identical fit results.

    Network: 2 targets → 3 hidden (volatile-state) → 4 predictors (volatile-state).
    Linear coupling at all levels.  Both fixed and dynamic learning rates.
    """
    n_targets, n_hidden, n_predictors = 2, 3, 4
    n_nodes = n_targets + n_hidden + n_predictors

    targets = list(range(n_targets))
    hidden = list(range(n_targets, n_targets + n_hidden))
    predictors = list(range(n_targets + n_hidden, n_nodes))

    np.random.seed(42)
    x = np.random.randn(5, n_predictors)
    y = np.random.randn(5, n_targets)

    for lr in [0.1, "dynamic"]:
        lr_label = f"lr={lr}"

        fitted = []
        for Network in NETWORK_CLASSES:
            net = (
                Network()
                .add_nodes(kind="continuous-state", n_nodes=n_targets)
                .add_layer(size=n_hidden)
                .add_layer(size=n_predictors)
            )
            net.fit(
                x=x,
                y=y,
                inputs_x_idxs=tuple(predictors),
                inputs_y_idxs=tuple(targets),
                lr=lr,
            )
            fitted.append(net)

        py_net, rs_net = fitted

        # ----- Compare mean and precision at every node -----
        for node_idx in range(n_nodes):
            for key in ["mean", "precision"]:
                py_val = np.asarray(py_net.node_trajectories[node_idx][key])
                rs_val = np.asarray(rs_net.node_trajectories[node_idx][key])
                assert np.allclose(py_val, rs_val, atol=1e-3), (
                    f"{lr_label}: node {node_idx} '{key}' mismatch\n"
                    f"  Py={py_val}\n  Rs={rs_val}"
                )

        # ----- Compare volatile-level mean/precision on hidden & predictor nodes -----
        for node_idx in hidden + predictors:
            for key in ["mean_vol", "precision_vol"]:
                py_val = np.asarray(py_net.node_trajectories[node_idx][key])
                rs_val = np.asarray(rs_net.node_trajectories[node_idx][key])
                assert np.allclose(py_val, rs_val, atol=1e-3), (
                    f"{lr_label}: node {node_idx} '{key}' mismatch\n"
                    f"  Py={py_val}\n  Rs={rs_val}"
                )

        # ----- Compare coupling weights (value_coupling_children) -----
        for node_idx in hidden + predictors:
            py_w = np.asarray(
                py_net.node_trajectories[node_idx]["value_coupling_children"]
            )
            rs_w = np.asarray(
                rs_net.node_trajectories[node_idx]["value_coupling_children"]
            )
            assert np.allclose(py_w, rs_w, atol=1e-3), (
                f"{lr_label}: node {node_idx} 'value_coupling_children' mismatch\n"
                f"  Py={py_w}\n  Rs={rs_w}"
            )


def test_deepnetwork_add_value_parent_layer():
    """Test building a fully connected parent layer."""
    net = DeepNetwork()

    # Create 4 bottom nodes
    net = net.add_nodes(kind="continuous-state", n_nodes=4, precision=1.0)
    bottom = list(range(4))

    # Add one parent layer of size 3
    n_nodes_before = net.n_nodes
    net = net.add_layer(
        size=3,
        value_children=bottom,
        precision=1.0,
        tonic_volatility=-1.0,
        autoconnection_strength=0.2,
    )

    # Expect exactly 3 new nodes
    assert net.n_nodes == n_nodes_before + 3

    # Get the indices of the newly added parents
    parents = list(range(n_nodes_before, net.n_nodes))

    # For each parent, check fully-connected structure
    for p in parents:
        assert net.edges[p].value_children == tuple(bottom)
        assert len(net.attributes[p]["value_coupling_children"]) == len(bottom)

    # Check that layer was tracked
    assert len(net.layers) == 2  # base layer + added layer
    assert net.layers[1] == parents


def test_deepnetwork_add_layer_stack():
    """Test building a multi-layer stack."""
    net = DeepNetwork()

    # Base layer of 4 nodes
    net = net.add_nodes(kind="continuous-state", n_nodes=4, precision=1.0)
    bottom = list(range(4))

    # Build 3 → 2 → 1 parent stack
    n_nodes_before = net.n_nodes
    net = net.add_layer_stack(
        value_children=bottom,
        layer_sizes=[3, 2, 1],
        precision=1.0,
        tonic_volatility=-1.0,
        autoconnection_strength=0.3,
    )

    # Check total nodes added (3 + 2 + 1 = 6)
    assert net.n_nodes == n_nodes_before + 6

    # Check that layers were tracked automatically
    assert len(net.layers) == 4  # base + 3 added layers

    # Get tracked layers (excluding base layer at index 0)
    layers = net.layers[1:]  # Skip base layer

    # Check layer sizes
    assert len(layers[0]) == 3
    assert len(layers[1]) == 2
    assert len(layers[2]) == 1

    # Check connections are fully dense
    # Layer 0 → bottom
    for p in layers[0]:
        assert net.edges[p].value_children == tuple(bottom)

    # Layer 1 → layer 0
    for p in layers[1]:
        assert net.edges[p].value_children == tuple(layers[0])

    # Layer 2 → layer 1
    for p in layers[2]:
        assert net.edges[p].value_children == tuple(layers[1])
