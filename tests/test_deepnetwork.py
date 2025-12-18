# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np

from pyhgf.model import DeepNetwork


def test_fit():
    """Test the fit function on a deep network."""
    # test fixed learning rate
    fixed_learning_net = (
        DeepNetwork()
        .add_nodes(kind="continuous-state", n_nodes=4, precision=5.0)
        .add_layer(size=3, precision=1.0, tonic_volatility=-1.0)
        .add_layer(size=2, precision=0.5, tonic_volatility=-2.0)
    )

    fixed_learning_net.fit(x=np.ones((10, 4)), y=np.ones((10, 2)), lr=0.001)

    # test dynamic learning rate
    dynamic_learning_net = (
        DeepNetwork()
        .add_nodes(kind="continuous-state", n_nodes=4, precision=5.0)
        .add_layer(size=3, precision=1.0, tonic_volatility=-1.0)
        .add_layer(size=2, precision=0.5, tonic_volatility=-2.0)
    )

    dynamic_learning_net.fit(x=np.ones((10, 4)), y=np.ones((10, 2)), lr="dynamic")


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
