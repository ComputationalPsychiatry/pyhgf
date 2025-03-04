# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
from pytest import raises

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.typing import AdjacencyLists
from pyhgf.utils import add_parent, list_branches, remove_node


def test_imports():
    """Test the data import function"""
    _ = load_data("continuous")
    _, _ = load_data("binary")

    with raises(Exception):
        load_data("error")


def test_add_edges():
    """Test the add_edges function."""

    # add value coupling
    network = Network().add_nodes(n_nodes=3)
    network.add_edges(parent_idxs=1, children_idxs=0, coupling_strengths=1.0)
    network.add_edges(parent_idxs=1, children_idxs=2, coupling_strengths=1.0)

    # add volatility coupling
    network = Network().add_nodes(n_nodes=3)
    network.add_edges(
        kind="volatility", parent_idxs=1, children_idxs=0, coupling_strengths=1
    )
    network.add_edges(
        kind="volatility", parent_idxs=1, children_idxs=2, coupling_strengths=1
    )

    # expected error for invalid type
    with raises(Exception):
        network.add_edges(kind="error")


def test_find_branch():
    """Test the find_branch function."""
    edges = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (None,)),
        AdjacencyLists(2, (4,), None, None, None, (None,)),
        AdjacencyLists(2, None, None, (3,), None, (None,)),
    )
    branch_list = list_branches([0], edges, branch_list=[])
    assert branch_list == [0, 1, 2]


def test_set_update_sequence():
    """Test the set_update_sequence function."""

    # a standard binary HGF
    network1 = (
        Network()
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0)
        .create_belief_propagation_fn()
    )

    predictions, updates = network1.update_sequence
    assert len(predictions) == 2
    assert len(updates) == 2

    # a standard continuous HGF
    network2 = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .create_belief_propagation_fn(update_type="standard")
    )
    predictions, updates = network2.update_sequence
    assert len(predictions) == 3
    assert len(updates) == 4

    # an EF state node
    network3 = Network().add_nodes(kind="ef-state").create_belief_propagation_fn()
    predictions, updates = network3.update_sequence
    assert len(predictions) == 0
    assert len(updates) == 1

    # a Dirichlet node
    network4 = (
        Network()
        .add_nodes(kind="dp-state", alpha=0.1, batch_size=2)
        .add_nodes(
            kind="ef-state",
            n_nodes=2,
            value_children=0,
            xis=jnp.array([0.0, 1.0]),
            nus=15.0,
        )
        .create_belief_propagation_fn()
    )
    predictions, updates = network4.update_sequence
    assert len(predictions) == 1
    assert len(updates) == 3


def test_add_parent():
    """Test the add_parent function."""
    network = (
        Network()
        .add_nodes(n_nodes=4)
        .add_nodes(value_children=2)
        .add_nodes(value_children=3)
    )
    attributes, edges, _ = network.get_network()
    new_attributes, new_edges = add_parent(attributes, edges, 1, "volatility", 1.0)

    assert len(new_attributes) == 8
    assert len(new_edges) == 7

    new_attributes, new_edges = add_parent(attributes, edges, 1, "value", 1.0)

    assert len(new_attributes) == 8
    assert len(new_edges) == 7


def test_remove_node():
    """Test the remove_node function."""
    network = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0, volatility_children=1)
        .add_nodes(volatility_children=2)
        .add_nodes(value_children=2)
    )

    attributes, edges, _ = network.get_network()
    new_attributes, new_edges = remove_node(attributes, edges, 2)

    assert len(new_attributes) == 5
    assert len(new_edges) == 4


def test_scan_sampling():
    """Test the scan_sampling function."""
    from pyhgf.model import scan_sampling

    # Mock attributes for a single node
    attributes = {0: {"expected_mean": 0.5, "expected_precision": 10.0}}
    rng_key = jnp.array([0, 1])
    node_idx = 0
    model_type = 2  # Continuous
    num_samples = 10

    samples = scan_sampling(attributes, node_idx, rng_key, model_type, num_samples)

    assert samples.shape == (num_samples,), "Sample size should match num_samples."
    assert jnp.all(
        jnp.isfinite(samples)
    ), "Samples should be finite and not include NaNs or infinities."


def test_handle_observation():
    """Test the handle_observation function."""
    from pyhgf.model import handle_observation

    # Mock attributes and inputs
    attributes = {0: {"expected_mean": 0.5, "expected_precision": 10.0}}
    rng_key = jnp.array([0, 1])
    node_idx = 0
    sophisticated = True
    edges = {0: {"node_type": 2}}  # Continuous

    updated_attributes, new_rng_key = handle_observation(
        attributes, node_idx, rng_key, sophisticated, edges
    )

    assert 0 in updated_attributes, "Node index should exist in updated attributes."
    assert (
        updated_attributes[0]["observed"] == 1
    ), "Observation should be set when sophisticated=True."
    assert (
        new_rng_key is not None
    ), "Random key should be updated after handling observation."
