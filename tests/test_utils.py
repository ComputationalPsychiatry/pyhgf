# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
import jax.numpy as jnp
from jax import random
from pytest import raises

import pyhgf.updates.observation as obs
from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.typing import AdjacencyLists, Attributes, Edges, UpdateSequence
from pyhgf.utils import add_parent, list_branches, remove_node, scan_sampling
from pyhgf.utils.beliefs_propagation import beliefs_propagation
from pyhgf.utils.sample_node_distribution import sample_node_distribution


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

    # Mock attributes for a single node
    attributes = {0: {"expected_mean": 0.5, "expected_precision": 10.0}}
    rng_key = jnp.array([0, 1], dtype=jnp.uint32)
    node_idx = 0
    model_type = 2  # Continuous
    num_samples = 10

    samples = scan_sampling(attributes, node_idx, rng_key, model_type, num_samples)

    assert samples.shape == (num_samples,), "Sample size should match num_samples."
    assert jnp.all(
        jnp.isfinite(samples)
    ), "Samples should be finite and not include NaNs or infinities."


# --- Dummy functions and classes for testing ---


def dummy_prediction_update(attributes, node_idx, edges):
    """Dummy prediction function that adds a flag."""
    attributes[node_idx]["predicted"] = True
    return attributes


def dummy_update(attributes, node_idx, edges):
    """Dummy update function that adds a flag."""
    attributes[node_idx]["updated"] = True
    return attributes


# DummyEdge class to simulate an edges object with a node_type attribute.
class DummyEdge:
    def __init__(self, node_type):
        self.node_type = node_type


def dummy_set_observation(attributes, node_idx, values, observed):
    """Dummy set_observation function that updates the dictionary."""
    attributes[node_idx]["observation"] = values
    attributes[node_idx]["observed"] = observed
    return attributes


# --- Tests on sample_node_distribution ---


def test_sample_node_distribution_continuous():
    """Test sampling for a continuous model (model_type == 2)."""
    attributes = {
        0: {"expected_mean": 5.0, "expected_precision": 4.0}
    }  # sigma = 1/√4 = 0.5
    rng_key = random.PRNGKey(0)
    sample, new_key = sample_node_distribution(attributes, 0, rng_key, model_type=2)
    # Check that sample is indeed a scalar (float or jnp.float32)
    assert sample is not None
    assert 5.0 - 3.0 * 0.5 <= float(sample) <= 5.0 + 3.0 * 0.5


def test_sample_node_distribution_discrete():
    """Test sampling for a discrete model."""
    attributes = {
        0: {"expected_mean": 0.0, "expected_precision": 1.0}
    }  # values not used here
    rng_key = random.PRNGKey(0)
    sample, new_key = sample_node_distribution(attributes, 0, rng_key, model_type=1)
    # The result should be 0.0 or 1.0
    sample_val = float(sample)
    assert sample_val in [0.0, 1.0]


def test_beliefs_propagation_sophisticated(monkeypatch):
    """Test beliefs_propagation in sophisticated mode."""
    # Replace set_observation with our dummy
    monkeypatch.setattr(obs, "set_observation", dummy_set_observation)

    # Prepare the sequence of updates (prediction and update) as tuples.
    prediction_steps = ((0, dummy_prediction_update),)
    update_steps = ((0, dummy_update),)
    update_sequence: UpdateSequence = (prediction_steps, update_steps)

    # Prepare attributes:
    # - Key -1 is used to store the time_step.
    # - Node 0 has parameters for sampling.
    attributes: Attributes = {
        -1: {},
        0: {"expected_mean": 1.0, "expected_precision": 1.0},
    }

    # Define a simple hashable dictionary class.
    class FrozenDict(dict):
        def __hash__(self):
            return hash(tuple(sorted(self.items())))

    # Use FrozenDict to create a hashable version of edges.
    edges: Edges = FrozenDict({0: DummyEdge(node_type=2)})

    # In sophisticated mode, inputs are directly the time_step.
    time_step = 10
    inputs = time_step
    input_idxs = (0,)
    rng_key = random.PRNGKey(0)

    updated_attr, _ = beliefs_propagation(
        attributes,
        inputs,
        update_sequence,
        edges,
        input_idxs,
        sophisticated=True,
        rng_key=rng_key,
    )
    # Check that time_step has been assigned correctly.
    assert updated_attr[-1]["time_step"] == time_step
    # Use equality comparisons rather than identity comparisons.
    assert updated_attr[0]["predicted"] is True
    assert updated_attr[0]["updated"] is True
    # Check that handle_observation has updated the observation.
    assert updated_attr[0]["observed"] == 1


def test_beliefs_propagation_non_sophisticated(monkeypatch):
    """Test beliefs_propagation in non-sophisticated mode."""
    # Replace set_observation with our dummy via the module object.
    monkeypatch.setattr(obs, "set_observation", dummy_set_observation)

    # Use tuples for the update sequence.
    prediction_steps = ((0, dummy_prediction_update),)
    update_steps = ((0, dummy_update),)
    update_sequence: UpdateSequence = (prediction_steps, update_steps)

    attributes: Attributes = {
        -1: {},
        0: {"expected_mean": 1.0, "expected_precision": 1.0},
    }

    # Define a simple hashable dictionary class.
    class FrozenDict(dict):
        def __hash__(self):
            return hash(tuple(sorted(self.items())))

    # Wrap edges in FrozenDict for hashability.
    edges: Edges = FrozenDict({0: DummyEdge(node_type=2)})

    # Prepare input data.
    values_tuple = (jnp.array([[1.0]]),)  # one observation per node, shape (1, 1)
    observed_tuple = (jnp.array([[0]]),)  # observation mask
    time_step = 10
    inputs = (values_tuple, observed_tuple, time_step)
    input_idxs = (0,)
    rng_key = random.PRNGKey(0)

    updated_attr, _ = beliefs_propagation(
        attributes,
        inputs,
        update_sequence,
        edges,
        input_idxs,
        sophisticated=False,
        rng_key=rng_key,
    )
    # Verify that time_step is correctly assigned.
    assert updated_attr[-1]["time_step"] == time_step
    # Verify that the prediction and update functions have been called.
    assert updated_attr[0]["predicted"] is True
    assert updated_attr[0]["updated"] is True
    # In non-sophisticated mode, the values are squeezed and used to update the mean.
    # Check that the node's mean is 1.0.
    assert updated_attr[0]["mean"] == 1.0
    # Check that observed is 0 (squeezing the array if necessary).
    assert jnp.squeeze(updated_attr[0]["observed"]) == 0
