# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import pytest
from jax.random import PRNGKey
from pytest import raises
import numpy as np
from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.typing import AdjacencyLists, UpdateSequence
from pyhgf.utils import add_parent, list_branches, remove_node, sample, set_coupling
from pyhgf.utils.beliefs_propagation import beliefs_propagation


def test_imports():
    """Test the data import function."""
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

    # ensure the coupling function match with the number of children
    network = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=[0, 1], coupling_fn=(jnp.tanh, jnp.tanh))
    )
    assert (
        len(network.edges[2].coupling_fn)
        == network.attributes[2]["value_coupling_children"].shape[0]
    )


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

    assert len(network1.update_sequence.prediction_steps) == 2
    assert len(network1.update_sequence.update_steps) == 2

    # a standard continuous HGF
    network2 = (
        Network(update_type="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .create_belief_propagation_fn()
    )
    assert len(network2.update_sequence.prediction_steps) == 3
    assert len(network2.update_sequence.update_steps) == 4

    # an EF state node
    network3 = Network().add_nodes(kind="ef-state").create_belief_propagation_fn()
    assert len(network3.update_sequence.prediction_steps) == 0
    assert len(network3.update_sequence.update_steps) == 1

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
    assert len(network4.update_sequence.prediction_steps) == 1
    assert len(network4.update_sequence.update_steps) == 3


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


def test_belief_propagation():
    """Test the belief propagation function for three observation types."""
    network = (
        Network()
        .add_nodes(kind="continuous-state")
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
    )

    attributes, edges, update_sequence = network.get_network()

    # 1 - External ---------------------------------------------------------------------
    new_attributes, _ = beliefs_propagation(
        attributes=attributes,
        inputs=(jnp.array([0.25, 1.0]), jnp.array([1, 1]), 1.0, None),
        update_sequence=update_sequence,
        edges=edges,
        input_idxs=(0, 1),
        observations="external",
    )

    # 2 - Generative -------------------------------------------------------------------
    rng_key = PRNGKey(0)
    new_attributes, _ = beliefs_propagation(
        attributes=attributes,
        inputs=(None, None, 1.0, rng_key),
        update_sequence=update_sequence,
        edges=edges,
        input_idxs=(0, 1),
        observations="generative",
    )
    assert jnp.isclose(new_attributes[0]["mean"], -0.20584226)
    assert jnp.isclose(new_attributes[1]["mean"], 1.0)

    # 3 - Deprived ---------------------------------------------------------------------
    new_attributes, _ = beliefs_propagation(
        attributes=attributes,
        inputs=(jnp.array([0.25, 1.0]), jnp.array([1, 1]), 1.0, None),
        update_sequence=update_sequence,
        edges=edges,
        input_idxs=(0, 1),
        observations="deprived",
    )
    assert jnp.isclose(new_attributes[0]["mean"], 0.0)
    assert jnp.isclose(new_attributes[1]["mean"], 0.0)

    # expected error when the parameter has invalid name
    with pytest.raises(KeyError):
        new_attributes, _ = beliefs_propagation(
            attributes=attributes,
            inputs=(jnp.array([0.25, 1.0]), jnp.array([1, 1]), 1.0, None),
            update_sequence=update_sequence,
            edges=edges,
            input_idxs=(0, 1),
            observations="error",
        )

    # with a custom functions
    def action_fn(node_idx, attributes, inputs):
        return attributes, inputs

    def update_fn(node_idx, attributes, edges, **args):
        return attributes

    update_sequence = UpdateSequence(
        prediction_steps=update_sequence.prediction_steps,
        update_steps=update_sequence.update_steps,
        pre_prediction_steps=((0, update_fn),),
        post_update_steps=((0, update_fn),),
        action_steps=((0, action_fn),),
    )

    new_attributes, _ = beliefs_propagation(
        attributes=attributes,
        inputs=(jnp.array([0.25, 1.0]), jnp.array([1, 1]), 1.0, None),
        update_sequence=update_sequence,
        edges=edges,
        input_idxs=(0, 1),
        observations="external",
    )


def test_learning():
    """Test the learning method for deep networks."""
    # here x represents the visual input (River / No River)
    x = np.array([1.0, 1.0])
    x += np.random.normal(size=x.shape) / 100

    # y represents the auditory and olfactory stimuli
    y = np.array([[1.0, 1.0], [1.0, 0.0]]).T

    network = (
        Network(update_type="unbounded")
        .add_nodes(n_nodes=2, precision=2.0, expected_precision=2.0)
        .add_nodes(
            value_children=[0, 1],
            autoconnection_strength=0,
            coupling_fn=(jnp.tanh, jnp.tanh),
        )
        .add_nodes(value_children=2, autoconnection_strength=0, coupling_fn=(jnp.tanh,))
    )

    network.train(
        x=x,
        y=y,
        inputs_x_idxs=(3,),
        inputs_y_idxs=(0, 1),
    )


def test_sample():
    """Test the sample function.

    Ensure it returns a dictionary of arrays, where each array's first dimension is
    equal to the number of predictions.
    """
    # Create a minimal network instance.
    network = (
        Network()
        .add_nodes(kind="continuous-state")
        .create_belief_propagation_fn(sampling_fn=True)
    )

    # Define the number of predictions to generate.
    n_predictions = 3

    # Call the predict function using a fixed RNG key.
    rng_key = PRNGKey(42)
    samples = sample(
        network, time_steps=jnp.ones(20), n_predictions=n_predictions, rng_key=rng_key
    )

    # Check that the returned sample is a dictionary.
    assert isinstance(samples, dict), "Predictions should be a dictionary."

    # Iterate over each key-value pair in the predictions dictionary.
    assert samples[0]["expected_mean"].shape[0] == n_predictions


def test_set_coupling():
    """Test the set_coupling function."""
    network = Network().add_nodes(n_nodes=3).add_nodes(value_children=[0, 1, 2])

    attributes = set_coupling(
        attributes=network.attributes,
        edges=network.edges,
        parent_idx=3,
        child_idx=0,
        coupling=0.5,
    )

    assert attributes[0]["value_coupling_parents"][0] == 0.5
    assert attributes[3]["value_coupling_children"][0] == 0.5
