# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.random import PRNGKey
from pytest import raises

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.typing import AdjacencyLists, UpdateSequence
from pyhgf.utils import add_edges, add_parent, list_branches, remove_node, sample
from pyhgf.utils.beliefs_propagation import beliefs_propagation


def assert_attributes_equal(attributes: dict, original: dict):
    """Assert that two sets of network attributes hold the same nodes and values."""
    assert jax.tree.structure(attributes) == jax.tree.structure(original)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, attributes, original))


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

    # the attributes of the network still match its edges
    original = deepcopy(network.attributes)
    add_edges(network.attributes, network.edges, parent_idxs=1, children_idxs=0)
    assert_attributes_equal(network.attributes, original)

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


@pytest.mark.parametrize("child_kind", ["binary-state", "continuous-state"])
@pytest.mark.parametrize(
    "build, parent_idx",
    [
        pytest.param(
            lambda kind: (
                Network()
                .add_nodes(kind=kind, n_nodes=2)
                .add_nodes(value_children=0)
                .add_edges(parent_idxs=2, children_idxs=1)
            ),
            2,
            id="add_edges",
        ),
        pytest.param(
            lambda kind: (
                Network()
                .add_nodes(kind=kind, n_nodes=2)
                .add_nodes()
                .add_edges(parent_idxs=2, children_idxs=[0, 1])
            ),
            2,
            id="add_edges-list",
        ),
        pytest.param(
            lambda kind: (
                Network()
                .add_nodes(kind=kind)
                .add_nodes(value_children=0)
                .add_nodes(kind=kind, value_parents=1)
            ),
            1,
            id="value_parents-one-at-a-time",
        ),
        pytest.param(
            lambda kind: (
                Network().add_nodes().add_nodes(kind=kind, n_nodes=2, value_parents=0)
            ),
            0,
            id="value_parents-at-once",
        ),
    ],
)
def test_add_edges_value_children_reach_parent_updates(build, parent_idx, child_kind):
    """Test that value children added after the first one update their parent.

    The posterior updates of a parent iterate over ``zip(value_children,
    value_coupling_children, coupling_fn)``: if ``coupling_fn`` or
    ``value_coupling_children`` is shorter than ``value_children``, the last children
    are silently dropped from the update, and ``predict_mean`` raises an IndexError for
    a continuous child whose position is beyond the end of ``coupling_fn``. Every
    construction path must give the same parent as ``value_children=[0, 1]``.
    """
    rng = np.random.default_rng(0)
    if child_kind == "binary-state":
        u = rng.binomial(1, [0.2, 0.8], size=(100, 2)).astype(float)
    else:
        u = rng.normal(size=(100, 2))

    reference = (
        Network().add_nodes(kind=child_kind, n_nodes=2).add_nodes(value_children=[0, 1])
    )
    reference.input_data(input_data=u)

    network = build(child_kind)
    assert len(network.edges[parent_idx].coupling_fn) == 2
    assert network.attributes[parent_idx]["value_coupling_children"].shape == (2,)
    network.input_data(input_data=u)
    assert jnp.allclose(
        network.node_trajectories[parent_idx]["mean"],
        reference.node_trajectories[2]["mean"],
    )


def test_add_edges_coupling_fn():
    """Test that the coupling functions passed to add_edges are used."""
    rng = np.random.default_rng(0)
    u_continuous = rng.normal(size=(100,))

    # first child of a parent created without children
    network = (
        Network()
        .add_nodes()
        .add_nodes()
        .add_edges(parent_idxs=1, children_idxs=0, coupling_fn=(jnp.tanh,))
    )
    reference = (
        Network().add_nodes().add_nodes(value_children=0, coupling_fn=(jnp.tanh,))
    )
    assert network.edges[1].coupling_fn == (jnp.tanh,)
    network.input_data(input_data=u_continuous)
    reference.input_data(input_data=u_continuous)
    assert jnp.allclose(
        network.node_trajectories[1]["mean"], reference.node_trajectories[1]["mean"]
    )

    # second child, then one function per child for a list of children
    network = Network().add_nodes(n_nodes=2).add_nodes(value_children=0)
    network.add_edges(parent_idxs=2, children_idxs=1, coupling_fn=(jnp.tanh,))
    assert network.edges[2].coupling_fn == (None, jnp.tanh)
    network = (
        Network()
        .add_nodes(n_nodes=3)
        .add_edges(parent_idxs=2, children_idxs=[0, 1], coupling_fn=(None, jnp.tanh))
    )
    assert network.edges[2].coupling_fn == (None, jnp.tanh)

    # a function declared with add_nodes is kept when add_edges does not get one
    network = Network().add_nodes(coupling_fn=(jnp.tanh,)).add_nodes(value_parents=0)
    assert network.edges[0].coupling_fn == (jnp.tanh,)
    network.add_nodes(value_parents=0)
    assert network.edges[0].coupling_fn == (jnp.tanh, None)

    # constant-state parents are always linearly coupled
    network = Network().add_nodes().add_nodes(kind="constant-state")
    network.add_edges(parent_idxs=1, children_idxs=0, coupling_fn=(jnp.tanh,))
    assert network.edges[1].coupling_fn == (None,)

    # the number of functions must match the number of children
    with raises(ValueError):
        Network().add_nodes(n_nodes=3).add_edges(
            parent_idxs=2, children_idxs=[0, 1], coupling_fn=(None, None, None)
        )


@pytest.mark.parametrize("kind", ["value", "volatility"])
def test_add_edges_coupling_strengths(kind):
    """Test that each edge stores one and the same strength at both ends.

    The prediction of a child reads the strength of an edge in the child's
    ``*_coupling_parents`` and the update of the parent in the parent's
    ``*_coupling_children``: both must hold the strength of that edge.
    """
    # two parents, one child
    network = Network().add_nodes(n_nodes=3)
    network.add_edges(
        kind=kind, parent_idxs=[1, 2], children_idxs=0, coupling_strengths=[0.5, 2.0]
    )
    assert jnp.array_equal(
        network.attributes[0][f"{kind}_coupling_parents"], jnp.array([0.5, 2.0])
    )
    assert jnp.array_equal(
        network.attributes[1][f"{kind}_coupling_children"], jnp.array([0.5])
    )
    assert jnp.array_equal(
        network.attributes[2][f"{kind}_coupling_children"], jnp.array([2.0])
    )

    # one parent, two children
    network = Network().add_nodes(n_nodes=3)
    network.add_edges(
        kind=kind, parent_idxs=2, children_idxs=[0, 1], coupling_strengths=[0.5, 2.0]
    )
    assert jnp.array_equal(
        network.attributes[2][f"{kind}_coupling_children"], jnp.array([0.5, 2.0])
    )
    assert jnp.array_equal(
        network.attributes[0][f"{kind}_coupling_parents"], jnp.array([0.5])
    )
    assert jnp.array_equal(
        network.attributes[1][f"{kind}_coupling_parents"], jnp.array([2.0])
    )

    # a single strength is shared by all the new edges
    network = Network().add_nodes(n_nodes=3)
    network.add_edges(kind=kind, parent_idxs=[1, 2], children_idxs=0)
    assert jnp.array_equal(
        network.attributes[0][f"{kind}_coupling_parents"], jnp.array([1.0, 1.0])
    )

    # integer, NumPy and JAX strengths are stored as floating-point arrays
    for strengths in [[1, 2], np.float32(0.5), jnp.array(0.5), jnp.array([1.0, 2.0])]:
        network = Network().add_nodes(n_nodes=3)
        network.add_edges(
            kind=kind, parent_idxs=2, children_idxs=[0, 1], coupling_strengths=strengths
        )
        stored = network.attributes[2][f"{kind}_coupling_children"]
        assert stored.shape == (2,)
        assert jnp.issubdtype(stored.dtype, jnp.floating)

    # same beliefs as when each edge is declared with add_nodes
    u = np.random.default_rng(0).normal(size=(100,))
    network = Network().add_nodes(n_nodes=3)
    network.add_edges(
        kind=kind, parent_idxs=[1, 2], children_idxs=0, coupling_strengths=[0.5, 2.0]
    )
    reference = (
        Network()
        .add_nodes()
        .add_nodes(**{f"{kind}_children": ([0], [0.5])})
        .add_nodes(**{f"{kind}_children": ([0], [2.0])})
    )
    network.input_data(input_data=u)
    reference.input_data(input_data=u)
    for idx in range(3):
        for key in ["mean", "precision"]:
            assert jnp.allclose(
                network.node_trajectories[idx][key],
                reference.node_trajectories[idx][key],
            ), (idx, key)

    # one strength per child or per parent only, otherwise the edges are ambiguous
    for parent_idxs, children_idxs in [(2, 0), ([2, 3], [0, 1])]:
        with raises(ValueError):
            Network().add_nodes(n_nodes=4).add_edges(
                kind=kind,
                parent_idxs=parent_idxs,
                children_idxs=children_idxs,
                coupling_strengths=[0.5, 2.0],
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
        Network(volatility_updates="standard")
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
    original = deepcopy(attributes)
    new_attributes, new_edges = add_parent(attributes, edges, 1, "volatility", 1.0)

    assert len(new_attributes) == 8
    assert len(new_edges) == 7

    # the attributes of the network still match its edges
    assert_attributes_equal(network.attributes, original)

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
    original = deepcopy(attributes)
    new_attributes, new_edges = remove_node(attributes, edges, 2)

    assert len(new_attributes) == 5
    assert len(new_edges) == 4

    # the attributes of the network still match its edges
    assert_attributes_equal(network.attributes, original)


def test_remove_node_with_several_couplings():
    """Test remove_node when the other end of an edge keeps other edges.

    The coupling strengths of the remaining edges, and the coupling functions of the
    remaining value children, must stay aligned with them, so that the network behaves
    as if it had been built without the removed node.
    """
    u = np.random.default_rng(0).normal(size=(100,))
    cases = [
        # parent with two value children, one of them removed
        (
            Network()
            .add_nodes(n_nodes=2)
            .add_nodes(
                value_children=([0, 1], [0.5, 2.0]), coupling_fn=(None, jnp.tanh)
            ),
            0,
            Network()
            .add_nodes()
            .add_nodes(value_children=([0], [2.0]), coupling_fn=(jnp.tanh,)),
        ),
        # parent with two volatility children, one of them removed
        (
            Network()
            .add_nodes(n_nodes=2)
            .add_nodes(volatility_children=([0, 1], [0.5, 2.0])),
            0,
            Network().add_nodes().add_nodes(volatility_children=([0], [2.0])),
        ),
        # child with two value parents, one of them removed
        (
            Network()
            .add_nodes()
            .add_nodes(value_children=([0], [0.5]))
            .add_nodes(value_children=([0], [2.0])),
            1,
            Network().add_nodes().add_nodes(value_children=([0], [2.0])),
        ),
    ]
    for network, index, reference in cases:
        attributes, edges, _ = network.get_network()
        new_attributes, new_edges = remove_node(attributes, edges, index)
        assert new_edges == reference.edges
        for idx in range(2):
            for key in [
                "value_coupling_children",
                "value_coupling_parents",
                "volatility_coupling_children",
                "volatility_coupling_parents",
            ]:
                expected = reference.attributes[idx][key]
                if expected is None:
                    assert new_attributes[idx][key] is None
                else:
                    assert jnp.array_equal(new_attributes[idx][key], expected)

        network = Network()
        network.attributes, network.edges = new_attributes, new_edges
        network.input_data(input_data=u)
        reference.input_data(input_data=u)
        for idx in range(2):
            for key in ["mean", "precision"]:
                assert jnp.allclose(
                    network.node_trajectories[idx][key],
                    reference.node_trajectories[idx][key],
                )

    # a single strength equal to zero is removed with its edge
    attributes, edges, _ = (
        Network().add_nodes().add_nodes(value_children=([0], [0.0])).get_network()
    )
    new_attributes, _ = remove_node(attributes, edges, 0)
    assert new_attributes[0]["value_coupling_children"] is None


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
