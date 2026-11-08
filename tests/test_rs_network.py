# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests for the Rust ``Network`` class: input bookkeeping and input validation."""

import numpy as np
import pytest
from pyhgf.rshgf import Network


def _assert_same_trajectories(actual, expected):
    assert len(actual) == len(expected)
    for actual_node, expected_node in zip(actual, expected, strict=True):
        assert actual_node.keys() == expected_node.keys()
        for field in actual_node:
            np.testing.assert_array_equal(actual_node[field], expected_node[field])


@pytest.mark.parametrize("relation", ["value", "volatility"])
@pytest.mark.parametrize("cached_sequence", [False, True])
def test_parent_first_construction_observes_only_the_leaf(relation, cached_sequence):
    """Adding a child removes its parent from the inputs and from a cached schedule."""
    parent_parameters = dict(
        precision=1.0, tonic_volatility=-4.0, autoconnection_strength=1.0
    )
    child_parameters = dict(
        precision=1.0, tonic_volatility=0.0, autoconnection_strength=0.0
    )
    parent_first = Network(volatility_updates="standard").add_nodes(**parent_parameters)
    if cached_sequence:
        parent_first.set_update_sequence()
    parent_first.add_nodes(**{f"{relation}_parents": 0}, **child_parameters)
    leaf_first = (
        Network(volatility_updates="standard")
        .add_nodes(**child_parameters)
        .add_nodes(**{f"{relation}_children": 0}, **parent_parameters)
    )
    assert parent_first.inputs == [1]
    assert leaf_first.inputs == [0]

    observations = np.array([0.2, 0.4, -0.1])
    parent_first.input_data(observations)
    leaf_first.input_data(observations)
    np.testing.assert_array_equal(
        parent_first.node_trajectories[1]["mean"], observations
    )
    for actual_idx, expected_idx in [(0, 1), (1, 0)]:
        actual = parent_first.node_trajectories[actual_idx]
        expected = leaf_first.node_trajectories[expected_idx]
        for field in ("mean", "expected_mean", "precision", "expected_precision"):
            np.testing.assert_allclose(actual[field], expected[field], rtol=1e-12)


def test_adding_a_child_preserves_other_input_nodes():
    """A new parent link removes only the node that gained a child."""
    net = Network().add_nodes(n_nodes=2).add_nodes(value_parents=0)
    assert net.inputs == [1, 2]


def test_input_data_accepts_one_column_per_input():
    """Two inputs accept two observation columns and one time step per row."""
    observations = np.array([[0.1, 0.2], [0.3, 0.4]])
    net = Network().add_nodes(n_nodes=2)
    assert net.input_data(observations, time_steps=[1.0, 0.5]) is net
    for idx in range(2):
        np.testing.assert_array_equal(
            net.node_trajectories[idx]["mean"], observations[:, idx]
        )


@pytest.mark.parametrize(
    ("observations", "time_steps", "message"),
    [
        pytest.param([0.3, 0.4], [1.0], "time_steps", id="short-time-steps"),
        pytest.param([0.3, 0.4], [1.0, 1.0, 1.0], "time_steps", id="long-time-steps"),
        pytest.param([[0.3, 0.4]], None, "input_data", id="extra-observation"),
        pytest.param([[]], None, "input_data", id="missing-observation"),
        pytest.param([[0.3], [0.4, 0.5]], None, "input_data", id="wide-later-row"),
        pytest.param([[0.3], []], None, "input_data", id="empty-later-row"),
    ],
)
def test_invalid_input_shapes_leave_the_filter_unchanged(
    observations, time_steps, message
):
    """An invalid input is rejected before any belief or trajectory changes."""

    def build():
        return (
            Network(volatility_updates="standard")
            .add_nodes()
            .add_nodes(value_children=0)
            .input_data([0.1, 0.2])
        )

    net, control = build(), build()
    before = net.node_trajectories
    with pytest.raises(ValueError, match=message):
        net.input_data(observations, time_steps=time_steps)
    _assert_same_trajectories(net.node_trajectories, before)

    net.input_data([0.3])
    control.input_data([0.3])
    _assert_same_trajectories(net.node_trajectories, control.node_trajectories)


def test_invalid_input_does_not_create_an_update_sequence():
    """Shapes are validated before the update sequence is built."""
    net = Network().add_nodes().add_nodes(value_children=0)
    with pytest.raises(ValueError, match="time_steps"):
        net.input_data([0.1], time_steps=[])
    assert net.update_sequence == []
    assert net.node_trajectories == []
