# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network


def _network():
    return (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data([0.1, 0.2])
    )


def _assert_trajectories_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_node, expected_node in zip(actual, expected, strict=True):
        assert actual_node.keys() == expected_node.keys()
        for field in actual_node:
            np.testing.assert_array_equal(actual_node[field], expected_node[field])


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
    """Reject the whole invalid input before changing any live filter state."""
    net, control = _network(), _network()
    before = net.node_trajectories
    with pytest.raises(ValueError, match=message):
        net.input_data(observations, time_steps=time_steps)
    _assert_trajectories_equal(net.node_trajectories, before)

    net.input_data([0.3])
    control.input_data([0.3])
    _assert_trajectories_equal(net.node_trajectories, control.node_trajectories)


def test_input_data_accepts_one_column_per_input():
    """Two inputs accept two observation columns and one time step per row."""
    observations = np.array([[0.1, 0.2], [0.3, 0.4]])
    net = Network().add_nodes(n_nodes=2)
    assert net.input_data(observations, time_steps=[1.0, 0.5]) is net
    for idx in range(2):
        np.testing.assert_array_equal(
            net.node_trajectories[idx]["mean"], observations[:, idx]
        )


def test_invalid_input_does_not_create_an_update_sequence():
    """Validate shapes before even creating the filter's lazy schedule."""
    net = Network().add_nodes().add_nodes(value_children=0)
    assert net.update_sequence == []
    assert net.node_trajectories == []
    with pytest.raises(ValueError, match="time_steps"):
        net.input_data([0.1], time_steps=[])
    assert net.update_sequence == []
    assert net.node_trajectories == []
