"""Observation selection changes must update routing without resetting beliefs."""

import numpy as np
from jax import random

from pyhgf.model import Network


def test_changed_input_target_preserves_parameters_and_carry():
    """Route new observations correctly while retaining caller parameters and carry."""
    net = Network().add_nodes(n_nodes=2, mean=3.0)
    net.input_data(np.array([1.0, 2.0]), input_idxs=(0,))
    carry = net.last_attributes
    net.attributes[1]["tonic_drift"] = 0.75
    net.attributes[1]["precision"] = 9.0

    net.input_idxs = (1,)

    assert net.last_attributes is carry
    assert net.attributes[1]["tonic_drift"] == 0.75
    assert net.attributes[1]["precision"] == 9.0
    net.input_data(np.array([11.0, 12.0]), input_idxs=(1,))
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], [11.0, 12.0])
    np.testing.assert_array_equal(net.node_trajectories[0]["mean"], [3.0, 3.0])


def test_reordered_input_targets_follow_column_order():
    """Changing target order must change which node receives each data column."""
    net = Network().add_nodes(n_nodes=2)
    net.input_data(np.array([[1.0, 2.0]]), input_idxs=(0, 1))
    net.input_data(np.array([[11.0, 12.0], [13.0, 14.0]]), input_idxs=(1, 0))
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], [11.0, 13.0])
    np.testing.assert_array_equal(net.node_trajectories[0]["mean"], [12.0, 14.0])


def test_changed_input_targets_recalculate_dimensions():
    """Switching from a scalar to a vector node must rebuild input dimensions."""
    net = (
        Network()
        .add_nodes(kind="ef-state")
        .add_nodes(kind="ef-state", distribution="multivariate-normal", dimension=2)
    )
    net.input_idxs = (0,)
    net.create_belief_propagation_fn()
    assert net.input_dim == [1]

    observations = np.array([[1.0, 2.0], [3.0, 4.0]])
    net.input_data(observations, input_idxs=(1,))

    assert net.input_dim == [2]
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], observations)


def test_unchanged_targets_preserve_replay_and_parameter_edits():
    """Selecting the same targets must preserve replay and caller parameter edits."""
    observations = np.linspace(0.0, 1.0, 6)
    net = Network().add_nodes().add_nodes(value_children=0)
    net.input_data(observations)
    first = np.asarray(net.node_trajectories[1]["mean"]).copy()
    scan_fn = net.scan_fn
    net.input_data(observations, input_idxs=(0,))
    assert net.scan_fn is scan_fn
    np.testing.assert_allclose(net.node_trajectories[1]["mean"], first)

    net.attributes[1]["tonic_drift"] = 0.5
    net.input_data(observations, input_idxs=(0,))
    reference = Network().add_nodes().add_nodes(value_children=0, tonic_drift=0.5)
    reference.input_data(observations)
    np.testing.assert_allclose(
        net.node_trajectories[1]["mean"], reference.node_trajectories[1]["mean"]
    )
    assert not np.allclose(net.node_trajectories[1]["mean"], first)


def test_same_targets_keep_append_history():
    """Equivalent explicit selections must not discard carry or recorded history."""
    observations = np.linspace(0.0, 1.0, 6)
    net = Network().add_nodes().add_nodes(value_children=0)
    net.input_data(observations[:3], input_idxs=(0,), append=True)
    net.input_idxs = (0,)
    net.input_data(observations[3:], input_idxs=(0,), append=True)
    reference = Network().add_nodes().add_nodes(value_children=0)
    reference.input_data(observations, input_idxs=(0,))
    for i in range(2):
        np.testing.assert_allclose(
            net.node_trajectories[i]["mean"],
            reference.node_trajectories[i]["mean"],
        )


def test_changed_targets_rebind_existing_generative_function():
    """An already enabled sampler must follow the newly selected input."""
    net = Network().add_nodes(n_nodes=2)
    net.input_data(np.array([1.0, 2.0]), input_idxs=(0,))
    net.create_belief_propagation_fn(sampling_fn=True)
    net.input_idxs = (1,)
    net.sample(n_predictions=1, time_steps=np.ones(32), rng_key=random.key(7))
    np.testing.assert_array_equal(net.samples[0]["mean"][0], np.full(32, 2.0))
    assert np.std(np.asarray(net.samples[1]["mean"][0])) > 0.0
