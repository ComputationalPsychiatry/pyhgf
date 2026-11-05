"""Custom update branches must retain final beliefs without changing replay."""

import numpy as np

from pyhgf.model import Network


def _build():
    return Network().add_nodes().add_nodes(value_children=0)


def _custom_input(net, observations):
    sequence = net.get_network()[2]
    return net.input_custom_sequence(
        update_branches=(sequence,),
        branches_idx=np.zeros(len(observations), dtype=int),
        input_data=observations,
    )


def _assert_same_beliefs(actual, expected):
    for i in range(2):
        for field in ("mean", "precision", "expected_mean", "expected_precision"):
            np.testing.assert_allclose(
                actual.node_trajectories[i][field],
                expected.node_trajectories[i][field],
                rtol=1e-6,
                atol=1e-7,
            )


def test_custom_sequence_then_append_matches_whole_stream():
    """An ordinary append must continue the final beliefs of a custom sequence."""
    observations = np.array([0.0, 1.0, 2.0, 3.0])
    custom = _custom_input(_build(), observations[:2])
    assert custom.last_attributes is not None
    custom.input_data(observations[2:], append=True)
    reference = _build().input_data(observations)
    _assert_same_beliefs(custom, reference)


def test_custom_sequence_replays_from_initial_attributes():
    """Retaining final carry must not make repeated custom calls append implicitly."""
    observations = np.array([0.0, 1.0, 2.0])
    custom = _custom_input(_build(), observations)
    _custom_input(custom, observations)
    reference = _build().input_data(observations)
    _assert_same_beliefs(custom, reference)
    assert len(custom.node_trajectories[1]["mean"]) == len(observations)


def test_default_input_after_custom_sequence_still_replays():
    """A subsequent append=False call must replace history and use initial beliefs."""
    custom = _custom_input(_build(), np.array([0.0, 1.0]))
    observations = np.array([2.0, 3.0])
    custom.input_data(observations)
    reference = _build().input_data(observations)
    _assert_same_beliefs(custom, reference)
    assert len(custom.node_trajectories[1]["mean"]) == len(observations)
