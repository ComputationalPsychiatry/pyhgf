# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network


@pytest.mark.parametrize("relation", ["value", "volatility"])
@pytest.mark.parametrize("cached_sequence", [False, True])
def test_parent_first_construction_observes_only_the_leaf(relation, cached_sequence):
    """Adding a child removes its parent from the automatic input nodes."""
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
