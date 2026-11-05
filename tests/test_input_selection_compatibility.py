"""Selector compatibility around target-dependent cache invalidation."""

import numpy as np

from pyhgf.model import Network


def test_numpy_selector_preserves_input_parameters():
    """Bookkeeping must not read the parameter-mutating automatic-input property."""
    net = Network().add_nodes(
        n_nodes=2, autoconnection_strength=0.8, tonic_volatility=-7.0
    )
    net.input_data(np.array([5.0, 6.0]), input_idxs=np.array([1]))
    assert net.input_idxs == (1,)
    for i in range(2):
        assert net.attributes[i]["autoconnection_strength"] == 0.8
        assert net.attributes[i]["tonic_volatility"] == -7.0
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], [5.0, 6.0])


def test_resetting_selector_restores_automatic_dimensions():
    """Resetting an explicit selection must rebuild the default input layout."""
    net = Network().add_nodes(n_nodes=2)
    net.input_data(np.array([1.0, 2.0]), input_idxs=(0,))
    net.input_idxs = None
    net.input_data(np.array([[11.0, 12.0], [13.0, 14.0]]))
    assert net.input_idxs == (0, 1)
    assert net.input_dim == [1, 1]
    np.testing.assert_array_equal(net.node_trajectories[0]["mean"], [11.0, 13.0])
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], [12.0, 14.0])
