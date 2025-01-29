# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp

from pyhgf import load_data
from pyhgf.model import Network


def test_one_node_hgf():

    # one level HGF and one observation step
    one_node__hgf = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data(input_data=jnp.array([0.2]))
    )

    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.0, 1.0, 0.2, 0.0],
    ):
        assert jnp.isclose(one_node__hgf.node_trajectories[0][key], val)
    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.9820137, 0.98201376, 0.10090748, 0.0],
    ):
        assert jnp.isclose(one_node__hgf.node_trajectories[1][key], val)


def test_two_nodes_hgf():

    # one level HGF with a volatility parent and one observation step
    two_nodes__hgf = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
        .input_data(input_data=jnp.array([0.2]))
    )

    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.0, 0.5, 0.2, 0.0],
    ):
        assert jnp.isclose(two_nodes__hgf.node_trajectories[0][key], val)
    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.4820137, 0.98201376, 0.06747576, 0.0],
    ):
        assert jnp.isclose(two_nodes__hgf.node_trajectories[1][key], val)
    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.1070137, 0.98201376, -0.12219789, 0.0],
    ):
        assert jnp.isclose(two_nodes__hgf.node_trajectories[2][key], val)


def test_nonlinear_coupling_fn():
    """Tests if the coupling function is passed correctly
    into the network"""

    # creating a simple coupling function
    def coupling_fn(x):
        return jnp.sin(x)

    def identity_fn(x):
        return x

    # I create a network with a node with 2 value children
    test_HGF = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0, coupling_fn=(identity_fn,))
        .add_nodes(value_children=1, coupling_fn=(coupling_fn,))
    )

    # check if the number of coupling fn matches the number of children
    coupling_fn_length = []
    children_number = []
    for node_idx in range(2, len(test_HGF.edges)):
        coupling_fn_length.append(len(test_HGF.edges[node_idx].coupling_fn))
        children_number.append(len(test_HGF.edges[node_idx].value_children))

    assert children_number == coupling_fn_length

    # pass data
    test_HGF.input_data(input_data=jnp.array([[0.2, 0.4], [0.2, 0.4]]))

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [2.9125834, 1.9125834, 0.13492969, 0.10090748],
    ):
        assert jnp.isclose(test_HGF.node_trajectories[2][idx][1], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [2.9124038, 1.9125834, 0.2689424, 0.20181496],
    ):
        assert jnp.isclose(test_HGF.node_trajectories[3][idx][1], val)


def test_continuous_scan_loop():
    timeserie = load_data("continuous")

    two_level_hgf = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
        .input_data(input_data=timeserie)
    )

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.0, 0.9716732, 0.8241, 0.79489124],
    ):
        assert jnp.isclose(two_level_hgf.node_trajectories[0][idx][-1], val)

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [7.785051, 6.813378, 0.79853684, 0.79489124],
    ):
        assert jnp.isclose(two_level_hgf.node_trajectories[1][idx][-1], val)

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [0.25191233, 0.25114372, -3.5367591, -3.5352085],
    ):
        assert jnp.isclose(two_level_hgf.node_trajectories[2][idx][-1], val)
