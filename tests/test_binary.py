# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp

from pyhgf import load_data
from pyhgf.math import binary_surprise, gaussian_density, sigmoid
from pyhgf.model import Network
from pyhgf.utils import beliefs_propagation


def test_gaussian_density():
    surprise = gaussian_density(
        x=jnp.array([1.0, 1.0]),
        mean=jnp.array([0.0, 0.0]),
        precision=jnp.array([1.0, 1.0]),
    )
    assert jnp.all(jnp.isclose(surprise, 0.24197073))


def test_sgm():
    assert jnp.all(jnp.isclose(sigmoid(jnp.array([0.3, 0.3])), 0.5744425))


def test_binary_surprise():
    surprise = binary_surprise(
        x=jnp.array([1.0]),
        expected_mean=jnp.array([0.2]),
    )
    assert jnp.all(jnp.isclose(surprise, 1.609438))


def test_update_binary_input_parents():

    binary_hgf = (
        Network()
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0, mean=1.0, tonic_volatility=1.0)
        .add_nodes(volatility_children=1, mean=1.0, tonic_volatility=1.0)
        .create_belief_propagation_fn()
    )

    data = jnp.ones(1)
    time_steps = jnp.ones(1)
    observed = jnp.ones(1)

    # apply sequence
    new_attributes, _ = beliefs_propagation(
        attributes=binary_hgf.attributes,
        inputs=(data, time_steps, observed),
        update_sequence=binary_hgf.update_sequence,
        edges=binary_hgf.edges,
        input_idxs=(0,),
    )
    for idx, val in zip(
        ["mean", "expected_mean", "expected_precision"],
        [1.0, 0.7310586, 0.19661193],
    ):
        assert jnp.isclose(new_attributes[0][idx], val)
    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [1.8515793, 1.0, 0.31581485, 0.11920292],
    ):
        assert jnp.isclose(new_attributes[1][idx], val)
    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [0.12210602, 1.0, 0.47702926, 0.26894143],
    ):
        assert jnp.isclose(new_attributes[2][idx], val)


def test_binary_scan_loop():

    u, _ = load_data("binary")

    binary_hgf = (
        Network()
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0, mean=0.0, tonic_volatility=1.0)
        .add_nodes(volatility_children=1, mean=0.0, tonic_volatility=1.0)
        .input_data(input_data=u)
    )

    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [-1.5600492, -1.5068374, 3.4091694, 3.2606704],
    ):
        assert jnp.isclose(binary_hgf.node_trajectories[1][idx][-1], val)
    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [-5.35091, -5.320357, 0.025971143, 0.024351958],
    ):
        assert jnp.isclose(binary_hgf.node_trajectories[2][idx][-1], val)
