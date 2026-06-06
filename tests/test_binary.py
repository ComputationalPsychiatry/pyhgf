# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp

from pyhgf import load_data
from pyhgf.math import binary_surprise, gaussian_density
from pyhgf.model import Network
from pyhgf.utils import beliefs_propagation


def test_gaussian_density():
    """Test the Gaussian density function."""
    surprise = gaussian_density(
        x=jnp.array([1.0, 1.0]),
        mean=jnp.array([0.0, 0.0]),
        precision=jnp.array([1.0, 1.0]),
    )
    assert jnp.all(jnp.isclose(surprise, 0.24197073))


def test_binary_surprise():
    """Test the binary surprise function."""
    surprise = binary_surprise(
        x=jnp.array([1.0]),
        expected_mean=jnp.array([0.2]),
    )
    assert jnp.all(jnp.isclose(surprise, 1.609438))


def test_update_binary_input_parents():
    """Test the update of the binary input parents."""
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
        inputs=(data, time_steps, observed, None),
        update_sequence=binary_hgf.update_sequence,
        edges=binary_hgf.edges,
        input_idxs=(0,),
    )
    # Expected values reflect the piHGF prediction step: the volatility-parent
    # variance enters node 1's predicted log-volatility through the exact
    # moment-generating-function correction κ²/(2 π̂_parent), which shifts
    # node 1's posterior mean and precision relative to the canonical formula.
    for idx, val in zip(
        ["mean", "expected_mean", "expected_precision"],
        [1.0, 0.7310586, 0.19661193],
    ):
        assert jnp.isclose(new_attributes[0][idx], val)
    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [2.2378633, 1.0, 0.21726260, 0.02065066],
    ):
        assert jnp.isclose(new_attributes[1][idx], val)
    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [0.37627372, 1.0, 0.39340553, 0.26894143],
    ):
        assert jnp.isclose(new_attributes[2][idx], val)


def test_binary_scan_loop():
    """Test the binary scan loop."""
    u, _ = load_data("binary")

    binary_hgf = (
        Network()
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0, mean=0.0, tonic_volatility=-4.0)
        .add_nodes(volatility_children=1, mean=0.0, tonic_volatility=-4.0)
        .input_data(input_data=u)
    )

    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [-2.3191204, -2.171629, 0.6937843, 0.6019279],
    ):
        assert jnp.isclose(binary_hgf.node_trajectories[1][idx][-1], val)
    for idx, val in zip(
        ["mean", "expected_mean", "precision", "expected_precision"],
        [2.478936, 2.4795845, 1.7116449, 1.6612335],
    ):
        assert jnp.isclose(binary_hgf.node_trajectories[2][idx][-1], val)


def test_volatile_examples():
    """Test extrem conditions to ensure numerical stability."""
    # repeated identical observations
    hgf = (
        Network(update_type="unbounded")
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=[0], tonic_volatility=5.0)
    )

    # simulate stable blocks of observations
    input_data = [1.0] * 50
    input_data.extend([0.0] * 50)

    hgf.input_data(input_data=jnp.asarray(input_data))
    assert not jnp.any(jnp.isnan(hgf.node_trajectories[0]["expected_mean"]))

    # repeated identical observations
    hgf = (
        Network(update_type="unbounded")
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=[0], tonic_volatility=5.0)
    )

    # simulate stable blocks of observations
    input_data = [0.0, 1.0] * 50

    hgf.input_data(input_data=jnp.asarray(input_data))
    assert not jnp.any(jnp.isnan(hgf.node_trajectories[0]["expected_mean"]))
