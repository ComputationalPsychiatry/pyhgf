"""Regression oracles for normalized nonnegative sample weights."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pyhgf.model import DeepNetwork
from pyhgf.utils.vectorised_belief_propagation import (
    _contract_factors,
    _reduce_importance,
)


def _net():
    return (
        DeepNetwork(feedforward_uncertainty=True)
        .add_layer(1, add_constant_input=False, volatility_parent=False)
        .add_layer(2, add_constant_input=False, volatility_parent=False)
        .add_layer(2, add_constant_input=False, volatility_parent=False)
    )


def _learn(net, x, y, sample_weight, kind):
    kwargs = (
        {"optimiser": optax.sgd(0.1)}
        if kind == "precision_weighted"
        else {"learning_kwargs": {"window": 100.0}}
    )
    net.batch_update(
        x,
        y,
        learning_kind=kind,
        sample_weight=sample_weight,
        update_precisions=True,
        **kwargs,
    )
    return net


def _assert_same_state(left, right):
    a, b = jax.tree_util.tree_leaves(left.state), jax.tree_util.tree_leaves(right.state)
    assert len(a) == len(b)
    for actual, expected in zip(a, b):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("kind", ["precision_weighted", "synaptic_uncertainty"])
def test_positive_weight_rescaling_preserves_all_carried_updates(kind):
    """The same relative weights yield the same means and precisions."""
    x = jnp.array([[0.2, -0.5], [1.3, 0.8], [-0.7, 0.4]])
    y = jnp.array([[0.6], [-0.3], [0.1]])
    weights = jnp.array([0.02, 0.03, 0.05])
    reference = _learn(_net(), x, y, weights * 100.0, kind)
    for scale in (1.0, 10.0):
        _assert_same_state(_learn(_net(), x, y, weights * scale, kind), reference)


@pytest.mark.parametrize("kind", ["precision_weighted", "synaptic_uncertainty"])
def test_binary_mask_matches_removing_finite_padded_rows(kind):
    """Zero-weight rows contribute neither gradient nor precision evidence."""
    x = jnp.array([[0.2, -0.5], [9.0, 8.0], [-0.7, 0.4], [-7.0, 6.0]])
    y = jnp.array([[0.6], [3.0], [0.1], [-2.0]])
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    weighted = _learn(_net(), x, y, mask, kind)
    selected = _learn(_net(), x[jnp.array([0, 2])], y[jnp.array([0, 2])], None, kind)
    _assert_same_state(weighted, selected)


def test_zero_total_has_zero_data_gradient_and_precision_increment():
    """Zero finite data mass stays finite and gives no SGD or evidence change."""
    net = _net()
    before = [np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(net.state)]
    _learn(
        net,
        jnp.array([[0.2, -0.5], [1.3, 0.8]]),
        jnp.array([[0.6], [-0.3]]),
        jnp.zeros(2),
        "precision_weighted",
    )
    for actual, expected in zip(jax.tree_util.tree_leaves(net.state), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("stacked", [False, True])
def test_zero_weight_factor_reductions_are_finite_zero(stacked):
    """Both matrix shapes have defined zero-data reductions."""
    shape_u, shape_v = ((3, 2, 2), (3, 2, 4)) if stacked else ((3, 2), (3, 4))
    factors = (jnp.ones(shape_u), jnp.full(shape_v, 2.0))
    for reducer in (_contract_factors, _reduce_importance):
        actual = reducer(factors, jnp.zeros(3))
        np.testing.assert_array_equal(actual, np.zeros(actual.shape))


def test_zero_total_retains_existing_synaptic_prior_dynamics():
    """No data evidence does not disable the rule's prior reversion or decay."""
    net = (
        DeepNetwork()
        .add_layer(1, kind="binary", add_constant_input=False, volatility_parent=False)
        .add_layer(1, add_constant_input=False, volatility_parent=False)
        .install_weight_belief()
    )
    parent = dataclasses.replace(
        net.state.layers[1], weights_precision_delta=jnp.array([[2.0]])
    )
    net.state = dataclasses.replace(net.state, layers=(net.state.layers[0], parent))
    net.batch_update(
        jnp.array([[0.0], [2.0]]),
        jnp.array([[0.0], [1.0]]),
        sample_weight=jnp.zeros(2),
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 10.0},
        update_precisions=False,
    )
    np.testing.assert_allclose(net.state.layers[1].weights_precision_delta, [[1.8]])
    np.testing.assert_allclose(net.state.layers[1].weights_mean, [[1.0 - 1.0 / 30.0]])
