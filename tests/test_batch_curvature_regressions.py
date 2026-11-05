"""Regression oracles for exact batch-mean synaptic curvature."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyhgf.model import DeepNetwork
from pyhgf.model.fused import FusedPipeline
from pyhgf.model.hybrid import DeepNetworkAdapter
from pyhgf.typing.vectorised import LayerStack
from pyhgf.utils.vectorised_belief_propagation import _reduce_importance


@pytest.mark.parametrize("kind", ["binary", "categorical"])
@pytest.mark.parametrize("mask", [None, [1.0, 0.0, 1.0]])
def test_batch_weight_precision_matches_likelihood_hessian(kind, mask):
    """A one-layer likelihood has an independent exact Hessian oracle."""
    x = jnp.array([[0.0, 0.5], [2.0, 1.0], [-1.0, 0.5]])
    if kind == "binary":
        weights = jnp.array([[0.8, -0.3]])
        targets = jnp.array([[0.0], [1.0], [0.0]])
    else:
        weights = jnp.array([[0.2, -0.7], [1.1, 0.3], [-0.4, 0.8]])
        targets = jax.nn.one_hot(jnp.array([0, 2, 1]), 3)
    sample_weight = None if mask is None else jnp.asarray(mask)

    def objective(flat_weights):
        logits = x @ flat_weights.reshape(weights.shape).T
        if kind == "binary":
            per_row = (jax.nn.softplus(logits) - targets * logits).sum(axis=-1)
        else:
            per_row = jax.scipy.special.logsumexp(logits, axis=-1) - (
                targets * logits
            ).sum(axis=-1)
        if sample_weight is None:
            return per_row.mean()
        return (sample_weight * per_row).sum() / sample_weight.sum()

    hessian = jax.hessian(objective)(weights.reshape(-1))
    expected = jnp.diag(hessian).reshape(weights.shape)
    net = (
        DeepNetwork()
        .add_layer(
            weights.shape[0],
            kind=kind,
            add_constant_input=False,
            volatility_parent=False,
        )
        .add_layer(2, add_constant_input=False, volatility_parent=False)
    )
    parent = dataclasses.replace(net.state.layers[1], weights_mean=weights)
    net.state = dataclasses.replace(net.state, layers=(net.state.layers[0], parent))
    net.batch_update(
        x,
        targets,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 100.0},
        update_precisions=False,
        sample_weight=sample_weight,
    )
    actual = net.state.layers[1].weights_precision_delta
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("mask", [None, [1.0, 0.0, 2.0]])
def test_importance_reduction_keeps_per_sample_factor_pairs(stacked, mask):
    """Deliberately correlated factors retain their joint weighted moments."""
    p = np.array(
        [
            [[0.1, 0.4], [0.2, 0.8]],
            [[0.7, 0.2], [0.9, 0.1]],
            [[0.3, 0.9], [0.4, 0.6]],
        ],
        dtype=np.float32,
    )
    q = np.array(
        [
            [[0.0, 1.0, 4.0], [2.0, 0.0, 3.0]],
            [[4.0, 2.0, 0.0], [0.0, 5.0, 1.0]],
            [[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    if not stacked:
        p, q = p[:, 0], q[:, 0]
    row_weights = np.ones(3) if mask is None else np.asarray(mask)
    per_sample = [
        np.asarray(a)[..., :, None] * np.asarray(b)[..., None, :] for a, b in zip(p, q)
    ]
    expected = np.average(np.stack(per_sample), axis=0, weights=row_weights)
    actual = _reduce_importance(
        (jnp.asarray(p), jnp.asarray(q)),
        None if mask is None else jnp.asarray(mask),
    )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def _stack_net(stacked):
    net = DeepNetwork(coupling_fn=jnp.tanh).add_layer(
        2, kind="categorical", add_constant_input=False, volatility_parent=False
    )
    kwargs = dict(add_constant_input=False, volatility_parent=False, precision=2.0)
    if stacked:
        net.add_layer_stack([2] * 5, **kwargs)
    else:
        for _ in range(5):
            net.add_layer(2, **kwargs)
    # Adding this layer rebuilds the builder after its scan block was registered.
    return net.add_layer(2, **kwargs).weight_initialisation(
        "xavier", key=jax.random.key(7)
    )


def _one_pipeline_step(stacked, weight_reuse):
    net = _stack_net(stacked)
    if stacked:
        assert any(isinstance(layer, LayerStack) for layer in net.state.layers)
    part = DeepNetworkAdapter(
        net,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 100.0},
        update_precisions=False,
        weight_reuse=weight_reuse,
    )
    pipeline = FusedPipeline(part, error_normalisation="rows")
    pipeline.step(
        jnp.array([[0.5, -0.1], [1.3, 0.4], [-0.8, 0.7]]),
        jax.nn.one_hot(jnp.array([0, 1, 0]), 2),
    )
    pipeline.merge()
    return [
        (np.asarray(w), np.asarray(p))
        for layer in net.state.layers
        if layer.weights_mean is not None
        for w, p in (
            zip(layer.weights_mean, layer.weights_precision_delta)
            if isinstance(layer, LayerStack)
            else [(layer.weights_mean, layer.weights_precision_delta)]
        )
    ]


def test_matrix_importance_reaches_stacks_and_weight_reuse():
    """Full increments preserve scanned parity and reuse scales every slice."""
    scanned = _one_pipeline_step(True, 3.0)
    plain = _one_pipeline_step(False, 3.0)
    single_use = _one_pipeline_step(True, 1.0)
    assert len(scanned) == len(plain) == len(single_use) == 6
    for (sw, sp), (pw, pp), (_, one_p) in zip(scanned, plain, single_use):
        assert np.all(np.isfinite(sp))
        np.testing.assert_allclose(sw, pw, rtol=2e-4, atol=1e-7)
        np.testing.assert_allclose(sp, pp, rtol=2e-4, atol=1e-9)
        np.testing.assert_allclose(sp, 3.0 * one_p, rtol=2e-4, atol=1e-9)
