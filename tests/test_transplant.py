# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from pyhgf.model.transplant import from_embedding, from_feedforward, from_linear

# The forward-parity checks below are learning-free wiring gates: a network
# built from transplanted Equinox weights must reproduce the Equinox forward
# pass to floating-point precision, which validates the layer order, the
# matrix orientation, and every bias placement independently of any question
# about learning.

_PARITY = dict(
    volatility_parent=False,
    precision=1e4,
    expected_precision=1e4,
)
_PARITY_LEAF = dict(volatility_parent=False)


def test_from_linear_forward_parity():
    """A transplanted Linear reproduces the Equinox forward pass, bias or not."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(5, 4)))

    for use_bias in (True, False):
        linear = eqx.nn.Linear(4, 3, use_bias=use_bias, key=random.key(1))
        net = from_linear(linear)
        np.testing.assert_allclose(
            net.predict(x), jax.vmap(linear)(x), rtol=1e-6, atol=1e-7
        )


def test_from_feedforward_forward_parity():
    """A transplanted feed-forward block reproduces ``fc2(gelu(fc1(x)))``.

    Both biases are exercised: ``fc1.bias`` inside the GELU (folded into the
    input→hidden matrix) and ``fc2.bias`` outside it (folded into the
    hidden→output matrix).
    """
    rng = np.random.default_rng(1)
    d, h = 6, 12
    k1, k2 = random.split(random.key(2))
    fc1 = eqx.nn.Linear(d, h, key=k1)
    fc2 = eqx.nn.Linear(h, d, key=k2)
    x = jnp.asarray(rng.normal(size=(5, d)))

    net = from_feedforward(fc1, fc2)
    oracle = jax.vmap(lambda row: fc2(jax.nn.gelu(fc1(row))))(x)
    np.testing.assert_allclose(net.predict(x), oracle, rtol=1e-5, atol=1e-6)


def test_from_embedding_forward_parity():
    """A transplanted embedding reproduces the table lookup from one-hot rows."""
    vocab, dim = 27, 8
    embedding = eqx.nn.Embedding(vocab, dim, key=random.key(3))
    ids = jnp.array([0, 5, 26, 5])

    net = from_embedding(embedding)
    one_hot = jax.nn.one_hot(ids, vocab)
    np.testing.assert_allclose(
        net.predict(one_hot), jax.vmap(embedding)(ids), rtol=1e-6, atol=1e-7
    )


def test_transplanted_feedforward_learns_like_backprop_with_biases():
    """One batched learning step on a transplanted, *biased* block matches backprop.

    Extends the bias-free parity result to the full feed-forward block with
    both bias vectors: in the parity configuration, the weight changes on
    every matrix — bias columns included — must match the batch-averaged
    gradients of a squared-error loss on the identical forward function, and
    the per-sample input errors must match the loss gradient at the input.
    """
    rng = np.random.default_rng(4)
    d, h, batch = 6, 12, 5
    k1, k2 = random.split(random.key(5))
    fc1 = eqx.nn.Linear(d, h, key=k1)
    fc2 = eqx.nn.Linear(h, d, key=k2)
    xb = jnp.asarray(rng.normal(size=(batch, d)))
    yb = jnp.asarray(rng.normal(size=(batch, d)))

    # Oracle: batch-mean gradients with respect to the folded [W | b] blocks.
    def forward(w1b, w2b, row):
        hidden = w1b[:, :-1] @ row + w1b[:, -1]
        return w2b[:, :-1] @ jax.nn.gelu(hidden) + w2b[:, -1]

    w1b = jnp.concatenate([fc1.weight, fc1.bias[:, None]], axis=1)
    w2b = jnp.concatenate([fc2.weight, fc2.bias[:, None]], axis=1)

    def mean_loss(w1b_, w2b_):
        out = jax.vmap(lambda row: forward(w1b_, w2b_, row))(xb)
        return jnp.mean(jnp.sum(0.5 * (yb - out) ** 2, axis=-1))

    g_w1b, g_w2b = jax.grad(mean_loss, argnums=(0, 1))(w1b, w2b)
    per_sample_dx = jax.vmap(
        lambda x_row, y_row: jax.grad(
            lambda a: jnp.sum(0.5 * (y_row - forward(w1b, w2b, a)) ** 2)
        )(x_row)
    )(xb, yb)

    # Transplant, one batch-synchronous step in the parity configuration.
    net = from_feedforward(fc1, fc2, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY)
    lr = 1e-3
    net.batch_update(xb, yb, optimiser=optax.sgd(lr), update_precisions=False)

    def norm_rel(a, b):
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    d_w2b = -(net.state.layers[1].weights_mean - w2b) / lr  # hidden→output block
    d_w1b = -(net.state.layers[2].weights_mean - w1b) / lr  # input→hidden block
    assert norm_rel(d_w1b, g_w1b) < 1e-2
    assert norm_rel(d_w2b, g_w2b) < 1e-2
    # PyHGF errors are observed-minus-predicted: the negative of the loss
    # gradient at the input.
    assert norm_rel(-net.input_errors, per_sample_dx) < 1e-2


def test_from_linears_stacks_and_validates():
    """Stack shared-input Linears, rejecting mismatched inputs or mixed biases."""
    import pytest

    from pyhgf.model.transplant import from_linears

    k1, k2, k3 = random.split(random.key(0), 3)
    a = eqx.nn.Linear(4, 3, key=k1)
    b = eqx.nn.Linear(4, 5, key=k2)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(6, 4)).astype("float32"))

    net = from_linears([a, b], leaf_kwargs=dict(volatility_parent=False))
    stacked = np.concatenate(
        [np.asarray(jax.vmap(a)(x)), np.asarray(jax.vmap(b)(x))], axis=1
    )
    np.testing.assert_allclose(
        np.asarray(net.predict(x)), stacked, rtol=1e-5, atol=1e-6
    )

    wrong_input = eqx.nn.Linear(5, 3, key=k3)
    with pytest.raises(ValueError, match="in_features"):
        from_linears([a, wrong_input])
    no_bias = eqx.nn.Linear(4, 3, use_bias=False, key=k3)
    with pytest.raises(ValueError, match="bias"):
        from_linears([a, no_bias])


def test_network_kwargs_reach_the_transplanted_networks():
    """Network-level settings are set at construction, not written afterwards.

    The transplant builders start from a bare ``DeepNetwork``, so a setting that only
    the constructor can place in the state has to travel with the call.
    """
    key = random.key(0)
    k1, k2 = random.split(key)
    linear = eqx.nn.Linear(4, 3, key=k1)
    embedding = eqx.nn.Embedding(6, 3, key=k2)
    fc1 = eqx.nn.Linear(4, 8, key=k1)
    fc2 = eqx.nn.Linear(8, 4, key=k2)

    for net in (
        from_linear(linear, network_kwargs=dict(feedforward_uncertainty=True)),
        from_embedding(embedding, network_kwargs=dict(feedforward_uncertainty=True)),
        from_feedforward(fc1, fc2, network_kwargs=dict(feedforward_uncertainty=True)),
    ):
        assert net.feedforward_uncertainty is True
        assert net.state.feedforward_uncertainty is True

    assert from_linear(linear).state.feedforward_uncertainty is False
