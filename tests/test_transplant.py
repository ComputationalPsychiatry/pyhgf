# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from pyhgf.model import from_embedding, from_feedforward, from_linear

# The forward-parity checks below are learning-free wiring gates: a network
# built from transplanted Equinox weights must reproduce the Equinox forward
# pass to floating-point precision, which validates the layer order, the
# matrix orientation, and every bias placement independently of any question
# about learning.


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
