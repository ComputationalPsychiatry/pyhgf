# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pyhgf.model import (
    DeepNetwork,
    DeepNetworkAdapter,
    FusedPipeline,
    PCSequential,
    Residual,
    gelu_adapter,
    layer_norm_adapter,
)

# Automatic differentiation appears in this file ONLY as a test oracle: the
# hand-derived backward formulas and the pipeline's error routing must match
# what autodiff computes on the same forward functions. The pipeline itself
# never uses autodiff.


def _random_layer_norm(dim: int, rng) -> eqx.nn.LayerNorm:
    """Build an ``eqx.nn.LayerNorm`` with non-trivial scale and shift."""
    ln = eqx.nn.LayerNorm(dim)
    ln = eqx.tree_at(lambda m: m.weight, ln, jnp.asarray(rng.normal(size=(dim,)) + 1.0))
    ln = eqx.tree_at(lambda m: m.bias, ln, jnp.asarray(rng.normal(size=(dim,)) * 0.3))
    return ln


def test_gelu_backward_matches_autodiff():
    """The hand-derived GELU backward equals the autodiff of the same forward."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(5, 7)))
    error = jnp.asarray(rng.normal(size=(5, 7)))

    part = gelu_adapter()
    y, cache = part.forward_fn(x)
    np.testing.assert_allclose(y, jax.nn.gelu(x), rtol=1e-6)

    _, vjp = jax.vjp(jax.nn.gelu, x)
    np.testing.assert_allclose(
        part.backward_fn(cache, error), vjp(error)[0], rtol=1e-5, atol=1e-6
    )


def test_layer_norm_backward_matches_autodiff():
    """The hand-derived LayerNorm backward equals the autodiff of the same forward."""
    rng = np.random.default_rng(1)
    dim = 6
    ln = _random_layer_norm(dim, rng)
    x = jnp.asarray(rng.normal(size=(4, dim)))
    error = jnp.asarray(rng.normal(size=(4, dim)))

    part = layer_norm_adapter(ln)
    y, cache = part.forward_fn(x)
    np.testing.assert_allclose(y, jax.vmap(ln)(x), rtol=1e-5, atol=1e-6)

    _, vjp = jax.vjp(lambda a: jax.vmap(ln)(a), x)
    np.testing.assert_allclose(
        part.backward_fn(cache, error), vjp(error)[0], rtol=1e-4, atol=1e-5
    )


def _parity_ff_net(d: int, h: int, w1: jnp.ndarray, w2: jnp.ndarray) -> DeepNetwork:
    """Build a Linear-GELU-Linear DeepNetwork in backprop-parity configuration.

    Bias-free, volatility frozen, high prior confidence on the hidden and input layers,
    unit precision on the observed output layer.
    """
    high_confidence = dict(
        volatility_parent=False,
        precision=1e4,
        expected_precision=1e4,
    )
    net = (
        DeepNetwork()
        .add_layer(
            size=d,
            add_constant_input=False,
            volatility_parent=False,
        )
        .add_layer(
            size=h,
            add_constant_input=False,
            coupling_fn=jax.nn.gelu,
            **high_confidence,
        )
        .add_layer(size=d, add_constant_input=False, **high_confidence)
    )
    elements = list(net.state.layers)
    elements[1] = dataclasses.replace(elements[1], weights_in=w1)
    elements[2] = dataclasses.replace(elements[2], weights_in=w2)
    net.state = dataclasses.replace(net.state, layers=tuple(elements))
    return net


def test_mixed_pipeline_block_matches_backprop():
    """A Transformer sub-block with mixed parts learns like backpropagation.

    The block is ``output = x + FF(LayerNorm(x))`` — a frozen LayerNorm
    (hand-derived backward), a PyHGF feed-forward in the parity
    configuration, and a residual shortcut, run by the fused executor. One
    training step against squared-error targets must (a) reproduce the
    composed forward function exactly, (b) update the feed-forward weights by
    the batch-averaged backprop gradient, and (c) return the loss gradient at
    the block's *input* — which exercises the residual copy-and-add, the
    LayerNorm formula, and both error-convention conversions in one pass.
    """
    rng = np.random.default_rng(42)
    d, h, batch = 8, 16, 6

    w1 = jnp.asarray(rng.normal(size=(d, h)) * (2.0 / h) ** 0.5)
    w2 = jnp.asarray(rng.normal(size=(h, d)) * (2.0 / d) ** 0.5)
    ln = _random_layer_norm(d, rng)
    xb = jnp.asarray(rng.normal(size=(batch, d)))
    yb = jnp.asarray(rng.normal(size=(batch, d)))

    # --- Oracle: autodiff on the same composed forward function.
    def block_fn(w1_, w2_, x_row):  # one sample
        z = ln(x_row)
        return x_row + w1_ @ jax.nn.gelu(w2_ @ z)

    def mean_loss(w1_, w2_):
        out = jax.vmap(lambda x_row: block_fn(w1_, w2_, x_row))(xb)
        return jnp.mean(jnp.sum(0.5 * (yb - out) ** 2, axis=-1))

    g_w1, g_w2 = jax.grad(mean_loss, argnums=(0, 1))(w1, w2)
    # Per-sample loss gradient at the block input (the message the block
    # should emit).
    per_sample_dx = jax.vmap(
        lambda x_row, y_row: jax.grad(
            lambda a: jnp.sum(0.5 * (y_row - block_fn(w1, w2, a)) ** 2)
        )(x_row)
    )(xb, yb)
    out_oracle = jax.vmap(lambda x_row: block_fn(w1, w2, x_row))(xb)

    # --- The mixed pipeline: one fused step; the default error_fn is
    # ``output - target``, exactly the squared-error loss gradient.
    lr = 1e-3
    ff = DeepNetworkAdapter(
        _parity_ff_net(d, h, w1, w2),
        optimizer=optax.sgd(lr),
        learning_kind="precision_weighted",
    )
    block = Residual(PCSequential([layer_norm_adapter(ln), ff]))
    fused = FusedPipeline(block)

    out, error_in = fused.step(xb, yb)
    fused.merge()

    # (a) Forward parity.
    np.testing.assert_allclose(out, out_oracle, rtol=1e-4, atol=1e-5)

    def norm_rel(a, b):
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    # (b) The weights moved by the batch-averaged backprop gradient.
    d_w1 = -(ff.net.state.layers[1].weights_in - w1) / lr
    d_w2 = -(ff.net.state.layers[2].weights_in - w2) / lr
    assert norm_rel(d_w1, g_w1) < 1e-2
    assert norm_rel(d_w2, g_w2) < 1e-2
    # (c) The emitted input error is the loss gradient at the input.
    assert norm_rel(error_in, per_sample_dx) < 1e-2


def test_step_report_collects_update_and_error_norms():
    """``step_report`` finds learning parts and reports their step magnitudes.

    A two-part pipeline (frozen LayerNorm around a learning feed-forward)
    behind a shortcut junction: after one training step, the report contains
    exactly one entry — the learning part — with positive update and error
    norms and its path inside the model.
    """
    from pyhgf.model import from_feedforward, step_report

    rng = np.random.default_rng(2)
    d, h, batch = 6, 12, 4
    k1, k2 = jax.random.split(jax.random.key(3))
    fc1 = eqx.nn.Linear(d, h, key=k1)
    fc2 = eqx.nn.Linear(h, d, key=k2)
    ln = _random_layer_norm(d, rng)
    x = jnp.asarray(rng.normal(size=(batch, d)))
    target = jnp.asarray(rng.normal(size=(batch, d)))

    ff = DeepNetworkAdapter(from_feedforward(fc1, fc2), optimizer=optax.sgd(1e-3))
    block = Residual(PCSequential([layer_norm_adapter(ln), ff]))
    fused = FusedPipeline(block)

    assert step_report(fused) == []  # nothing has stepped yet

    fused.step(x, target)

    report = step_report(fused)
    assert len(report) == 1
    entry = report[0]
    assert entry["part"] == "Residual.branch.parts[1]"
    assert entry["layer_sizes"] == [d, h, d]
    assert entry["update_norm"] > 0
    assert entry["error_norm"] > 0
