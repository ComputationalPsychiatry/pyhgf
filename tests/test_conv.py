import jax
import jax.numpy as jnp
import numpy as np
import optax

from pyhgf.model import (
    FusedPipeline,
    avg_pool_adapter,
    conv_block,
    conv_output_shape,
    conv_patch_network,
    im2col_adapter,
    max_pool_adapter,
    spatial_reshape_adapter,
)

# Automatic differentiation appears in this file ONLY as a test oracle: the
# hand-derived im2col/pool backward formulas, and the parity-recipe learning
# step, must match what autodiff computes on the same forward functions. The
# adapters themselves never use autodiff.

_PARITY = dict(
    volatility_parent=False,
    precision=1e4,
    expected_precision=1e4,
)
_PARITY_LEAF = dict(volatility_parent=False)


def _norm_rel(a, b) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# ---------------------------------------------------------------------------
# 1. im2col fold formula vs. autodiff oracle
# ---------------------------------------------------------------------------


def test_im2col_backward_matches_autodiff():
    """The hand-derived fold (col2im) equals the autodiff of the same forward."""
    rng = np.random.default_rng(0)
    batch, channels, height, width = 2, 4, 7, 9
    x = jnp.asarray(rng.normal(size=(batch, channels, height, width)))

    part = im2col_adapter(filter_shape=(3, 3), strides=(2, 2), padding="SAME")
    flat, cache = part.forward_fn(x)
    error = jnp.asarray(rng.normal(size=flat.shape))

    def fwd(x_):
        y, _ = part.forward_fn(x_)
        return y

    _, vjp = jax.vjp(fwd, x)
    np.testing.assert_allclose(
        part.backward_fn(cache, error), vjp(error)[0], rtol=1e-4, atol=1e-5
    )


def test_im2col_backward_matches_autodiff_valid_padding():
    """Same check under VALID padding, non-square input, unit stride."""
    rng = np.random.default_rng(1)
    batch, channels, height, width = 2, 3, 8, 10
    x = jnp.asarray(rng.normal(size=(batch, channels, height, width)))

    part = im2col_adapter(filter_shape=(3, 3), strides=(1, 1), padding="VALID")
    flat, cache = part.forward_fn(x)
    error = jnp.asarray(rng.normal(size=flat.shape))

    def fwd(x_):
        y, _ = part.forward_fn(x_)
        return y

    _, vjp = jax.vjp(fwd, x)
    np.testing.assert_allclose(
        part.backward_fn(cache, error), vjp(error)[0], rtol=1e-4, atol=1e-5
    )


def test_avg_pool_backward_matches_autodiff():
    """The hand-derived uniform-split backward equals the autodiff of block-mean."""
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(size=(2, 128, 16, 16)))

    part = avg_pool_adapter(pool_size=(2, 2))
    y, cache = part.forward_fn(x)
    error = jnp.asarray(rng.normal(size=y.shape))

    def fwd(x_):
        y_, _ = part.forward_fn(x_)
        return y_

    _, vjp = jax.vjp(fwd, x)
    np.testing.assert_allclose(
        part.backward_fn(cache, error), vjp(error)[0], rtol=1e-4, atol=1e-5
    )


def test_max_pool_backward_matches_autodiff():
    """The hand-derived winner-take-all backward equals the autodiff of block-max."""
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(2, 8, 16, 16)))

    part = max_pool_adapter(pool_size=(2, 2))
    y, cache = part.forward_fn(x)
    error = jnp.asarray(rng.normal(size=y.shape))

    def fwd(x_):
        y_, _ = part.forward_fn(x_)
        return y_

    _, vjp = jax.vjp(fwd, x)
    np.testing.assert_allclose(
        part.backward_fn(cache, error), vjp(error)[0], rtol=1e-5, atol=1e-6
    )


def test_max_pool_forward_matches_reduce_window():
    """Sanity check: the tap-stack max equals jax.lax.reduce_window's block-max."""
    rng = np.random.default_rng(8)
    x = jnp.asarray(rng.normal(size=(2, 4, 10, 10)))

    part = max_pool_adapter(pool_size=(2, 2))
    y, _ = part.forward_fn(x)
    ref = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID"
    )
    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Wiring gate: im2col + per-patch matmul reproduces a real convolution
# ---------------------------------------------------------------------------


def test_conv_block_matches_real_convolution():
    """A frozen conv_block, given a transplanted kernel, equals jax.lax convolution.

    Validates the whole decomposition end to end (im2col -> shared linear kernel ->
    reshape -> pool) against ground truth, independent of any PyHGF-specific machinery —
    the same "wiring gate" discipline as ``tests/test_hybrid.py``'s forward-parity
    checks.
    """
    import dataclasses

    rng = np.random.default_rng(3)
    batch, in_ch, out_ch, height, width = 2, 3, 5, 10, 10
    kh, kw = 3, 3
    x = jnp.asarray(rng.normal(size=(batch, in_ch, height, width)))
    kernel = jnp.asarray(rng.normal(size=(out_ch, in_ch, kh, kw)) * 0.1)
    bias = jnp.asarray(rng.normal(size=(out_ch,)) * 0.1)

    part, out_shape = conv_block(
        in_channels=in_ch,
        out_channels=out_ch,
        in_height=height,
        in_width=width,
        filter_shape=(kh, kw),
        padding="SAME",
        pool_size=(1, 1),  # disable pooling to isolate the conv itself
    )
    # part.parts: [im2col, DeepNetworkAdapter(patch_net), gelu, reshape, pool]
    adapter = part.parts[1]
    elements = list(adapter.net.state.layers)
    flat_kernel = kernel.reshape(out_ch, in_ch * kh * kw)
    weights = jnp.concatenate([flat_kernel, bias[:, None]], axis=1)
    elements[1] = dataclasses.replace(elements[1], weights_mean=weights)
    adapter.net.state = dataclasses.replace(adapter.net.state, layers=tuple(elements))

    fused = FusedPipeline(part)
    out = fused.predict(x)

    ref = (
        jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        + bias[None, :, None, None]
    )
    ref = jax.nn.gelu(ref)  # conv_block applies GELU after the linear map

    assert out_shape == (out_ch, height, width)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_conv_output_shape_matches_actual_pipeline():
    """conv_output_shape's prediction matches what im2col_adapter actually produces."""
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=(1, 4, 17, 13)))
    for padding in ("SAME", "VALID"):
        part = im2col_adapter(filter_shape=(3, 3), strides=(2, 2), padding=padding)
        flat, _ = part.forward_fn(x)
        h_out, w_out = conv_output_shape(17, 13, (3, 3), (2, 2), padding)
        assert flat.shape[1] == h_out * w_out


# ---------------------------------------------------------------------------
# 3. Parity gate: pinned-precision conv_block matches a backprop twin
# ---------------------------------------------------------------------------


def test_conv_block_parity_matches_backprop():
    """One training step of a parity-recipe conv_block matches backprop's gradient.

    A single small conv layer (no pooling, to keep the oracle simple), trained
    for one step with the parity recipe: the weight (kernel+bias) delta and the
    error emitted at the input must match ``jax.grad`` of an equivalent plain
    convolution, at the same tolerance used by the library's own parity tests
    (``tests/test_fused.py``, ``tests/test_hybrid.py``).
    """
    import dataclasses

    rng = np.random.default_rng(5)
    batch, in_ch, out_ch, height, width = 4, 2, 3, 6, 6
    kh, kw = 3, 3
    x = jnp.asarray(rng.normal(size=(batch, in_ch, height, width)))
    kernel0 = jnp.asarray(rng.normal(size=(out_ch, in_ch, kh, kw)) * 0.3)
    bias0 = jnp.asarray(rng.normal(size=(out_ch,)) * 0.1)
    target = jnp.asarray(rng.normal(size=(batch, out_ch, height, width)))

    lr = 1e-3
    part, out_shape = conv_block(
        in_channels=in_ch,
        out_channels=out_ch,
        in_height=height,
        in_width=width,
        filter_shape=(kh, kw),
        padding="SAME",
        pool_size=(1, 1),
        optimizer=optax.sgd(lr),
        leaf_kwargs=_PARITY_LEAF,
        layer_kwargs=_PARITY,
    )
    adapter = part.parts[1]
    elements = list(adapter.net.state.layers)
    flat_kernel0 = kernel0.reshape(out_ch, in_ch * kh * kw)
    weights0 = jnp.concatenate([flat_kernel0, bias0[:, None]], axis=1)
    elements[1] = dataclasses.replace(elements[1], weights_mean=weights0)
    adapter.net.state = dataclasses.replace(adapter.net.state, layers=tuple(elements))
    adapter.net.opt_state = None

    fused = FusedPipeline(part)
    out, input_error = fused.step(
        x, jax.nn.gelu(target)
    )  # error_fn default: out - target
    fused.merge()

    # --- Oracle: plain conv + GELU trained by autodiff SGD on the same loss.
    def forward(kernel_, bias_, x_):
        y = (
            jax.lax.conv_general_dilated(
                x_,
                kernel_,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            + bias_[None, :, None, None]
        )
        return jax.nn.gelu(y)

    def loss(kernel_, bias_):
        pred = forward(kernel_, bias_, x)
        return jnp.mean(
            jnp.sum(0.5 * (jax.nn.gelu(target) - pred) ** 2, axis=(1, 2, 3))
        )

    g_kernel, g_bias = jax.grad(loss, argnums=(0, 1))(kernel0, bias0)
    # input_errors is per-sample and unaveraged (see _batch_step's docstring),
    # so the oracle must be too -- vmap a per-sample grad of a per-sample SUM
    # loss, not jax.grad of a batch-mean scalar (which would divide by an
    # extra factor of `batch`). Matches tests/test_fused.py's own convention.
    gelu_target = jax.nn.gelu(target)

    def per_sample_loss(x_row, target_row):
        pred_row = forward(kernel0, bias0, x_row[None])[0]
        return jnp.sum(0.5 * (target_row - pred_row) ** 2)

    dx_oracle = jax.vmap(jax.grad(per_sample_loss))(x, gelu_target)

    new_weights = adapter.net.state.layers[1].weights_mean
    new_kernel = new_weights[:, :-1].reshape(out_ch, in_ch, kh, kw)
    new_bias = new_weights[:, -1]
    d_kernel = -(new_kernel - kernel0) / lr
    d_bias = -(new_bias - bias0) / lr

    assert _norm_rel(d_kernel, g_kernel) < 2e-2
    assert _norm_rel(d_bias, g_bias) < 2e-2
    assert _norm_rel(input_error, dx_oracle) < 2e-2


def test_spatial_reshape_roundtrips():
    """spatial_reshape_adapter's forward/backward are exact inverses of each other."""
    rng = np.random.default_rng(6)
    batch, channels, height, width = 2, 5, 4, 4
    x_patches = jnp.asarray(rng.normal(size=(batch, height * width, channels)))

    part = spatial_reshape_adapter(height, width)
    spatial, cache = part.forward_fn(x_patches)
    assert spatial.shape == (batch, channels, height, width)

    back = part.backward_fn(cache, spatial)
    np.testing.assert_allclose(back, x_patches, rtol=1e-6, atol=1e-6)


def test_conv_patch_network_shapes():
    """conv_patch_network builds a network with the requested in/out sizes."""
    net = conv_patch_network(in_features=27, out_channels=16, key=jax.random.key(0))
    assert net.layer_sizes == [16, 27]
    assert net.state.layers[1].weights_mean.shape == (16, 28)  # +1 bias column
