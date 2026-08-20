import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pyhgf.model import FusedPipeline
from pyhgf.model.conv import (
    avg_pool_adapter,
    conv_block,
    conv_output_shape,
    conv_patch_network,
    flatten_adapter,
    im2col_adapter,
    max_pool_adapter,
    spatial_reshape_adapter,
)
from pyhgf.model.transplant import from_conv

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


def test_flatten_adapter_roundtrips():
    """flatten_adapter's backward restores exactly the layout its forward consumed."""
    rng = np.random.default_rng(9)
    batch, channels, height, width = 3, 4, 5, 6
    x = jnp.asarray(rng.normal(size=(batch, channels, height, width)))

    part = flatten_adapter(channels, height, width)
    flat, cache = part.forward_fn(x)
    assert flat.shape == (batch, channels * height * width)

    error = jnp.asarray(rng.normal(size=flat.shape))
    back = part.backward_fn(cache, error)
    assert back.shape == x.shape

    def fwd(x_):
        y_, _ = part.forward_fn(x_)
        return y_

    _, vjp = jax.vjp(fwd, x)
    np.testing.assert_allclose(back, vjp(error)[0], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Transplanting an externally-trained kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_bias", [True, False])
def test_from_conv_reproduces_the_kernel_it_was_built_from(with_bias):
    """A from_conv network, placed in a conv_block, equals jax.lax convolution.

    Covers both bias conventions: with a bias the input layer carries a constant
    node holding it, without one it does not, and the patch row the shared kernel
    multiplies is the same width either way.
    """
    rng = np.random.default_rng(10)
    batch, in_ch, out_ch, height, width = 2, 3, 4, 8, 8
    kh, kw = 3, 3
    x = jnp.asarray(rng.normal(size=(batch, in_ch, height, width)))
    kernel = jnp.asarray(rng.normal(size=(out_ch, in_ch, kh, kw)) * 0.1)
    bias = jnp.asarray(rng.normal(size=(out_ch,)) * 0.1) if with_bias else None

    net = from_conv(kernel, bias)
    assert net.layer_sizes == [out_ch, in_ch * kh * kw]

    part, out_shape = conv_block(
        in_channels=in_ch,
        out_channels=out_ch,
        in_height=height,
        in_width=width,
        filter_shape=(kh, kw),
        padding="SAME",
        pool_size=(1, 1),  # disable pooling to isolate the conv itself
        patch_net=net,
    )
    out = FusedPipeline(part).predict(x)

    ref = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    if bias is not None:
        ref = ref + bias[None, :, None, None]
    ref = jax.nn.gelu(ref)  # conv_block applies GELU after the linear map

    assert out_shape == (out_ch, height, width)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_from_conv_flattens_channel_major_then_spatial():
    """from_conv's flattening matches im2col_adapter's patch column order.

    A kernel that reads exactly one (channel, row, column) tap must produce, for every
    patch, the value im2col put in the matching column. This pins the layout convention
    itself rather than an end-to-end sum that could hide a permutation.
    """
    in_ch, kh, kw = 3, 3, 3
    c, i, j = 2, 0, 1
    kernel = np.zeros((1, in_ch, kh, kw), dtype="float32")
    kernel[0, c, i, j] = 1.0

    net = from_conv(jnp.asarray(kernel))
    weights = np.asarray(net.state.layers[1].weights_mean)
    assert weights.shape == (1, in_ch * kh * kw)
    assert np.flatnonzero(weights[0]).tolist() == [c * kh * kw + i * kw + j]


# ---------------------------------------------------------------------------
# 5. Where the shared-kernel gradient correction has to live
# ---------------------------------------------------------------------------


def _transplanted_block(kernel0, bias0, height, width, **kwargs):
    """Build a conv_block holding ``kernel0``/``bias0``, with pooling disabled."""
    import dataclasses

    out_ch, in_ch, kh, kw = kernel0.shape
    part, _ = conv_block(
        in_channels=in_ch,
        out_channels=out_ch,
        in_height=height,
        in_width=width,
        filter_shape=(kh, kw),
        padding="SAME",
        pool_size=(1, 1),
        leaf_kwargs=_PARITY_LEAF,
        layer_kwargs=_PARITY,
        **kwargs,
    )
    adapter = part.parts[1]
    elements = list(adapter.net.state.layers)
    weights0 = jnp.concatenate(
        [kernel0.reshape(out_ch, in_ch * kh * kw), bias0[:, None]], axis=1
    )
    elements[1] = dataclasses.replace(elements[1], weights_mean=weights0)
    adapter.net.state = dataclasses.replace(adapter.net.state, layers=tuple(elements))
    adapter.net.opt_state = None
    return part, adapter, weights0


def _conv_gelu_gradient(kernel0, bias0, x, target):
    """jax.grad of a plain convolution + GELU on the batch-mean squared error."""

    def loss(kernel_, bias_):
        y = (
            jax.lax.conv_general_dilated(
                x,
                kernel_,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            + bias_[None, :, None, None]
        )
        return jnp.mean(jnp.sum(0.5 * (target - jax.nn.gelu(y)) ** 2, axis=(1, 2, 3)))

    return jax.grad(loss, argnums=(0, 1))(kernel0, bias0)


def test_conv_block_parity_matches_backprop_under_adam():
    """One Adam step of a conv_block matches Adam applied to the true conv gradient.

    The SGD parity test pins the gradient's *magnitude*; Adam normalises a uniform
    factor away, so this pins its *structure* instead: the same optimiser fed the
    ``jax.grad`` gradient of an equivalent plain convolution must land on the same
    weights. A gradient wrong in anything but a single overall scale fails here.
    """
    rng = np.random.default_rng(11)
    batch, in_ch, out_ch, height, width = 4, 2, 3, 6, 6
    kh, kw = 3, 3
    x = jnp.asarray(rng.normal(size=(batch, in_ch, height, width)))
    kernel0 = jnp.asarray(rng.normal(size=(out_ch, in_ch, kh, kw)) * 0.3)
    bias0 = jnp.asarray(rng.normal(size=(out_ch,)) * 0.1)
    target = jax.nn.gelu(jnp.asarray(rng.normal(size=(batch, out_ch, height, width))))

    lr = 1e-3
    part, adapter, weights0 = _transplanted_block(
        kernel0, bias0, height, width, optimizer=optax.adam(lr)
    )
    fused = FusedPipeline(part)
    fused.step(x, target)  # error_fn default: out - target
    fused.merge()

    g_kernel, g_bias = _conv_gelu_gradient(kernel0, bias0, x, target)
    g_matrix = jnp.concatenate(
        [g_kernel.reshape(out_ch, in_ch * kh * kw), g_bias[:, None]], axis=1
    )
    oracle_opt = optax.adam(lr)
    updates, _ = oracle_opt.update(g_matrix, oracle_opt.init(weights0), weights0)
    expected = optax.apply_updates(weights0, updates)

    assert _norm_rel(adapter.net.state.layers[1].weights_mean, expected) < 2e-2


def test_adam_absorbs_the_shared_kernel_reuse_factor():
    """Adam is insensitive to ``weight_reuse``, so SGD is what pins its value.

    ``conv_block`` multiplies the mean gradient by the patch count to recover the sum a
    shared kernel's true gradient takes over its spatial applications. Adam divides by
    the gradient's own root-mean-square, so any uniform factor cancels. This is a
    property of Adam rather than of the correction, and it is asserted here so that a
    future reader does not add a second compensation for it.
    """
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.normal(size=(4, 1, 6, 6)))
    target = jnp.asarray(rng.normal(size=(4, 2, 6, 6)))
    kernel0 = jnp.asarray(rng.normal(size=(2, 1, 3, 3)) * 0.3)
    bias0 = jnp.asarray(rng.normal(size=(2,)) * 0.1)

    def step(weight_reuse, **kwargs):
        part, adapter, weights0 = _transplanted_block(kernel0, bias0, 6, 6, **kwargs)
        adapter.weight_reuse = float(weight_reuse)
        fused = FusedPipeline(part)
        fused.step(x, target)
        fused.merge()
        return np.asarray(adapter.net.state.layers[1].weights_mean - weights0)

    n_patches = 36
    adam_1 = step(1.0, optimizer=optax.adam(1e-3))
    adam_n = step(n_patches, optimizer=optax.adam(1e-3))
    np.testing.assert_allclose(adam_n, adam_1, rtol=1e-4, atol=1e-9)

    # Under SGD the same factor passes straight through, which is the regime the
    # correction exists for.
    sgd_1 = step(1.0, optimizer=optax.sgd(1e-3))
    sgd_n = step(n_patches, optimizer=optax.sgd(1e-3))
    np.testing.assert_allclose(sgd_n, sgd_1 * n_patches, rtol=1e-3, atol=1e-9)


def _belief_step(kernel0, bias0, weight_reuse, steps=1, window=100):
    """Run weight-belief steps with no optimiser, returning mean and precision."""
    rng = np.random.default_rng(13)
    x = jnp.asarray(rng.normal(size=(4, 1, 6, 6)))
    target = jnp.asarray(rng.normal(size=(4, 2, 6, 6)))

    part, adapter, weights0 = _transplanted_block(
        kernel0,
        bias0,
        6,
        6,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": window},
    )
    adapter.weight_reuse = float(weight_reuse)
    assert adapter.optimizer is None
    fused = FusedPipeline(part)
    for _ in range(steps):
        fused.step(x, target)
        fused.merge()
    assert adapter.net.opt_state is None  # no optimiser state was ever built
    layer = adapter.net.state.layers[1]
    return (
        np.asarray(layer.weights_mean - weights0),
        np.asarray(layer.weights_precision_delta),
    )


def test_weight_belief_rule_learns_without_any_optimiser():
    """The belief rule carries both halves of the step with ``optimizer=None``.

    Its step size is each weight's own belief variance, so the kernel has to keep
    learning with no optimiser present, and the accumulated precision has to grow
    monotonically as evidence arrives rather than staying at its prior value.
    """
    rng = np.random.default_rng(14)
    kernel0 = jnp.asarray(rng.normal(size=(2, 1, 3, 3)) * 0.3)
    bias0 = jnp.asarray(rng.normal(size=(2,)) * 0.1)

    delta_1, precision_1 = _belief_step(kernel0, bias0, 1.0, steps=1)
    delta_3, precision_3 = _belief_step(kernel0, bias0, 1.0, steps=3)

    assert np.linalg.norm(delta_1) > 0.0
    assert np.linalg.norm(delta_3) > np.linalg.norm(delta_1)
    assert np.max(precision_1) > 0.0
    assert np.max(precision_3) > np.max(precision_1)


def test_weight_reuse_scales_both_halves_of_the_belief_step():
    """``weight_reuse`` reaches the importance increment, not only the gradient.

    A shared kernel applied at ``k`` positions accumulates ``k`` times the curvature per
    sample, exactly as it accumulates ``k`` times the gradient. Scaling only the
    gradient would move the mean ``k`` times faster while the belief tightened at the
    one-use rate; both halves have to carry the factor.
    """
    rng = np.random.default_rng(15)
    kernel0 = jnp.asarray(rng.normal(size=(2, 1, 3, 3)) * 0.3)
    bias0 = jnp.asarray(rng.normal(size=(2,)) * 0.1)

    n_patches = 36.0
    _, precision_1 = _belief_step(kernel0, bias0, 1.0, steps=1)
    delta_n, precision_n = _belief_step(kernel0, bias0, n_patches, steps=1)

    # One step from a fresh install: the increment is the importance itself, so it
    # scales exactly with the reuse count.
    np.testing.assert_allclose(precision_n, precision_1 * n_patches, rtol=1e-3)
    assert np.linalg.norm(delta_n) > 0.0


def test_conv_block_rejects_a_pool_window_larger_than_its_feature_map():
    """An unsatisfiable pool_size raises instead of yielding a zero-sized map."""
    with pytest.raises(ValueError, match="does not fit"):
        conv_block(
            in_channels=1,
            out_channels=2,
            in_height=3,
            in_width=3,
            pool_size=(4, 4),
        )
