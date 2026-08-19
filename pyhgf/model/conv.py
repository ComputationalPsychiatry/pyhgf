"""Convolutional layers for the hybrid pipeline.

A convolution is a weight-shared linear map over patches
(``docs/source/notebooks/0.6-Deep_networks_implementation.md``, §4.6): extracting
patches (im2col) and pooling are frozen, weightless spatial reshuffles with a
closed-form backward; the only learned part is one small :class:`~pyhgf.model.DeepNetwork`
applied identically to every patch, exactly the per-position batching
:class:`~pyhgf.model.hybrid.DeepNetworkAdapter` already uses for token positions
and attention tables.

Patch layout convention: patches are flattened channel-major-then-spatial,
``index = c * kh * kw + i * kw + j`` — matching
``jax.lax.conv_general_dilated_patches`` and a PyTorch-style kernel of shape
``(out_channels, in_channels, kh, kw)`` reshaped the same way.
"""

from __future__ import annotations

from typing import Optional, Union

import jax
import jax.numpy as jnp
import optax

from pyhgf.model.deep_network import DeepNetwork
from pyhgf.model.hybrid import (
    DeepNetworkAdapter,
    EquinoxAdapter,
    PCSequential,
    gelu_adapter,
)

__all__ = [
    "im2col_adapter",
    "avg_pool_adapter",
    "max_pool_adapter",
    "spatial_reshape_adapter",
    "flatten_adapter",
    "conv_patch_network",
    "conv_block",
    "conv_output_shape",
]

Padding = Union[str, tuple[tuple[int, int], tuple[int, int]]]


def _same_padding(in_size: int, kernel_size: int, stride: int) -> tuple[int, int]:
    """One axis of TF/JAX-style ``"SAME"`` padding: out = ceil(in / stride)."""
    out_size = -(-in_size // stride)
    total = max((out_size - 1) * stride + kernel_size - in_size, 0)
    lo = total // 2
    return lo, total - lo


def _resolve_padding(
    padding: Padding, in_h: int, in_w: int, kh: int, kw: int, sh: int, sw: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if padding == "SAME":
        return _same_padding(in_h, kh, sh), _same_padding(in_w, kw, sw)
    if padding == "VALID":
        return (0, 0), (0, 0)
    if isinstance(padding, str):
        raise ValueError(
            f"padding must be 'SAME', 'VALID' or an explicit pair, got {padding!r}"
        )
    return padding


def conv_output_shape(
    in_height: int,
    in_width: int,
    filter_shape: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    padding: Padding = "SAME",
) -> tuple[int, int]:
    """Return the ``(height, width)`` a convolution produces, without running it."""
    kh, kw = filter_shape
    sh, sw = strides
    (ph_lo, ph_hi), (pw_lo, pw_hi) = _resolve_padding(
        padding, in_height, in_width, kh, kw, sh, sw
    )
    h_out = (in_height + ph_lo + ph_hi - kh) // sh + 1
    w_out = (in_width + pw_lo + pw_hi - kw) // sw + 1
    return h_out, w_out


def im2col_adapter(
    filter_shape: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    padding: Padding = "SAME",
) -> EquinoxAdapter:
    """Build a frozen patch extractor: images in, one row per patch out.

    Forward, ``x`` of shape ``(batch, C, H, W)`` becomes ``(batch, n_patches, C*kh*kw)``
    via ``jax.lax.conv_general_dilated_patches`` — the same "one row per sample" layout
    :class:`~pyhgf.model.hybrid.DeepNetworkAdapter` already expects for token positions,
    so the following per-patch network needs no special-casing. Backward (fold / col2im)
    is the geometric adjoint: each of the ``kh*kw`` kernel taps is a strided slice of
    the (padded) image, so its gradient is scattered back with
    :func:`jax.numpy.ndarray.at.add` at the same stride — a closed-form sum over
    overlapping patches, unrolled over the (small, static) kernel taps. Verified against
    ``jax.vjp`` of this forward as a test oracle (see ``tests/test_conv.py``), never at
    call time.
    """
    kh, kw = filter_shape
    sh, sw = strides

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        batch, channels, height, width = x.shape
        ph, pw = _resolve_padding(padding, height, width, kh, kw, sh, sw)
        patches = jax.lax.conv_general_dilated_patches(
            x, filter_shape=(kh, kw), window_strides=(sh, sw), padding=[ph, pw]
        )  # (batch, C*kh*kw, h_out, w_out)
        _, ckk, h_out, w_out = patches.shape
        flat = patches.reshape(batch, ckk, h_out * w_out).transpose(0, 2, 1)
        return flat, (x.shape, h_out, w_out, ph, pw)

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        x_shape, h_out, w_out, ph, pw = cache
        batch, channels, height, width = x_shape
        e = error.transpose(0, 2, 1).reshape(batch, channels, kh, kw, h_out, w_out)
        h_pad = height + ph[0] + ph[1]
        w_pad = width + pw[0] + pw[1]
        grad_padded = jnp.zeros((batch, channels, h_pad, w_pad), dtype=error.dtype)
        for i in range(kh):
            for j in range(kw):
                h_end = i + sh * (h_out - 1) + 1
                w_end = j + sw * (w_out - 1) + 1
                grad_padded = grad_padded.at[:, :, i:h_end:sh, j:w_end:sw].add(
                    e[:, :, i, j, :, :]
                )
        return grad_padded[:, :, ph[0] : ph[0] + height, pw[0] : pw[0] + width]

    return EquinoxAdapter(forward, backward)


def avg_pool_adapter(
    pool_size: tuple[int, int] = (2, 2),
    stride: Optional[tuple[int, int]] = None,
) -> EquinoxAdapter:
    """Build a frozen average-pool: block-mean forward, uniform-split backward.

    Forward, each non-overlapping (or strided) window is averaged via
    ``jax.lax.reduce_window``. Backward, the incoming gradient at one pooled position is
    split evenly back across every position that fed it — exactly linear, same closed-
    form spirit as :func:`~pyhgf.model.hybrid.linear_adapter`. Chosen over max-pool for
    a discontinuity-free backward.
    """
    ph, pw = pool_size
    sh, sw = stride if stride is not None else pool_size

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        batch, channels, height, width = x.shape
        h_out = (height - ph) // sh + 1
        w_out = (width - pw) // sw + 1
        y = jax.lax.reduce_window(
            x, 0.0, jax.lax.add, (1, 1, ph, pw), (1, 1, sh, sw), "VALID"
        ) / (ph * pw)
        return y, (x.shape, h_out, w_out)

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        x_shape, h_out, w_out = cache
        scale = 1.0 / (ph * pw)
        grad = jnp.zeros(x_shape, dtype=error.dtype)
        for i in range(ph):
            for j in range(pw):
                h_end = i + sh * (h_out - 1) + 1
                w_end = j + sw * (w_out - 1) + 1
                grad = grad.at[:, :, i:h_end:sh, j:w_end:sw].add(error * scale)
        return grad

    return EquinoxAdapter(forward, backward)


def max_pool_adapter(
    pool_size: tuple[int, int] = (2, 2),
    stride: Optional[tuple[int, int]] = None,
) -> EquinoxAdapter:
    """Build a frozen max-pool: block-max forward, winner-take-all backward.

    Forward, each non-overlapping (or strided) window's taps are gathered as an explicit
    stack (same strided-slice-per-tap style as :func:`avg_pool_adapter`'s backward) and
    reduced with ``jnp.max``, the winning tap index cached alongside. Backward routes
    the incoming gradient only to the tap that won each window (``jnp.argmax``'s lowest-
    index tie-break — matching common max-pool convention, and inconsequential for
    continuous post-GELU activations), zero everywhere else — a closed-form scatter, not
    the discontinuity-through-differentiation a subgradient would need.
    """
    ph, pw = pool_size
    sh, sw = stride if stride is not None else pool_size
    n_taps = ph * pw

    def _taps(x: jnp.ndarray, h_out: int, w_out: int) -> jnp.ndarray:
        rows = []
        for i in range(ph):
            for j in range(pw):
                h_end = i + sh * (h_out - 1) + 1
                w_end = j + sw * (w_out - 1) + 1
                rows.append(x[:, :, i:h_end:sh, j:w_end:sw])
        return jnp.stack(rows, axis=0)  # (n_taps, batch, channels, h_out, w_out)

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        batch, channels, height, width = x.shape
        h_out = (height - ph) // sh + 1
        w_out = (width - pw) // sw + 1
        stacked = _taps(x, h_out, w_out)
        y = jnp.max(stacked, axis=0)
        winner = jnp.argmax(stacked, axis=0)  # (batch, channels, h_out, w_out)
        return y, (x.shape, h_out, w_out, winner)

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        x_shape, h_out, w_out, winner = cache
        grad = jnp.zeros(x_shape, dtype=error.dtype)
        idx = 0
        for i in range(ph):
            for j in range(pw):
                h_end = i + sh * (h_out - 1) + 1
                w_end = j + sw * (w_out - 1) + 1
                grad = grad.at[:, :, i:h_end:sh, j:w_end:sw].add(
                    jnp.where(winner == idx, error, 0.0)
                )
                idx += 1
        return grad

    return EquinoxAdapter(forward, backward)


def spatial_reshape_adapter(height: int, width: int) -> EquinoxAdapter:
    """Build a frozen reshape between patch rows and the spatial layout.

    Maps ``(batch, n_patches, C)`` to ``(batch, C, H, W)`` and back, converting the
    output of the per-patch network into the spatial layout that pooling and the next
    :func:`im2col_adapter` expect. A pure transpose plus reshape — no learned content,
    backward is just the inverse reshape.
    """

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        batch, n_patches, channels = x.shape
        spatial = x.transpose(0, 2, 1).reshape(batch, channels, height, width)
        return spatial, (channels,)

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        (channels,) = cache
        batch = error.shape[0]
        return error.reshape(batch, channels, height * width).transpose(0, 2, 1)

    return EquinoxAdapter(forward, backward)


def flatten_adapter(channels: int, height: int, width: int) -> EquinoxAdapter:
    """Build a frozen flatten: spatial ``(batch, C, H, W)`` -> ``(batch, C*H*W)``."""

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        batch = x.shape[0]
        return x.reshape(batch, channels * height * width), ()

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        batch = error.shape[0]
        return error.reshape(batch, channels, height, width)

    return EquinoxAdapter(forward, backward)


def conv_patch_network(
    in_features: int,
    out_channels: int,
    key: Optional[jax.Array] = None,
    strategy: str = "he",
    leaf_kwargs: Optional[dict] = None,
    layer_kwargs: Optional[dict] = None,
    network_kwargs: Optional[dict] = None,
) -> DeepNetwork:
    """Build a fresh 2-layer :class:`DeepNetwork` computing one conv kernel's map.

    Same shape as :func:`~pyhgf.model.transplant.from_linear` (one weight matrix,
    bias folded in as the input layer's constant node) but freshly initialised
    rather than transplanted — this *is* the conv kernel, shared across every
    patch via :class:`~pyhgf.model.hybrid.DeepNetworkAdapter`'s per-row batching.

    Parameters
    ----------
    in_features :
        ``in_channels * kh * kw``, the flattened patch size.
    out_channels :
        Number of output feature maps this kernel produces.
    key :
        PRNG key for weight initialisation.
    strategy :
        Forwarded to :meth:`DeepNetwork.weight_initialisation` (default ``"he"``,
        appropriate for a GELU-coupled conv stack).
    leaf_kwargs :
        Extra ``add_layer`` kwargs for the bottom (output) layer — e.g. the
        backprop-parity configuration.
    layer_kwargs :
        Extra ``add_layer`` kwargs for the top (input) layer.
    network_kwargs :
        Constructor arguments for the :class:`DeepNetwork` itself, such as
        ``feedforward_uncertainty``. These reach the network's state when it is
        built and cannot be set afterwards, so they belong here rather than in
        the per-layer keyword sets.
    """
    net = (
        DeepNetwork(**(network_kwargs or {}))
        .add_layer(size=out_channels, add_constant_input=False, **(leaf_kwargs or {}))
        .add_layer(size=in_features, add_constant_input=True, **(layer_kwargs or {}))
    )
    return net.weight_initialisation(strategy, key=key)


def conv_block(
    in_channels: int,
    out_channels: int,
    in_height: int,
    in_width: int,
    filter_shape: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
    padding: Padding = "SAME",
    pool_size: tuple[int, int] = (2, 2),
    pool_stride: Optional[tuple[int, int]] = None,
    pool_kind: str = "avg",
    optimizer: Optional[optax.GradientTransformation] = None,
    learning_kind: str = "precision_weighted",
    learning_kwargs: Optional[dict] = None,
    update_precisions: bool = False,
    time_step: float = 1.0,
    key: Optional[jax.Array] = None,
    leaf_kwargs: Optional[dict] = None,
    layer_kwargs: Optional[dict] = None,
    network_kwargs: Optional[dict] = None,
    patch_net: Optional[DeepNetwork] = None,
) -> tuple[PCSequential, tuple[int, int, int]]:
    """Assemble one conv layer as a part tree: im2col -> shared kernel -> GELU -> pool.

    A conv layer has exactly one weight table (unlike attention's four), so
    plain :class:`~pyhgf.model.hybrid.PCSequential` composition of existing
    primitives is enough — no bespoke composite class needed.

    Parameters
    ----------
    pool_kind :
        ``"avg"`` (default, backward-compatible) for :func:`avg_pool_adapter`
        or ``"max"`` for :func:`max_pool_adapter`.
    learning_kind :
        Weight-gradient mode, forwarded to
        :class:`~pyhgf.model.hybrid.DeepNetworkAdapter`. Note that
        ``"synaptic_uncertainty"`` steps each weight by its own belief variance
        and never calls the optimiser, so the ``n_patches`` gradient
        compensation below — which rides on an :mod:`optax` transformation —
        does not apply to it.
    learning_kwargs :
        Settings of the learning rule, used by
        ``learning_kind="synaptic_uncertainty"``.
    network_kwargs :
        Constructor arguments for the shared-kernel :class:`DeepNetwork`, such
        as ``feedforward_uncertainty``. Ignored when ``patch_net`` is given,
        since that network is already built.
    patch_net :
        A pre-built shared-kernel network to use in place of a freshly
        initialised :func:`conv_patch_network` — e.g. one built by
        :func:`~pyhgf.model.transplant.from_conv` to transplant a trained or
        externally-initialised kernel. When given, ``key`` is ignored.

    Returns
    -------
    part :
        The ``PCSequential`` to place in a larger pipeline.
    out_shape :
        ``(out_channels, pooled_height, pooled_width)`` — feed forward into the
        next block's ``in_channels``/``in_height``/``in_width``.
    """
    kh, kw = filter_shape
    h_out, w_out = conv_output_shape(
        in_height, in_width, filter_shape, strides, padding
    )
    in_features = in_channels * kh * kw
    if patch_net is None:
        patch_net = conv_patch_network(
            in_features,
            out_channels,
            key=key,
            leaf_kwargs=leaf_kwargs,
            layer_kwargs=layer_kwargs,
            network_kwargs=network_kwargs,
        )

    ph, pw = pool_size
    sh, sw = pool_stride if pool_stride is not None else pool_size
    pooled_h = (h_out - ph) // sh + 1
    pooled_w = (w_out - pw) // sw + 1

    # DeepNetworkAdapter/_batch_step averages the weight gradient over every
    # row it sees (docstring: "invariant to repeating the batch"). It sees
    # batch*n_patches rows here, so its mean divides by n_patches too many
    # times relative to a true convolution's gradient, which *sums* across
    # the n_patches spatial applications of the shared kernel and averages
    # only over the batch. Compensate by pre-scaling what the optimizer sees
    # by n_patches (verified against a jax.grad oracle in test_conv.py).
    n_patches = h_out * w_out
    if optimizer is not None:
        optimizer = optax.chain(optax.scale(n_patches), optimizer)

    if pool_kind == "avg":
        pool = avg_pool_adapter(pool_size, (sh, sw))
    elif pool_kind == "max":
        pool = max_pool_adapter(pool_size, (sh, sw))
    else:
        raise ValueError(f"pool_kind must be 'avg' or 'max', got {pool_kind!r}")

    part = PCSequential([
        im2col_adapter(filter_shape, strides, padding),
        DeepNetworkAdapter(
            patch_net,
            optimizer=optimizer,
            learning_kind=learning_kind,
            learning_kwargs=learning_kwargs,
            update_precisions=update_precisions,
            time_step=time_step,
        ),
        gelu_adapter(),
        spatial_reshape_adapter(h_out, w_out),
        pool,
    ])
    return part, (out_channels, pooled_h, pooled_w)
