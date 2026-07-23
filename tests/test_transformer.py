# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from pyhgf.model import (
    DeepNetworkAdapter,
    FusedPipeline,
    MultiHeadAttention,
    from_feedforward,
    hybrid_from_gpt,
    linear_adapter,
)

# Automatic differentiation appears in this file ONLY as a test oracle. The
# reference Equinox model below mirrors the architecture of
# docs/source/notebooks/0.8-Transformers.ipynb.


class EqxAttention(eqx.Module):
    """Reference causal multi-head self-attention, plain Equinox/autodiff."""

    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim, n_heads, key):
        k1, k2, k3, k4 = random.split(key, 4)
        self.wq = eqx.nn.Linear(dim, dim, use_bias=False, key=k1)
        self.wk = eqx.nn.Linear(dim, dim, use_bias=False, key=k2)
        self.wv = eqx.nn.Linear(dim, dim, use_bias=False, key=k3)
        self.wo = eqx.nn.Linear(dim, dim, use_bias=False, key=k4)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x):  # x: (T, D)
        """Apply causal self-attention to a single sequence."""
        seq_len, dim = x.shape
        q, k, v = jax.vmap(self.wq)(x), jax.vmap(self.wk)(x), jax.vmap(self.wv)(x)
        split = lambda a: a.reshape(seq_len, self.n_heads, self.head_dim).transpose(
            1, 0, 2
        )
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask[None] == 0, -jnp.inf, scores)
        attn = jax.nn.softmax(scores, axis=-1)
        out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, dim)
        return jax.vmap(self.wo)(out)


class EqxFeedForward(eqx.Module):
    """Reference two-layer GELU feed-forward, plain Equinox/autodiff."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim, hidden, key):
        k1, k2 = random.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x):  # x: (T, D)
        """Apply the feed-forward block to a single sequence."""
        return jax.vmap(self.fc2)(jax.nn.gelu(jax.vmap(self.fc1)(x)))


class EqxBlock(eqx.Module):
    """Reference pre-norm Transformer block (attention + feed-forward)."""

    attn: EqxAttention
    ff: EqxFeedForward
    n1: eqx.nn.LayerNorm
    n2: eqx.nn.LayerNorm

    def __init__(self, dim, n_heads, hidden, key):
        k1, k2 = random.split(key)
        self.attn = EqxAttention(dim, n_heads, k1)
        self.ff = EqxFeedForward(dim, hidden, k2)
        self.n1 = eqx.nn.LayerNorm(dim)
        self.n2 = eqx.nn.LayerNorm(dim)

    def __call__(self, x):  # x: (T, D)
        """Apply the block's attention and feed-forward shortcut junctions."""
        x = x + self.attn(jax.vmap(self.n1)(x))
        x = x + self.ff(jax.vmap(self.n2)(x))
        return x


class EqxGPT(eqx.Module):
    """Reference GPT-style Transformer, plain Equinox/autodiff."""

    tok_emb: eqx.nn.Embedding
    pos_emb: eqx.nn.Embedding
    blocks: list
    norm_f: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    context_length: int = eqx.field(static=True)

    def __init__(self, vocab_size, dim, n_heads, hidden, n_layers, context_length, key):
        keys = random.split(key, n_layers + 3)
        self.tok_emb = eqx.nn.Embedding(vocab_size, dim, key=keys[0])
        self.pos_emb = eqx.nn.Embedding(context_length, dim, key=keys[1])
        self.blocks = [
            EqxBlock(dim, n_heads, hidden, keys[2 + i]) for i in range(n_layers)
        ]
        self.norm_f = eqx.nn.LayerNorm(dim)
        self.head = eqx.nn.Linear(dim, vocab_size, use_bias=False, key=keys[-1])
        self.context_length = context_length

    def __call__(self, idx):  # idx: (T,) -> logits (T, vocab_size)
        """Embed token and position indices and run the Transformer stack."""
        seq_len = idx.shape[0]
        x = jax.vmap(self.tok_emb)(idx) + jax.vmap(self.pos_emb)(jnp.arange(seq_len))
        for block in self.blocks:
            x = block(x)
        return jax.vmap(self.head)(jax.vmap(self.norm_f)(x))


_PARITY = dict(
    volatility_parent=False,
    precision=1e4,
    expected_precision=1e4,
)
_PARITY_LEAF = dict(volatility_parent=False)


def _norm_rel(a, b) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_attention_backward_matches_autodiff():
    """The attention composite routes errors exactly like autodiff.

    A fully frozen attention part (Q/K/V/O linear adapters around the hand-derived
    mixing formula) must reproduce the Equinox attention forward, and its backward pass
    must return the same input error as the autodiff of that forward — including the
    causal re-routing of errors across token positions.
    """
    rng = np.random.default_rng(0)
    dim, n_heads, seq_len, batch = 16, 4, 8, 3
    eqx_attn = EqxAttention(dim, n_heads, key=random.key(1))
    x = jnp.asarray(rng.normal(size=(batch, seq_len, dim)))
    error = jnp.asarray(rng.normal(size=(batch, seq_len, dim)))

    part = MultiHeadAttention(
        wq=linear_adapter(eqx_attn.wq),
        wk=linear_adapter(eqx_attn.wk),
        wv=linear_adapter(eqx_attn.wv),
        wo=linear_adapter(eqx_attn.wo),
        n_heads=n_heads,
    )
    # The "target" here is the raw error to inject at the output — the
    # identity error_fn turns step() into one forward + one backward pass.
    fused = FusedPipeline(part, error_fn=lambda out, e: e)

    oracle_forward, vjp = jax.vjp(lambda a: jax.vmap(eqx_attn)(a), x)
    out, error_in = fused.step(x, error)
    np.testing.assert_allclose(out, oracle_forward, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(error_in, vjp(error)[0], rtol=1e-4, atol=1e-5)


def test_full_model_forward_gate():
    """The fully transplanted, fully frozen model reproduces the Equinox GPT.

    The learning-free wiring gate: every weight of the reference model is
    carried into the mixed pipeline, and the assembled pipeline must produce
    the same logits to floating-point precision — validating every adapter,
    every bias placement, and the whole assembly at once, before any
    learning is involved.
    """
    rng = np.random.default_rng(2)
    vocab, dim, n_heads, hidden, n_layers, seq_len = 11, 16, 4, 32, 2, 8
    gpt = EqxGPT(vocab, dim, n_heads, hidden, n_layers, seq_len, key=random.key(3))
    ids = jnp.asarray(rng.integers(0, vocab, size=(3, seq_len)))

    hybrid = hybrid_from_gpt(gpt)
    np.testing.assert_allclose(
        FusedPipeline(hybrid).predict(ids), jax.vmap(gpt)(ids), rtol=1e-4, atol=1e-5
    )


def test_ff_swap_matches_backprop():
    """The feed-forward swap experiment: PyHGF learns what backprop learns.

    Everything in the model is frozen at the reference weights except the feed-forward
    blocks, which become PyHGF networks in the parity configuration. One training step
    from the cross-entropy error at the logits must change every feed-forward matrix —
    bias columns included, in both blocks — by the batch-averaged backprop gradient of
    the same loss, with the error routed backward through the frozen head,
    normalisations, attention, and shortcut junctions of the full model.
    """
    rng = np.random.default_rng(4)
    vocab, dim, n_heads, hidden, n_layers, seq_len = 11, 16, 4, 32, 2, 8
    batch = 3
    gpt = EqxGPT(vocab, dim, n_heads, hidden, n_layers, seq_len, key=random.key(5))
    ids = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))
    targets = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))

    # Oracle: backprop gradients of the mean cross-entropy on the full model.
    def loss_fn(model):
        logits = jax.vmap(model)(ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    grads = eqx.filter_grad(loss_fn)(gpt)

    # The mixed pipeline: frozen everywhere, PyHGF feed-forwards.
    lr = 1e-3
    ff_parts = [
        DeepNetworkAdapter(
            from_feedforward(
                block.ff.fc1,
                block.ff.fc2,
                leaf_kwargs=_PARITY_LEAF,
                layer_kwargs=_PARITY,
            ),
            optimizer=optax.sgd(lr),
            learning_kind="precision_weighted",
        )
        for block in gpt.blocks
    ]
    hybrid = hybrid_from_gpt(gpt, ff_parts=ff_parts)

    # Per-position cross-entropy error at the logits, formed in-trace; the
    # learning parts average over all batch × seq positions internally,
    # which supplies the loss's 1 / (batch * seq) factor.
    fused = FusedPipeline(
        hybrid,
        error_fn=lambda logits, t: (
            jax.nn.softmax(logits, axis=-1) - jax.nn.one_hot(t, vocab)
        ),
    )
    fused.step(ids, targets)
    fused.merge()

    for i, block in enumerate(gpt.blocks):
        net = ff_parts[i].net
        w1b = jnp.concatenate([block.ff.fc2.weight, block.ff.fc2.bias[:, None]], axis=1)
        w2b = jnp.concatenate([block.ff.fc1.weight, block.ff.fc1.bias[:, None]], axis=1)
        g_fc2 = jnp.concatenate(
            [grads.blocks[i].ff.fc2.weight, grads.blocks[i].ff.fc2.bias[:, None]],
            axis=1,
        )
        g_fc1 = jnp.concatenate(
            [grads.blocks[i].ff.fc1.weight, grads.blocks[i].ff.fc1.bias[:, None]],
            axis=1,
        )
        d_fc2 = -(net.state.layers[1].weights_in - w1b) / lr
        d_fc1 = -(net.state.layers[2].weights_in - w2b) / lr
        assert _norm_rel(d_fc2, g_fc2) < 1e-2, f"block {i}, hidden→output"
        assert _norm_rel(d_fc1, g_fc1) < 1e-2, f"block {i}, input→hidden"


def _pyhgf_linear_part(linear, lr):
    from pyhgf.model import from_linear

    return DeepNetworkAdapter(
        from_linear(linear, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY),
        optimizer=optax.sgd(lr),
        learning_kind="precision_weighted",
    )


def test_full_pyhgf_gpt_matches_backprop():
    """Every weight learning through PyHGF matches backprop, per step.

    All weight tables — attention Q/K/V/O, feed-forwards, head, and both embeddings —
    become PyHGF networks in the parity configuration; only the weightless calculations
    (normalisations, attention mixing, shortcuts) stay frozen. One training step from
    the cross-entropy error at the logits must (a) reproduce the reference forward pass,
    and (b) change every single weight matrix by the batch-averaged backprop gradient of
    the same loss. No backpropagation runs anywhere in the pipeline.
    """
    from pyhgf.model import from_embedding

    rng = np.random.default_rng(6)
    vocab, dim, n_heads, hidden, n_layers, seq_len = 11, 16, 4, 32, 2, 8
    batch = 3
    gpt = EqxGPT(vocab, dim, n_heads, hidden, n_layers, seq_len, key=random.key(7))
    ids = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))
    targets = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))

    def loss_fn(model):
        logits = jax.vmap(model)(ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    grads = eqx.filter_grad(loss_fn)(gpt)

    lr = 1e-3
    attention_parts = [
        {
            name: _pyhgf_linear_part(getattr(block.attn, name), lr)
            for name in ("wq", "wk", "wv", "wo")
        }
        for block in gpt.blocks
    ]
    ff_parts = [
        DeepNetworkAdapter(
            from_feedforward(
                block.ff.fc1,
                block.ff.fc2,
                leaf_kwargs=_PARITY_LEAF,
                layer_kwargs=_PARITY,
            ),
            optimizer=optax.sgd(lr),
            learning_kind="precision_weighted",
        )
        for block in gpt.blocks
    ]
    embed_parts = {
        name: DeepNetworkAdapter(
            from_embedding(
                getattr(gpt, name), leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY
            ),
            optimizer=optax.sgd(lr),
            learning_kind="precision_weighted",
        )
        for name in ("tok_emb", "pos_emb")
    }
    hybrid = hybrid_from_gpt(
        gpt,
        ff_parts=ff_parts,
        attention_parts=attention_parts,
        head_part=_pyhgf_linear_part(gpt.head, lr),
        token_part=embed_parts["tok_emb"],
        position_part=embed_parts["pos_emb"],
    )

    fused = FusedPipeline(
        hybrid,
        error_fn=lambda logits, t: (
            jax.nn.softmax(logits, axis=-1) - jax.nn.one_hot(t, vocab)
        ),
    )
    logits, _ = fused.step(ids, targets)
    np.testing.assert_allclose(logits, jax.vmap(gpt)(ids), rtol=1e-3, atol=1e-4)
    fused.merge()

    def delta(part, layer_index, reference):
        return -(part.net.state.layers[layer_index].weights_in - reference) / lr

    # Attention tables and the head (no biases).
    for i, block in enumerate(gpt.blocks):
        for name in ("wq", "wk", "wv", "wo"):
            d = delta(attention_parts[i][name], 1, getattr(block.attn, name).weight)
            g = getattr(grads.blocks[i].attn, name).weight
            assert _norm_rel(d, g) < 2e-2, f"block {i}, attention {name}"
    d_head = delta(hybrid.pipeline.parts[-1], 1, gpt.head.weight)
    assert _norm_rel(d_head, grads.head.weight) < 2e-2, "head"

    # Feed-forwards (bias columns included).
    for i, block in enumerate(gpt.blocks):
        w_fc2 = jnp.concatenate(
            [block.ff.fc2.weight, block.ff.fc2.bias[:, None]], axis=1
        )
        w_fc1 = jnp.concatenate(
            [block.ff.fc1.weight, block.ff.fc1.bias[:, None]], axis=1
        )
        g_fc2 = jnp.concatenate(
            [grads.blocks[i].ff.fc2.weight, grads.blocks[i].ff.fc2.bias[:, None]],
            axis=1,
        )
        g_fc1 = jnp.concatenate(
            [grads.blocks[i].ff.fc1.weight, grads.blocks[i].ff.fc1.bias[:, None]],
            axis=1,
        )
        assert _norm_rel(delta(ff_parts[i], 1, w_fc2), g_fc2) < 2e-2
        assert _norm_rel(delta(ff_parts[i], 2, w_fc1), g_fc1) < 2e-2

    # Embeddings (tables live transposed in the PyHGF layout).
    d_tok = delta(embed_parts["tok_emb"], 1, gpt.tok_emb.weight.T)
    assert _norm_rel(d_tok, grads.tok_emb.weight.T) < 2e-2, "token embedding"
    d_pos = delta(embed_parts["pos_emb"], 1, gpt.pos_emb.weight.T)
    assert _norm_rel(d_pos, grads.pos_emb.weight.T) < 2e-2, "position embedding"


def test_binary_head_at_vocabulary_width():
    """The binary output head composes at vocabulary width (decision Q5).

    The head's bottom layer is one binary (yes/no) node per character; the
    one-hot truth is clamped directly, no error injection. Three properties
    are pinned, all matching a sigmoid-cross-entropy backprop head: (a) the
    forward pass gives the per-class sigmoid of the logits; (b) the error
    message at the head's *input* equals the loss gradient there — the
    Bernoulli variance scaling of the stored binary prediction error cancels
    against the routing gain; (c) the weight update equals the plain loss
    gradient ``(sigmoid - truth) ⊗ input`` — the weight-gradient kernel uses
    the raw residual, which for a sigmoid output *is* the logit-space
    gradient. The remaining deviation from the reference model is therefore
    only 27 independent yes/no questions vs one 27-way choice.
    """
    from pyhgf.model import from_linear

    rng = np.random.default_rng(8)
    vocab, dim, batch = 27, 16, 6
    head = eqx.nn.Linear(dim, vocab, use_bias=False, key=random.key(9))
    x = jnp.asarray(rng.normal(size=(batch, dim)))
    targets = jnp.asarray(rng.integers(0, vocab, size=(batch,)))
    one_hot = jax.nn.one_hot(targets, vocab)

    net = from_linear(
        head,
        leaf_kwargs=dict(kind="binary", **_PARITY_LEAF),
        layer_kwargs=_PARITY,
    )

    # (a) Forward: per-class sigmoid of the logits.
    probs = net.predict(x)
    np.testing.assert_allclose(
        probs, jax.nn.sigmoid(x @ head.weight.T), rtol=1e-4, atol=1e-5
    )

    lr = 1e-3
    net.batch_update(x, one_hot, optimizer=optax.sgd(lr), update_confidences=False)

    # (b) Input message: matches the sigmoid-cross-entropy gradient at x.
    def bce(x_row, y_row):
        logits = x_row @ head.weight.T
        return jnp.sum(optax.sigmoid_binary_cross_entropy(logits, y_row))

    g_x = jax.vmap(lambda a, b: jax.grad(bce)(a, b))(x, one_hot)
    assert _norm_rel(-net.input_errors, g_x) < 1e-2

    # (c) Weight update: the plain sigmoid-cross-entropy gradient.
    expected_grad = ((probs - one_hot)[..., None] * x[:, None, :]).mean(axis=0)
    d_w = -(net.state.layers[1].weights_in - head.weight) / lr
    assert _norm_rel(d_w, expected_grad) < 1e-2


def test_categorical_head_matches_softmax_backprop():
    """The categorical output head composes at vocabulary width.

    The head's bottom layer is one softmax choice across 27 nodes; the
    one-hot truth is clamped directly. All three properties must match a
    softmax-cross-entropy backprop head: (a) the forward pass is the softmax
    of the logits; (b) the weight update equals the batch-averaged loss
    gradient; (c) the error message at the head's input equals the loss
    gradient there. The unit-precision convention of the categorical leaf
    makes the raw residual the cross-entropy gradient in logit space, so the
    generic kernels produce these results with no special-casing.
    """
    from pyhgf.model import from_linear

    rng = np.random.default_rng(10)
    vocab, dim, batch = 27, 16, 6
    head = eqx.nn.Linear(dim, vocab, use_bias=False, key=random.key(11))
    x = jnp.asarray(rng.normal(size=(batch, dim)))
    targets = jnp.asarray(rng.integers(0, vocab, size=(batch,)))
    one_hot = jax.nn.one_hot(targets, vocab)

    net = from_linear(
        head,
        leaf_kwargs=dict(kind="categorical", **_PARITY_LEAF),
        layer_kwargs=_PARITY,
    )

    # (a) Forward: softmax of the logits.
    probs = net.predict(x)
    np.testing.assert_allclose(
        probs, jax.nn.softmax(x @ head.weight.T, axis=-1), rtol=1e-4, atol=1e-6
    )

    lr = 1e-3
    net.batch_update(x, one_hot, optimizer=optax.sgd(lr), update_confidences=False)

    # Oracle: batch-mean softmax cross-entropy gradients.
    def loss(w, x_):
        logits = x_ @ w.T
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    g_w, g_x = jax.grad(lambda w, x_: loss(w, x_), argnums=(0, 1))(head.weight, x)

    # (b) Weight update: the plain cross-entropy gradient.
    d_w = -(net.state.layers[1].weights_in - head.weight) / lr
    assert _norm_rel(d_w, g_w) < 1e-2
    # (c) Input message: PyHGF errors are observed-minus-predicted — the
    # negative of the loss gradient at the input. The oracle's mean over the
    # batch corresponds to the per-sample messages divided by batch.
    assert _norm_rel(-net.input_errors / batch, g_x) < 1e-2


def test_full_pyhgf_gpt_with_categorical_head():
    """The all-PyHGF Transformer with a categorical head matches backprop-CE.

    Same protocol as ``test_full_pyhgf_gpt_matches_backprop``, but the head
    is a categorical leaf and the pipeline is fed ``probs - one_hot`` — the
    correct-and-clamp entry then clamps exactly the one-hot, so the
    cross-entropy error arises from the model's own beliefs. Every weight
    matrix must change by the backprop gradient of the softmax
    cross-entropy loss; the objective caveat of the binary head disappears.
    """
    from pyhgf.model import from_embedding, from_linear

    rng = np.random.default_rng(12)
    vocab, dim, n_heads, hidden, n_layers, seq_len = 11, 16, 4, 32, 2, 8
    batch = 3
    gpt = EqxGPT(vocab, dim, n_heads, hidden, n_layers, seq_len, key=random.key(13))
    ids = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))
    targets = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))

    def loss_fn(model):
        logits = jax.vmap(model)(ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    grads = eqx.filter_grad(loss_fn)(gpt)

    lr = 1e-3
    ff_parts = [
        DeepNetworkAdapter(
            from_feedforward(
                b.ff.fc1, b.ff.fc2, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY
            ),
            optimizer=optax.sgd(lr),
            learning_kind="precision_weighted",
        )
        for b in gpt.blocks
    ]
    attention_parts = [
        {
            name: _pyhgf_linear_part(getattr(block.attn, name), lr)
            for name in ("wq", "wk", "wv", "wo")
        }
        for block in gpt.blocks
    ]
    head_part = DeepNetworkAdapter(
        from_linear(
            gpt.head,
            leaf_kwargs=dict(kind="categorical", **_PARITY_LEAF),
            layer_kwargs=_PARITY,
        ),
        optimizer=optax.sgd(lr),
        learning_kind="precision_weighted",
    )
    token_part = DeepNetworkAdapter(
        from_embedding(gpt.tok_emb, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY),
        optimizer=optax.sgd(lr),
        learning_kind="precision_weighted",
    )
    hybrid = hybrid_from_gpt(
        gpt,
        ff_parts=ff_parts,
        attention_parts=attention_parts,
        head_part=head_part,
        token_part=token_part,
    )

    fused = FusedPipeline(
        hybrid, error_fn=lambda probs, t: probs - jax.nn.one_hot(t, vocab)
    )
    probs, _ = fused.step(ids, targets)  # (B, T, vocab), rows sum to one
    np.testing.assert_allclose(
        probs, jax.nn.softmax(jax.vmap(gpt)(ids), axis=-1), rtol=1e-3, atol=1e-4
    )
    fused.merge()

    def delta(part, layer_index, reference):
        return -(part.net.state.layers[layer_index].weights_in - reference) / lr

    d_head = delta(head_part, 1, gpt.head.weight)
    assert _norm_rel(d_head, grads.head.weight) < 2e-2, "categorical head"
    for i, block in enumerate(gpt.blocks):
        for name in ("wq", "wk", "wv", "wo"):
            d = delta(attention_parts[i][name], 1, getattr(block.attn, name).weight)
            g = getattr(grads.blocks[i].attn, name).weight
            assert _norm_rel(d, g) < 2e-2, f"block {i}, attention {name}"
        w_fc1 = jnp.concatenate(
            [block.ff.fc1.weight, block.ff.fc1.bias[:, None]], axis=1
        )
        g_fc1 = jnp.concatenate(
            [grads.blocks[i].ff.fc1.weight, grads.blocks[i].ff.fc1.bias[:, None]],
            axis=1,
        )
        assert _norm_rel(delta(ff_parts[i], 2, w_fc1), g_fc1) < 2e-2
    d_tok = delta(token_part, 1, gpt.tok_emb.weight.T)
    assert _norm_rel(d_tok, grads.tok_emb.weight.T) < 2e-2, "token embedding"


def test_categorical_layer_is_leaf_only():
    """A categorical layer anywhere but the bottom raises a clear error."""
    from pyhgf.model import DeepNetwork

    with np.testing.assert_raises(ValueError):
        DeepNetwork().add_layer(size=4).add_layer(size=27, kind="categorical")
