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
    from_embedding,
    from_feedforward,
    from_linear,
    hybrid_from_gpt,
)

# The executor's gates, verified against
# automatic differentiation on the same forward functions — autodiff appears
# here ONLY as a test oracle; the pipeline itself never uses it. The
# single-part gate is here; the mixed-block gate lives in test_hybrid.py and
# the full-model gradient gates in test_transformer.py.

_PARITY = dict(
    volatility_parent=False,
    precision=1e4,
    expected_precision=1e4,
)
_PARITY_LEAF = dict(volatility_parent=False)


def _norm_rel(a, b) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_fused_adapter_tracks_backprop_twin():
    """A single learning part tracks an SGD backprop twin, step for step.

    A biased feed-forward network in the parity configuration, trained by the
    fused executor for three steps, must follow a twin trained by autodiff
    SGD on the identical forward function: same weights (bias columns
    included) after every step, and the same per-sample input errors.
    """
    rng = np.random.default_rng(0)
    d, h, batch, lr = 6, 12, 5, 1e-3
    k1, k2 = random.split(random.key(1))
    fc1 = eqx.nn.Linear(d, h, key=k1)
    fc2 = eqx.nn.Linear(h, d, key=k2)

    part = DeepNetworkAdapter(
        from_feedforward(fc1, fc2, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY),
        optimizer=optax.sgd(lr),
        learning_kind="precision_weighted",
    )
    fused = FusedPipeline(part)

    # The twin: the same folded [W | b] blocks, trained by autodiff.
    w1b = jnp.concatenate([fc1.weight, fc1.bias[:, None]], axis=1)
    w2b = jnp.concatenate([fc2.weight, fc2.bias[:, None]], axis=1)

    def forward(w1b_, w2b_, row):
        hidden = w1b_[:, :-1] @ row + w1b_[:, -1]
        return w2b_[:, :-1] @ jax.nn.gelu(hidden) + w2b_[:, -1]

    for step in range(3):
        xb = jnp.asarray(rng.normal(size=(batch, d)))
        yb = jnp.asarray(rng.normal(size=(batch, d)))

        def mean_loss(w1b_, w2b_):
            out = jax.vmap(lambda row: forward(w1b_, w2b_, row))(xb)
            return jnp.mean(jnp.sum(0.5 * (yb - out) ** 2, axis=-1))

        g_w1b, g_w2b = jax.grad(mean_loss, argnums=(0, 1))(w1b, w2b)
        per_sample_dx = jax.vmap(
            lambda x_row, y_row: jax.grad(
                lambda a: jnp.sum(0.5 * (y_row - forward(w1b, w2b, a)) ** 2)
            )(x_row)
        )(xb, yb)

        _, input_error = fused.step(xb, yb)
        w1b = w1b - lr * g_w1b
        w2b = w2b - lr * g_w2b

        fused.merge()
        assert _norm_rel(part.net.state.layers[2].weights_in, w1b) < 1e-4, f"{step}"
        assert _norm_rel(part.net.state.layers[1].weights_in, w2b) < 1e-4, f"{step}"
        assert _norm_rel(input_error, per_sample_dx) < 1e-2, f"step {step}"


def test_fused_confidences_carry_across_steps():
    """Released confidences adapt across steps; pinned ones stay put."""
    rng = np.random.default_rng(2)
    k1, k2 = random.split(random.key(3))
    fc1 = eqx.nn.Linear(6, 12, key=k1)
    fc2 = eqx.nn.Linear(12, 6, key=k2)

    def run(update_confidences):
        leaf = _PARITY_LEAF if not update_confidences else {}
        layer = (
            _PARITY
            if not update_confidences
            else dict(precision=30.0, expected_precision=30.0)
        )
        part = DeepNetworkAdapter(
            from_feedforward(fc1, fc2, leaf_kwargs=leaf, layer_kwargs=layer),
            optimizer=optax.adam(1e-3),
            update_confidences=update_confidences,
        )
        fused = FusedPipeline(part)
        before = fused.state[0].layers[1].state.precision
        for _ in range(3):
            fused.step(
                jnp.asarray(rng.normal(size=(4, 6))),
                jnp.asarray(rng.normal(size=(4, 6))),
            )
        return before, fused.state[0].layers[1].state.precision

    before, after = run(update_confidences=True)
    assert not jnp.allclose(before, after)  # the confidence state moved
    before, after = run(update_confidences=False)
    np.testing.assert_allclose(before, after)  # pinned for parity


def test_fused_gpt_trajectory_matches_backprop():
    """A parity Transformer follows its backprop twin over a training run.

    The all-weights-PyHGF model in the parity configuration and an autodiff
    twin of the same architecture, trained by plain SGD at the same rate on
    identical batches for 100 steps: the two cross-entropy curves must stay
    close over the whole run (they are the same mathematics; only
    floating-point noise separates them, and it compounds — hence a looser
    tolerance late than the per-step gates use).
    """
    from test_transformer import EqxGPT

    rng = np.random.default_rng(4)
    vocab, dim, n_heads, hidden, n_layers, seq_len = 11, 16, 2, 24, 1, 8
    batch, lr, n_steps = 4, 0.1, 100
    gpt = EqxGPT(vocab, dim, n_heads, hidden, n_layers, seq_len, key=random.key(5))

    def part(net):
        return DeepNetworkAdapter(
            net, optimizer=optax.sgd(lr), learning_kind="precision_weighted"
        )

    hybrid = hybrid_from_gpt(
        gpt,
        ff_parts=[
            part(
                from_feedforward(
                    b.ff.fc1, b.ff.fc2, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY
                )
            )
            for b in gpt.blocks
        ],
        attention_parts=[
            {
                name: part(
                    from_linear(
                        getattr(b.attn, name),
                        leaf_kwargs=_PARITY_LEAF,
                        layer_kwargs=_PARITY,
                    )
                )
                for name in ("wq", "wk", "wv", "wo")
            }
            for b in gpt.blocks
        ],
        head_part=part(
            from_linear(
                gpt.head,
                leaf_kwargs=dict(kind="categorical", **_PARITY_LEAF),
                layer_kwargs=_PARITY,
            )
        ),
        token_part=part(
            from_embedding(gpt.tok_emb, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY)
        ),
        position_part=part(
            from_embedding(gpt.pos_emb, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY)
        ),
    )
    fused = FusedPipeline(
        hybrid, error_fn=lambda probs, t: probs - jax.nn.one_hot(t, vocab)
    )

    # The backprop twin, same weights, same optimiser, same batches.
    twin = gpt
    sgd = optax.sgd(lr)
    opt_state = sgd.init(eqx.filter(twin, eqx.is_array))

    def ce_loss(model, x, y):
        logits = jax.vmap(model)(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @eqx.filter_jit
    def twin_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(ce_loss)(model, x, y)
        updates, opt_state = sgd.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    losses_f, losses_b = [], []
    for _step in range(n_steps):
        ids = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))
        targets = jnp.asarray(rng.integers(0, vocab, size=(batch, seq_len)))

        probs, _ = fused.step(ids, targets)
        picked = jnp.take_along_axis(probs, targets[..., None], axis=-1)
        losses_f.append(float(-jnp.log(picked + 1e-9).mean()))

        twin, opt_state, loss = twin_step(twin, opt_state, ids, targets)
        losses_b.append(float(loss))

    gap = np.max(np.abs(np.array(losses_f) - np.array(losses_b)))
    assert gap < 0.05, f"largest cross-entropy gap over the run: {gap:.4f}"


def test_fused_merge_writes_state_back():
    """``merge`` puts the advanced state back onto the wrapped parts."""
    rng = np.random.default_rng(6)
    k1, k2 = random.split(random.key(7))
    fc1 = eqx.nn.Linear(6, 12, key=k1)
    fc2 = eqx.nn.Linear(12, 6, key=k2)
    adapter = DeepNetworkAdapter(
        from_feedforward(fc1, fc2, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY),
        optimizer=optax.adam(1e-3),
    )
    before = adapter.net.state.layers[1].weights_in
    fused = FusedPipeline(adapter)
    fused.step(
        jnp.asarray(rng.normal(size=(4, 6))), jnp.asarray(rng.normal(size=(4, 6)))
    )
    assert jnp.allclose(adapter.net.state.layers[1].weights_in, before)  # untouched
    fused.merge()
    assert not jnp.allclose(adapter.net.state.layers[1].weights_in, before)
    assert adapter.net.opt_state is fused.state[1]


def test_fused_predict_is_read_only():
    """``predict`` returns the forward pass and advances nothing."""
    rng = np.random.default_rng(8)
    k1, k2 = random.split(random.key(9))
    fc1 = eqx.nn.Linear(6, 12, key=k1)
    fc2 = eqx.nn.Linear(12, 6, key=k2)
    adapter = DeepNetworkAdapter(
        from_feedforward(fc1, fc2, leaf_kwargs=_PARITY_LEAF, layer_kwargs=_PARITY),
        optimizer=optax.adam(1e-3),
    )
    fused = FusedPipeline(adapter)
    x = jnp.asarray(rng.normal(size=(4, 6)))

    state_before = fused.state
    out = fused.predict(x)
    assert fused.state is state_before
    oracle = jax.vmap(lambda row: fc2(jax.nn.gelu(fc1(row))))(x)
    np.testing.assert_allclose(out, oracle, rtol=1e-4, atol=1e-5)


def test_fused_rejects_unsupported_parts():
    """Part types outside the supported set are refused, and say so."""
    from pyhgf.model import PCModule

    class Opaque(PCModule):
        """Custom PCModule that isn't a recognized executor type."""

        def __init__(self):
            pass

        def init_state(self):
            return ()

    with np.testing.assert_raises(NotImplementedError):
        FusedPipeline(Opaque())
