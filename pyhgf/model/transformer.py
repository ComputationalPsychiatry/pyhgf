# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""A GPT-style Transformer as a mixed pipeline.

Assembles the parts from :mod:`pyhgf.model.hybrid` into the architecture of the
reference Transformer (see ``docs/source/notebooks/0.8-Transformers.ipynb``):
embeddings, repeated blocks of attention and feed-forward sub-layers with normalisation
and shortcut junctions, a final normalisation, and an output head. Any slot can hold a
frozen calculation or a learning PyHGF network; errors travel backward part by part,
with no global backward sweep (see :mod:`pyhgf.model.hybrid` for the conventions and
:mod:`pyhgf.model.fused` for the executor).

The one part specific to Transformers is :class:`MultiHeadAttention`: the only place
where token positions exchange information. Its four weight tables (Q, K, V, O) are
ordinary per-position parts; the *mixing* between them — compare queries against keys,
turn the scores into attention percentages, blend the values — has no weights of its
own, so its role in the backward pass is purely to re-route errors across positions,
using a hand-derived formula (no automatic differentiation).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from pyhgf.model.hybrid import (
    PCModule,
    PCSequential,
    Residual,
    gelu_adapter,
    layer_norm_adapter,
    linear_adapter,
)

__all__ = [
    "MultiHeadAttention",
    "HybridGPT",
    "hybrid_from_gpt",
]


class MultiHeadAttention(PCModule):
    """Causal multi-head self-attention as a composite pipeline part.

    Holds four single-input parts for the Q, K, V, and O weight tables (any
    mix of frozen and learning parts) around the weight-free mixing step.
    Operates on ``(batch, seq, features)`` arrays — attention is the one
    part that needs the sequence axis explicit, because it moves information
    *between* positions; everywhere else each position is independent.

    Forward: every position emits a query, a key, and a value through the
    Q/K/V parts; each position's attention scores against all *earlier*
    positions (causal mask) become percentages through a softmax; the values
    are blended accordingly and pass through the O part.

    Backward: the error at the output goes back through O, is re-routed
    across positions by the mixing formula (an error at position ``t`` flows
    back to the positions ``t`` attended to, and to the query/key pair that
    set those percentages), and the three resulting messages return through
    Q, K, and V — whose input errors add, since all three read the same
    input.
    """

    def __init__(
        self,
        wq: PCModule,
        wk: PCModule,
        wv: PCModule,
        wo: PCModule,
        n_heads: int,
    ):
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.n_heads = n_heads

    def init_state(self) -> tuple:
        """Return the ``(q, k, v, o)`` tuple of the four weight tables' states."""
        return (
            self.wq.init_state(),
            self.wk.init_state(),
            self.wv.init_state(),
            self.wo.init_state(),
        )


def _split_heads(a: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    # (B, T, D) -> (B, H, T, D // H)
    b, t, d = a.shape
    return a.reshape(b, t, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def _merge_heads(a: jnp.ndarray) -> jnp.ndarray:
    # (B, H, T, hd) -> (B, T, D)
    b, h, t, hd = a.shape
    return a.transpose(0, 2, 1, 3).reshape(b, t, h * hd)


def _mixing_forward(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, n_heads: int
) -> tuple[jnp.ndarray, tuple]:
    """Compute the weight-free attention mixing.

    Splits the query/key/value streams into heads, scores every position against the
    earlier positions (causal mask), softmaxes the scores into attention percentages,
    and blends the values. Returns the blended context, ``(batch, seq, features)``, and
    the head-shaped cache the backward pass needs. Plain and unjitted: it is staged into
    the executor's compiled step.
    """
    qh = _split_heads(q, n_heads)
    kh = _split_heads(k, n_heads)
    vh = _split_heads(v, n_heads)

    seq_len, head_dim = qh.shape[2], qh.shape[-1]
    scores = qh @ kh.transpose(0, 1, 3, 2) / jnp.sqrt(float(head_dim))
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    scores = jnp.where(causal_mask[None, None] == 0, -jnp.inf, scores)
    attn = jax.nn.softmax(scores, axis=-1)

    return _merge_heads(attn @ vh), (attn, qh, kh, vh)


def _mixing_backward(
    cache: tuple, d_ctx: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Re-route the context error across positions — the mixing's backward.

    An error at position ``t`` flows back to the positions ``t`` attended to (through
    the cached attention percentages), and to the query/key pair that set those
    percentages (through the softmax, where the row-weighted mean of the incoming error
    is subtracted because each row of percentages sums to one; masked positions carry
    zero attention and drop out automatically).
    """
    attn, qh, kh, vh = cache
    head_dim = qh.shape[-1]

    d_ctx_h = _split_heads(d_ctx, attn.shape[1])
    d_attn = d_ctx_h @ vh.transpose(0, 1, 3, 2)
    d_v = attn.transpose(0, 1, 3, 2) @ d_ctx_h
    d_scores = (
        attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
    ) / jnp.sqrt(float(head_dim))
    d_q = d_scores @ kh
    d_k = d_scores.transpose(0, 1, 3, 2) @ qh

    return _merge_heads(d_q), _merge_heads(d_k), _merge_heads(d_v)


class HybridGPT:
    """The full Transformer as a mixed pipeline, from token ids to logits.

    The embeddings sit outside the part contract (token ids are integers,
    not features). By default they are frozen lookup tables held directly.
    Passing ``token_part`` / ``position_part`` (learning parts fed one-hot
    rows, e.g. built with :func:`~pyhgf.model.transplant.from_embedding`)
    makes them learn: a lookup equals a one-hot vector times the table, so
    the parts see one-hot inputs and receive the error at the embedding
    output. The rest of the model — blocks, final normalisation, head — is
    one :class:`~pyhgf.model.hybrid.PCSequential` over
    ``(batch, seq, features)`` arrays.

    Run it with :class:`~pyhgf.model.fused.FusedPipeline`, whose ``step``
    and ``predict`` take batches of integer token-id sequences,
    ``(batch, seq)``.
    """

    def __init__(
        self,
        tok_table: jnp.ndarray,
        pos_table: jnp.ndarray,
        pipeline: PCSequential,
        token_part=None,
        position_part=None,
    ):
        self.tok_table = jnp.asarray(tok_table)
        self.pos_table = jnp.asarray(pos_table)
        self.pipeline = pipeline
        self.token_part = token_part
        self.position_part = position_part

    def init_state(self) -> tuple:
        """Return ``(pipeline_state, token_state, position_state)``.

        The embedding states are empty tuples when their parts are frozen.
        """
        token_state = self.token_part.init_state() if self.token_part else ()
        position_state = self.position_part.init_state() if self.position_part else ()
        return (self.pipeline.init_state(), token_state, position_state)


def hybrid_from_gpt(
    gpt,
    ff_parts: Optional[list] = None,
    attention_parts: Optional[list] = None,
    head_part: Optional[PCModule] = None,
    token_part: Optional[PCModule] = None,
    position_part: Optional[PCModule] = None,
) -> HybridGPT:
    """Assemble a :class:`HybridGPT` from a reference Equinox GPT.

    Expects the attribute layout of the reference model in
    ``docs/source/notebooks/0.8-Transformers.ipynb``: ``tok_emb``,
    ``pos_emb``, ``blocks`` (each with ``attn.wq/wk/wv/wo``, ``ff.fc1/fc2``,
    ``n1``, ``n2``), ``norm_f``, and ``head``. Every slot is frozen at the
    given model's weights unless a learning part is supplied for it.

    With all slots frozen, the assembled pipeline computes exactly the same
    function as the Equinox model — the transplant wiring gate. Supplying
    parts (e.g. :class:`~pyhgf.model.hybrid.DeepNetworkAdapter`s built by
    the :mod:`~pyhgf.model.transplant` converters) turns the corresponding
    slots into PyHGF learners while everything else stays pinned — from the
    single-component swap experiment up to a model whose every weight learns
    through PyHGF, with only the weightless calculations (normalisations,
    attention mixing) frozen.

    Parameters
    ----------
    gpt :
        The Equinox reference model whose weights are transplanted.
    ff_parts :
        Optional list of parts, one per block, replacing the frozen
        feed-forwards.
    attention_parts :
        Optional list of dicts, one per block, with keys ``"wq"``, ``"wk"``,
        ``"wv"``, ``"wo"``, replacing the frozen attention weight tables.
    head_part :
        Optional part replacing the frozen output head.
    token_part, position_part :
        Optional parts replacing the frozen embedding tables (fed one-hot
        rows — see :class:`HybridGPT`).

    Returns
    -------
    HybridGPT
        The assembled mixed pipeline.
    """
    layers: list = []
    for i, block in enumerate(gpt.blocks):
        attn_tables = attention_parts[i] if attention_parts is not None else {}
        attention = MultiHeadAttention(
            wq=attn_tables.get("wq") or linear_adapter(block.attn.wq),
            wk=attn_tables.get("wk") or linear_adapter(block.attn.wk),
            wv=attn_tables.get("wv") or linear_adapter(block.attn.wv),
            wo=attn_tables.get("wo") or linear_adapter(block.attn.wo),
            n_heads=block.attn.n_heads,
        )
        if ff_parts is not None:
            feed_forward = ff_parts[i]
        else:
            feed_forward = PCSequential([
                linear_adapter(block.ff.fc1),
                gelu_adapter(),
                linear_adapter(block.ff.fc2),
            ])
        layers.append(
            PCSequential([
                Residual(PCSequential([layer_norm_adapter(block.n1), attention])),
                Residual(PCSequential([layer_norm_adapter(block.n2), feed_forward])),
            ])
        )
    layers.append(layer_norm_adapter(gpt.norm_f))
    layers.append(head_part if head_part is not None else linear_adapter(gpt.head))

    return HybridGPT(
        tok_table=gpt.tok_emb.weight,
        pos_table=gpt.pos_emb.weight,
        pipeline=PCSequential(layers),
        token_part=token_part,
        position_part=position_part,
    )
