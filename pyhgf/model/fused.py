# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""The pipeline executor: one compiled program per training step.

The classes in :mod:`pyhgf.model.hybrid` and :mod:`pyhgf.model.transformer` *declare* a
model — which slots learn, which are frozen, how parts nest. :class:`FusedPipeline`
runs such a part tree: every part's state (network beliefs, weights, optimiser
moments) enters and leaves as explicit values, and the forward walk, the error at the
output, and every local learning step are staged into a single compiled program —
nothing crosses a compilation boundary inside one training step.

See ``docs/source/notebooks/0.6-Deep_networks_implementation.md`` for the architecture guide and the
verification gates.

Supported part trees: :class:`~pyhgf.model.hybrid.DeepNetworkAdapter`,
:class:`~pyhgf.model.hybrid.EquinoxAdapter`, :class:`~pyhgf.model.hybrid.PCSequential`,
:class:`~pyhgf.model.hybrid.Residual`,
:class:`~pyhgf.model.transformer.MultiHeadAttention`, and the assembled
:class:`~pyhgf.model.transformer.HybridGPT` — the entire Transformer trains in one
compiled call per step.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from pyhgf.model.hybrid import (
    DeepNetworkAdapter,
    EquinoxAdapter,
    PCSequential,
    Residual,
)
from pyhgf.model.transformer import (
    HybridGPT,
    MultiHeadAttention,
    _mixing_backward,
    _mixing_forward,
)
from pyhgf.utils.vectorized_belief_propagation import (
    _batch_step,
    _prediction_sweep,
)

__all__ = ["FusedPipeline", "step_report"]


class _Core(NamedTuple):
    """A part tree's training step as pure functions with explicit state.

    ``forward(state, x)`` returns the tree's output and a cache of whatever the backward
    walk needs; ``backward(state, cache, error)`` returns the error at the tree's input,
    the advanced state, and a tuple of per-part report entries (see
    :func:`step_report`). All are pure — the recursion over the tree happens while
    tracing, staging one graph. ``write_back(state)`` copies a state pytree onto the
    part objects the core was built from.
    """

    init_state: Any
    forward: Callable[[Any, jnp.ndarray], tuple[jnp.ndarray, Any]]
    backward: Callable[[Any, Any, jnp.ndarray], tuple[jnp.ndarray, Any, tuple]]
    write_back: Callable[[Any], None]


def _adapter_core(part: DeepNetworkAdapter, path: str) -> _Core:
    """Build the core of a learning part: sweep forward, batch learning backward.

    The state is the pair (network, optimiser state). The forward pass keeps the per-
    sample swept states as the cache, so the backward pass starts from them instead of
    re-running the sweep — inside one trace, the handover is free.
    """
    optimizer = part.optimizer
    learning_kind = part.learning_kind
    update_confidences = part.update_confidences
    time_step = part.time_step
    layer_sizes = list(part.net.layer_sizes)

    # part.init_state() returns the (network, opt_state) pytree directly.
    init_state_pytree = part.init_state()

    def forward(state, x):
        network, _ = state
        flat = x.reshape(-1, x.shape[-1])

        def one(xi):
            swept = _prediction_sweep(network, xi, time_step=time_step)
            return tuple(elem.state for elem in swept.layers)

        states = jax.vmap(one)(flat)
        out = states[0].expected_mean
        return out.reshape(*x.shape[:-1], -1), (flat, states)

    def backward(state, cache, error):
        network, opt_state = state
        flat_x, states = cache
        out = states[0].expected_mean
        weights_before = network.weights_tuple()
        # Correct-and-clamp: the observation the network should have
        # produced is its output corrected by the error.
        network, opt_state, input_errors = _batch_step(
            network,
            opt_state,
            flat_x,
            out - error.reshape(-1, error.shape[-1]),
            optimizer=optimizer,
            learning_kind=learning_kind,
            update_confidences=update_confidences,
            time_step=time_step,
            predicted=states,
        )
        update_norm = jnp.sqrt(
            sum(
                jnp.sum((after - before) ** 2)
                for before, after in zip(weights_before, network.weights_tuple())
                if after is not None
            )
        )
        report = (
            {
                "part": path,
                "layer_sizes": layer_sizes,
                "update_norm": update_norm,
                "error_norm": jnp.linalg.norm(error),
            },
        )
        # PyHGF's input errors are observed-minus-predicted; the pipeline
        # convention is descent.
        return (
            -input_errors.reshape(*error.shape[:-1], -1),
            (network, opt_state),
            report,
        )

    def write_back(state):
        part.net.state, part.net.opt_state = state

    return _Core(init_state_pytree, forward, backward, write_back)


def _frozen_core(part: EquinoxAdapter, path: str) -> _Core:
    """Build the core of a frozen part: its formula pair, cache threaded explicitly.

    A frozen part has no learnable state, so its state pytree is empty.
    """
    init_state_pytree = ()

    def forward(state, x):
        return part.forward_fn(x)

    def backward(state, cache, error):
        return part.backward_fn(cache, error), state, ()

    return _Core(init_state_pytree, forward, backward, lambda state: None)


def _sequential_core(part: PCSequential, path: str) -> _Core:
    """Build the core of a chain: children forward in order, backward in reverse.

    The state is a tuple of the children's states, built from the child cores.
    """
    cores = [_core(child, f"{path}.parts[{i}]") for i, child in enumerate(part.parts)]

    init_state_pytree = tuple(core.init_state for core in cores)

    def forward(state, x):
        caches = []
        for core, child_state in zip(cores, state):
            x, cache = core.forward(child_state, x)
            caches.append(cache)
        return x, tuple(caches)

    def backward(state, caches, error):
        new_state = list(state)
        reports: list = []
        for i in reversed(range(len(cores))):
            error, new_state[i], report = cores[i].backward(state[i], caches[i], error)
            reports[:0] = report  # keep pipeline order, not visit order
        return error, tuple(new_state), tuple(reports)

    def write_back(state):
        for core, child_state in zip(cores, state):
            core.write_back(child_state)

    return _Core(init_state_pytree, forward, backward, write_back)


def _residual_core(part: Residual, path: str) -> _Core:
    """Build the core of the shortcut junction: the branch's messages add back.

    State is obtained from the branch via its _Core's init_state pytree. The shortcut
    itself has no state (it just routes signals), so the pytree is just the branch's
    state.
    """
    inner = _core(part.branch, f"{path}.branch")

    def forward(state, x):
        y, cache = inner.forward(state, x)
        return x + y, cache

    def backward(state, cache, error):
        error_in, new_state, report = inner.backward(state, cache, error)
        return error + error_in, new_state, report

    return _Core(inner.init_state, forward, backward, inner.write_back)


def _attention_core(part: MultiHeadAttention, path: str) -> _Core:
    """Build the core of the attention composite: Q/K/V, the mixing, then O.

    Backward, the output error returns through O, is re-routed across positions by the
    mixing formula, and the three resulting messages return through Q, K, and V — whose
    input errors add, since all three read the same input.

    State is obtained from each of the four weight table cores (Q, K, V, O), which have
    already been initialized via their respective ``init_state()`` calls during _core
    construction.
    """
    q_core, k_core, v_core, o_core = (
        _core(child, f"{path}.{name}")
        for name, child in (
            ("wq", part.wq),
            ("wk", part.wk),
            ("wv", part.wv),
            ("wo", part.wo),
        )
    )
    n_heads = part.n_heads

    # Aggregate init_state pytrees from all four cores.
    init_state_pytree = tuple(
        core.init_state for core in (q_core, k_core, v_core, o_core)
    )

    def forward(state, x):
        state_q, state_k, state_v, state_o = state
        q, cache_q = q_core.forward(state_q, x)
        k, cache_k = k_core.forward(state_k, x)
        v, cache_v = v_core.forward(state_v, x)
        ctx, cache_mix = _mixing_forward(q, k, v, n_heads)
        y, cache_o = o_core.forward(state_o, ctx)
        return y, (cache_q, cache_k, cache_v, cache_mix, cache_o)

    def backward(state, cache, error):
        state_q, state_k, state_v, state_o = state
        cache_q, cache_k, cache_v, cache_mix, cache_o = cache
        d_ctx, new_o, report_o = o_core.backward(state_o, cache_o, error)
        d_q, d_k, d_v = _mixing_backward(cache_mix, d_ctx)
        error_q, new_q, report_q = q_core.backward(state_q, cache_q, d_q)
        error_k, new_k, report_k = k_core.backward(state_k, cache_k, d_k)
        error_v, new_v, report_v = v_core.backward(state_v, cache_v, d_v)
        return (
            error_q + error_k + error_v,
            (new_q, new_k, new_v, new_o),
            report_q + report_k + report_v + report_o,
        )

    def write_back(state):
        for core, child_state in zip((q_core, k_core, v_core, o_core), state):
            core.write_back(child_state)

    return _Core(init_state_pytree, forward, backward, write_back)


def _gpt_core(part: HybridGPT, path: str) -> _Core:
    """Build the core of the full Transformer: embeddings, blocks, and head.

    The forward pass takes integer token ids, not feature arrays: the one-hot encoding
    (or the frozen table lookup) happens in-trace. Backward, the error at the embedding
    output goes to both embedding parts — the position part receives the batch mean,
    matching its one-row-per-position sample count — and is returned, ending the chain.

    State is obtained from the pipeline and optional embedding parts via their
    respective _Core init_state pytrees. Frozen embeddings contribute empty states.
    """
    pipeline_core = _core(part.pipeline, f"{path}.pipeline")
    token_core = (
        None
        if part.token_part is None
        else _core(part.token_part, f"{path}.token_part")
    )
    position_core = (
        None
        if part.position_part is None
        else _core(part.position_part, f"{path}.position_part")
    )
    tok_table, pos_table = part.tok_table, part.pos_table

    # Aggregate init_state pytrees: pipeline (always present) and optional
    # token/position cores (empty tuples if frozen).
    init_state_pytree = (
        pipeline_core.init_state,
        () if token_core is None else token_core.init_state,
        () if position_core is None else position_core.init_state,
    )

    def forward(state, ids):
        state_pipe, state_tok, state_pos = state
        seq_len = ids.shape[1]
        if token_core is not None:
            tok, cache_tok = token_core.forward(
                state_tok, jax.nn.one_hot(ids, tok_table.shape[0])
            )
        else:
            tok, cache_tok = tok_table[ids], ()
        if position_core is not None:
            pos, cache_pos = position_core.forward(
                state_pos, jax.nn.one_hot(jnp.arange(seq_len), pos_table.shape[0])
            )
        else:
            pos, cache_pos = pos_table[:seq_len], ()
        out, cache_pipe = pipeline_core.forward(state_pipe, tok + pos[None])
        return out, (cache_pipe, cache_tok, cache_pos)

    def backward(state, cache, error):
        state_pipe, state_tok, state_pos = state
        cache_pipe, cache_tok, cache_pos = cache
        embedding_error, new_pipe, reports = pipeline_core.backward(
            state_pipe, cache_pipe, error
        )
        new_tok, new_pos = state_tok, state_pos
        if token_core is not None:
            _, new_tok, report = token_core.backward(
                state_tok, cache_tok, embedding_error
            )
            reports = reports + report
        if position_core is not None:
            _, new_pos, report = position_core.backward(
                state_pos, cache_pos, embedding_error.mean(axis=0)
            )
            reports = reports + report
        return embedding_error, (new_pipe, new_tok, new_pos), reports

    def write_back(state):
        state_pipe, state_tok, state_pos = state
        pipeline_core.write_back(state_pipe)
        if token_core is not None:
            token_core.write_back(state_tok)
        if position_core is not None:
            position_core.write_back(state_pos)

    return _Core(init_state_pytree, forward, backward, write_back)


def _core(part, path: Optional[str] = None) -> _Core:
    """Build the functional core for a part tree, by its type."""
    if path is None:
        path = type(part).__name__
    if isinstance(part, DeepNetworkAdapter):
        return _adapter_core(part, path)
    if isinstance(part, EquinoxAdapter):
        return _frozen_core(part, path)
    if isinstance(part, PCSequential):
        return _sequential_core(part, path)
    if isinstance(part, Residual):
        return _residual_core(part, path)
    if isinstance(part, MultiHeadAttention):
        return _attention_core(part, path)
    if isinstance(part, HybridGPT):
        return _gpt_core(part, path)
    raise NotImplementedError(
        "FusedPipeline supports DeepNetworkAdapter, EquinoxAdapter, "
        "PCSequential, Residual, MultiHeadAttention, and HybridGPT trees; "
        f"got {type(part).__name__}."
    )


class FusedPipeline:
    """Run a part tree, one compiled program per training step.

    Holds every part's mutable state (the network beliefs and weights, the optimiser
    states) as one explicit pytree, and compiles a single step function: forward walk →
    error at the output → backward walk with every local learning step → error at the
    input. The error is formed *inside* the program by ``error_fn``, so no intermediate
    crosses a compilation boundary.

    The part objects only declare the model; they are not advanced while the executor
    runs. Call :meth:`merge` to write the current state back onto them (e.g. to inspect
    a network's layers, or to save it).

    Parameters
    ----------
    part :
        The part tree to execute: any composition of
        :class:`~pyhgf.model.hybrid.DeepNetworkAdapter`,
        :class:`~pyhgf.model.hybrid.EquinoxAdapter`,
        :class:`~pyhgf.model.hybrid.PCSequential`,
        :class:`~pyhgf.model.hybrid.Residual`, and
        :class:`~pyhgf.model.transformer.MultiHeadAttention`, or a full
        :class:`~pyhgf.model.transformer.HybridGPT` (whose inputs are then
        integer token ids).
    error_fn :
        How the descent error at the tree's output is formed from the output and the
        training target, e.g. ``lambda out, target: out - target`` for a squared-error
        objective (the default), or
        ``lambda probs, ids: probs - jax.nn.one_hot(ids, vocab)`` for a categorical
        head trained on integer targets. Must be a pure function of arrays.
    """

    def __init__(
        self,
        part,
        error_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    ):
        if error_fn is None:
            error_fn = lambda out, target: out - target  # noqa: E731
        self._part = part
        self._core_fns = _core(part)
        self.state = self._core_fns.init_state
        self._last_report: Optional[tuple] = None

        def step(state, x, target):
            output, cache = self._core_fns.forward(state, x)
            error = error_fn(output, target)
            input_error, state, reports = self._core_fns.backward(state, cache, error)
            return state, output, input_error, reports

        self._step = eqx.filter_jit(step)
        self._predict = eqx.filter_jit(
            lambda state, x: self._core_fns.forward(state, x)[0]
        )

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the forward pass only — no error, no learning, no state change.

        Use for evaluation and generation. For a
        :class:`~pyhgf.model.transformer.HybridGPT`, ``x`` is a batch of integer token-
        id sequences.
        """
        return self._predict(self.state, jnp.asarray(x))

    def step(
        self, x: jnp.ndarray, target: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Advance the held state by one training step, in one compiled call.

        Parameters
        ----------
        x :
            Inputs, shape ``(batch, n_input_features)`` — or integer token ids,
            ``(batch, seq)``, for a full Transformer.
        target :
            Training targets, shaped as ``error_fn`` expects alongside the tree's
            output.

        Returns
        -------
        output :
            The tree's predictions for ``x``, shape ``(batch, n_output_features)``.
        input_error :
            The descent error at the tree's input — what a part behind the tree
            would receive.
        """
        self.state, output, input_error, self._last_report = self._step(
            self.state, jnp.asarray(x), jnp.asarray(target)
        )
        return output, input_error

    def merge(self):
        """Write the held state back onto the wrapped parts; return the tree."""
        self._core_fns.write_back(self.state)
        return self._part


def step_report(pipeline: FusedPipeline) -> list:
    """Per-part step magnitudes after a training step, for rate calibration.

    Returns one entry per learning part of the last :meth:`FusedPipeline.step`
    call: the norm of its applied weight change and of the error it received,
    labelled by the part's path inside the model. Reading the two side by side
    shows at a glance which parts are being over- or under-driven — the
    practical symptom of a mis-calibrated learning rate (see the architecture
    guide on choosing the optimizer and rate).

    Parameters
    ----------
    pipeline :
        The executor to read. Its part tree must have taken at least one step.

    Returns
    -------
    list of dict
        One entry per learning part, in pipeline order, with keys ``"part"``
        (its path inside the model), ``"layer_sizes"`` (the wrapped network's
        layout), ``"update_norm"`` and ``"error_norm"``.
    """
    if pipeline._last_report is None:
        return []
    return [
        {
            "part": entry["part"],
            "layer_sizes": entry["layer_sizes"],
            "update_norm": float(entry["update_norm"]),
            "error_norm": float(entry["error_norm"]),
        }
        for entry in pipeline._last_report
    ]
