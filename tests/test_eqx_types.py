# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Phase 1 tests: Equinox PyTree types live next to the legacy NamedTuples.

These tests verify the new ``pyhgf.typing.vectorised`` module without touching any
existing belief-propagation code path.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pytest

from pyhgf.model import DeepNetwork
from pyhgf.typing.vectorised import Layer, LayerParams, LayerState, Network


def _make_network_no_weights(layers: tuple) -> Network:
    """Test helper: build a Network with default statics."""
    return Network(
        layers=layers,
        update_type="eHGF",
        max_posterior_precision=1e10,
    )


def test_layer_state_create_defaults():
    """LayerState.create reproduces the legacy defaults (zeros / ones)."""
    n = 5
    s = LayerState.create(n)
    # Zeros
    for fname in (
        "mean",
        "expected_mean",
        "effective_precision",
        "value_prediction_error",
        "mean_vol",
        "expected_mean_vol",
        "effective_precision_vol",
        "volatility_prediction_error",
    ):
        assert jnp.all(getattr(s, fname) == 0.0), fname
        assert getattr(s, fname).shape == (n,)
    # Ones
    for fname in (
        "precision",
        "expected_precision",
        "conditional_expected_precision",
        "precision_vol",
        "expected_precision_vol",
    ):
        assert jnp.all(getattr(s, fname) == 1.0), fname
        assert getattr(s, fname).shape == (n,)


def test_layer_params_create_defaults():
    """LayerParams.create matches the legacy class's defaults."""
    n = 4
    p = LayerParams.create(n)
    assert jnp.all(p.tonic_volatility == -4.0)
    assert jnp.all(p.tonic_volatility_vol == -4.0)
    assert jnp.all(p.volatility_coupling == 1.0)
    assert jnp.all(p.autoconnection_strength_vol == 1.0)
    for fname in LayerParams.__dataclass_fields__:
        assert getattr(p, fname).shape == (n,), fname


def test_layer_state_is_pytree():
    """LayerState is a PyTree: tree_leaves yields the array fields."""
    s = LayerState.create(3)
    leaves = jtu.tree_leaves(s)
    # 13 array fields; no statics.
    assert len(leaves) == 13
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_layer_static_fields_are_in_treedef_not_leaves():
    """Static fields on Layer must not appear as PyTree leaves."""
    layer = Layer(
        state=LayerState.create(2),
        params=LayerParams.create(2),
        weights_in=jnp.zeros((3, 2)),
        coupling_fn=jnp.tanh,
        add_constant_input=True,
        has_volatility_parent=True,
        is_input_layer=False,
        fully_connected=True,
        kind="volatile",
    )
    leaves = jtu.tree_leaves(layer)
    # 13 (state) + 4 (params) + 1 (weights_in) = 18 array leaves
    assert len(leaves) == 18
    treedef_str = str(jtu.tree_structure(layer))
    # Statics show up inside the treedef, not as leaves.
    assert "volatile" in treedef_str
    assert "tanh" in treedef_str or "PjitFunction" in treedef_str


def test_layer_weights_in_can_be_none():
    """The bottom layer carries weights_in=None (no child below)."""
    layer = Layer(
        state=LayerState.create(2),
        params=LayerParams.create(2),
        weights_in=None,
        coupling_fn=lambda x: x,
        add_constant_input=False,
        has_volatility_parent=True,
        is_input_layer=True,
        fully_connected=True,
        kind="volatile",
    )
    assert layer.weights_in is None
    # tree_leaves still flattens cleanly.
    leaves = jtu.tree_leaves(layer)
    # 13 + 4 = 17 leaves (no weights_in)
    assert len(leaves) == 17


def test_network_is_pytree_with_static_meta():
    """Network round-trips through jtu.tree_(un)flatten."""
    layer = Layer(
        state=LayerState.create(2),
        params=LayerParams.create(2),
        weights_in=None,
        coupling_fn=jnp.tanh,
        add_constant_input=False,
        has_volatility_parent=True,
        is_input_layer=True,
        fully_connected=True,
        kind="volatile",
    )
    net = _make_network_no_weights((layer,))
    leaves, treedef = jtu.tree_flatten(net)
    rebuilt = jtu.tree_unflatten(treedef, leaves)
    assert rebuilt.update_type == "eHGF"
    assert rebuilt.max_posterior_precision == 1e10
    assert rebuilt.n_layers == 1


def test_network_jit_does_not_retrace_on_array_change():
    """JIT-compiling a function of Network re-uses the trace for new array values."""
    layer = Layer(
        state=LayerState.create(3),
        params=LayerParams.create(3),
        weights_in=None,
        coupling_fn=jnp.tanh,
        add_constant_input=False,
        has_volatility_parent=True,
        is_input_layer=True,
        fully_connected=True,
        kind="volatile",
    )
    net = _make_network_no_weights((layer,))

    @eqx.filter_jit
    def sum_means(network: Network) -> jax.Array:
        return jnp.sum(network.layers[0].state.mean)

    out1 = sum_means(net)
    # Mutate an array field via tree_at; static fields untouched.
    net2 = eqx.tree_at(lambda n: n.layers[0].state.mean, net, jnp.ones(3) * 2.0)
    out2 = sum_means(net2)
    assert float(out1) == 0.0
    assert float(out2) == 6.0


def test_serialisation_roundtrip(tmp_path):
    """tree_serialise_leaves / tree_deserialise_leaves preserves values."""
    layer = Layer(
        state=LayerState.create(2),
        params=LayerParams.create(2),
        weights_in=jnp.arange(6.0).reshape(3, 2),
        coupling_fn=jnp.tanh,
        add_constant_input=False,
        has_volatility_parent=True,
        is_input_layer=False,
        fully_connected=True,
        kind="volatile",
    )
    net = _make_network_no_weights((layer,))
    path = tmp_path / "net.eqx"
    eqx.tree_serialise_leaves(str(path), net)
    restored = eqx.tree_deserialise_leaves(str(path), net)
    assert jnp.array_equal(restored.layers[0].weights_in, net.layers[0].weights_in)
    assert restored.update_type == net.update_type


def test_deepnetwork_state_is_eqx_network():
    """After Phase 2, ``DeepNetwork.state`` is an Equinox ``Network`` PyTree."""
    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .add_layer(size=1)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    assert isinstance(dn.state, Network)
    assert dn.state.n_layers == 3
    assert dn.state.get_layer_sizes() == [2, 3, 1]
    # Layer 0: no child below; layer 1 holds weights between layers 0 and 1.
    assert dn.state.layers[0].weights_in is None
    assert dn.state.layers[0].is_input_layer is True
    assert dn.state.layers[1].weights_in is not None
    assert dn.state.layers[1].weights_in.shape == (2, 4)  # (prev_size, size + bias)
    # Statics propagated from the builder.
    assert dn.state.update_type == "unbounded"
    assert dn.state.max_posterior_precision == 1e10


def test_optax_handles_none_slot_in_weights_tuple():
    """``Network.weights_tuple()`` has ``None`` at index 0; optax handles it.

    Phase 3 carries the optimiser state alongside the network in
    ``jax.lax.scan``. ``optax.init`` / ``optax.update`` /
    ``optax.apply_updates`` must all walk past the ``None`` slot without
    crashing — none of the moment buffers should be created for the bottom
    layer (no weights below it). Round-trip the empty slot through the full
    optax cycle and assert it stays ``None`` after both ``init`` and
    ``apply_updates``.
    """
    import optax

    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .add_layer(size=1)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    weights = dn.state.weights_tuple()
    assert weights[0] is None
    assert weights[1].shape == (2, 4)
    assert weights[2].shape == (3, 2)

    optim = optax.adam(1e-3)
    opt_state = optim.init(weights)

    # Build a non-trivial gradient with the same tree shape (None preserved).
    grads = (None, jnp.ones_like(weights[1]), jnp.ones_like(weights[2]))
    updates, new_opt_state = optim.update(grads, opt_state, weights)
    new_weights = optax.apply_updates(weights, updates)

    # `None` propagates through init/update/apply, never blows up.
    assert new_weights[0] is None
    assert new_weights[1].shape == weights[1].shape
    assert new_weights[2].shape == weights[2].shape
    # Adam moves the weight slightly negative (descent on ones gradient).
    assert float(new_weights[1].mean()) < float(weights[1].mean())


def test_deepnetwork_weights_property_legacy_shape():
    """``network.weights`` exposes the legacy tuple-of-matrices view."""
    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .add_layer(size=1)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    weights = dn.state.weights
    # Length n_layers - 1 (layer 0 has no weights_in).
    assert len(weights) == 2
    # weights[i] == layers[i+1].weights_in.
    np.testing.assert_array_equal(weights[0], dn.state.layers[1].weights_in)
    np.testing.assert_array_equal(weights[1], dn.state.layers[2].weights_in)


# ---------------------------------------------------------------------------
# Phase 5 — builder ergonomics
# ---------------------------------------------------------------------------


def test_phase5_eager_state_init_after_add_layer():
    """``dn.state`` is populated on every ``add_layer``, not lazily."""
    dn = DeepNetwork()
    assert dn.state is None  # empty builder: nothing to initialise yet

    dn.add_layer(size=2)
    assert isinstance(dn.state, Network)
    assert dn.state.n_layers == 1
    assert dn.state.get_layer_sizes() == [2]

    dn.add_layer(size=3)
    assert dn.state.n_layers == 2
    assert dn.state.get_layer_sizes() == [2, 3]


def test_phase5_weight_initialisation_requires_key_not_seed():
    """The legacy ``seed=int`` argument is removed; ``key=`` is the only path."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)

    # `key=` is the supported API and is deterministic.
    dn.weight_initialisation("xavier", key=jax.random.key(7))
    weights_a = np.asarray(dn.state.weights[0])

    dn.weight_initialisation("xavier", key=jax.random.key(7))
    weights_b = np.asarray(dn.state.weights[0])
    np.testing.assert_array_equal(weights_a, weights_b)

    # `seed=` is no longer accepted.
    with pytest.raises(TypeError):
        dn.weight_initialisation("xavier", seed=7)


def test_phase5_predict_free_function_matches_method():
    """``predict(network, x)`` (free fn) returns the same as ``dn.predict(x)``."""
    from pyhgf.model import predict

    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .add_layer(size=2)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    x_single = jnp.array([0.5, -0.5])
    np.testing.assert_array_equal(
        np.asarray(predict(dn.state, x_single)),
        np.asarray(dn.predict(np.asarray(x_single))),
    )


def test_phase5_weight_initialisation_before_any_layer_raises():
    """Calling ``weight_initialisation`` before any layer raises ``ValueError``."""
    dn = DeepNetwork()
    with pytest.raises(ValueError, match="at least one layer"):
        dn.weight_initialisation("xavier", key=jax.random.key(0))


# ---------------------------------------------------------------------------
# Phase 6 — selective recording, save/load, vmap ensembles, pandas export.
# ---------------------------------------------------------------------------


def test_phase6_record_selective_fields_only_in_trajectories():
    """``fit(record=("expected_mean",))`` records only that field."""
    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    x = np.random.RandomState(0).randn(5, 3).astype(np.float32)
    y = np.random.RandomState(1).randn(5, 2).astype(np.float32)
    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), record=("expected_mean", "precision"))
    assert set(dn.trajectories) == {"expected_mean", "precision"}
    # Per-layer T-leading arrays.
    assert dn.trajectories["expected_mean"][0].shape == (5, 2)
    assert dn.trajectories["precision"][1].shape == (5, 3)


def test_phase6_record_unknown_field_raises():
    """``fit(record=("nope",))`` raises ``ValueError`` at validation time."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    x = np.zeros((3, 3), dtype=np.float32)
    y = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown record field"):
        dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), record=("nope",))


def test_phase6_record_all_constant_matches_all_layerstate_fields():
    """``RECORD_ALL`` covers every ``LayerState`` field."""
    from pyhgf.typing.vectorised import RECORD_ALL

    assert set(RECORD_ALL) == set(LayerState.__dataclass_fields__.keys())


def test_phase6_save_load_roundtrip(tmp_path):
    """`save` then `load` into a fresh-but-identical topology restores weights."""
    key0 = jax.random.key(7)
    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .add_layer(size=1)
        .weight_initialisation("xavier", key=key0)
    )
    path = tmp_path / "dn.eqx"
    dn.save(path)
    weights_before = [np.asarray(w) for w in dn.state.weights]

    # Rebuild a fresh DeepNetwork with the same topology (uniform weights to
    # start), then load.
    fresh = DeepNetwork().add_layer(size=2).add_layer(size=3).add_layer(size=1)
    # Uniform weights, distinct from the saved ones.
    assert not np.array_equal(np.asarray(fresh.state.weights[0]), weights_before[0])

    fresh.load(path)
    for before, after in zip(weights_before, fresh.state.weights):
        np.testing.assert_array_equal(before, np.asarray(after))


def test_phase6_vmap_ensemble_run_scan_runs_n_networks_in_parallel():
    """``eqx.filter_vmap(run_scan)`` works on a batched (Network, opt_state).

    Demonstrates the ensemble-training capability unlocked by the Equinox
    refactor: stack N independent networks on a leading axis, run them in
    a single vmapped scan call.
    """
    from pyhgf.utils.vectorized_belief_propagation import run_scan

    keys = jax.random.split(jax.random.key(0), 3)
    nets = [
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .weight_initialisation("xavier", key=k)
        for k in keys
    ]
    # Stack the individual ``Network`` PyTrees along a new leading axis.
    stacked_network = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *(n.state for n in nets)
    )
    optimizer = optax.sgd(0.1)
    # Each network gets its own opt_state.
    stacked_opt_state = jax.vmap(optimizer.init)(stacked_network.weights_tuple())

    x = jnp.zeros((4, 3), dtype=jnp.float32)
    y = jnp.zeros((4, 2), dtype=jnp.float32)

    ensemble_step = eqx.filter_vmap(run_scan, in_axes=(0, None, None, None, None, None))
    (final_network, _), preds = ensemble_step(
        (stacked_network, stacked_opt_state),
        (x, y),
        optimizer,
        "precision_weighted",
        True,
        (),
    )
    # 3 networks × 4 steps × 2 nodes in layer 0.
    assert preds.shape == (3, 4, 2)
    # Each network's state stays distinct (the weights diverged from the
    # stacked init).
    assert final_network.layers[1].weights_in.shape == (3, 2, 4)


def test_phase6_to_pandas_flattens_trajectories():
    """``DeepNetwork.to_pandas`` exposes the trajectory dict as a wide table."""
    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    x = np.random.RandomState(0).randn(4, 3).astype(np.float32)
    y = np.random.RandomState(1).randn(4, 2).astype(np.float32)
    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), record=("expected_mean",))
    df = dn.to_pandas()
    # 4 time steps × (layer 0: 2 nodes + layer 1: 3 nodes) = 5 columns.
    assert df.shape == (4, 5)
    assert set(df.columns) == {
        "L0_N0_expected_mean",
        "L0_N1_expected_mean",
        "L1_N0_expected_mean",
        "L1_N1_expected_mean",
        "L1_N2_expected_mean",
    }


def test_phase6_to_pandas_without_record_raises():
    """`to_pandas` with no trajectories raises ``ValueError``."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    with pytest.raises(ValueError, match="record"):
        dn.to_pandas()


# ---------------------------------------------------------------------------
# Phase 8 — LayerStack + scan dispatch.
# ---------------------------------------------------------------------------


def test_phase8_stack_layers_builds_consistent_pytree():
    """``stack_layers`` produces a ``LayerStack`` with the right leaf shapes."""
    from pyhgf.typing.vectorised import LayerStack, stack_layers

    coupling = jax.nn.leaky_relu
    layers = [
        Layer(
            state=LayerState.create(5),
            params=LayerParams.create(5),
            weights_in=jnp.zeros((5, 6)),
            coupling_fn=coupling,
            add_constant_input=True,
            has_volatility_parent=True,
            is_input_layer=False,
            fully_connected=True,
            kind="volatile",
        )
        for _ in range(4)
    ]
    stack = stack_layers(layers)
    assert isinstance(stack, LayerStack)
    assert stack.n_layers == 4
    assert stack.state.mean.shape == (4, 5)
    assert stack.weights_in.shape == (4, 5, 6)
    assert stack.coupling_fn is coupling


def test_phase8_stack_layers_rejects_mismatched_statics():
    """``stack_layers`` rejects layers with differing static fields."""
    from pyhgf.typing.vectorised import stack_layers

    coupling = jax.nn.leaky_relu
    layer_a = Layer(
        state=LayerState.create(5),
        params=LayerParams.create(5),
        weights_in=jnp.zeros((5, 6)),
        coupling_fn=coupling,
        add_constant_input=True,
        has_volatility_parent=True,
        is_input_layer=False,
        fully_connected=True,
        kind="volatile",
    )
    layer_b = dataclasses.replace(layer_a, has_volatility_parent=False)
    with pytest.raises(ValueError, match="has_volatility_parent"):
        stack_layers([layer_a, layer_b])


def test_phase8_add_layer_stack_auto_collapses_into_layerstack():
    """``add_layer_stack`` auto-collapses ≥5 identical layers into a ``LayerStack``."""
    from pyhgf.typing.vectorised import LayerStack

    dn = (
        DeepNetwork(coupling_fn=jax.nn.leaky_relu, update_type="unbounded")
        .add_layer(size=1, kind="binary")
        .add_layer(
            size=6,
            add_constant_input=True,
            tonic_volatility=-4.0,
            volatility_parent=False,
        )
        .add_layer_stack(
            layer_sizes=[6] * 5,
            add_constant_input=True,
            tonic_volatility=-4.0,
            tonic_volatility_vol=-8.0,
            volatility_parent=True,
        )
        .add_layer(size=2, add_constant_input=False, coupling_fn=lambda x: x)
        .weight_initialisation("he", key=jax.random.key(0))
    )
    assert dn.state.n_layers == 4
    assert dn.state.n_total_slices == 8
    types = [type(e).__name__ for e in dn.state.layers]
    assert types == ["Layer", "Layer", "LayerStack", "Layer"]
    stack = dn.state.layers[2]
    assert isinstance(stack, LayerStack)
    assert stack.n_layers == 5
    assert stack.weights_in.shape == (5, 6, 7)


def _phase8_build(scan, depth=5, width=6, seed=0):
    """Build either the unrolled or auto-scanned version of the same test network.

    ``depth`` defaults to 5 so the scanned branch trips the ``_SCAN_AUTO_THRESHOLD`` and
    the resulting network actually contains a ``LayerStack`` element to compare against
    the unrolled twin.
    """
    net = (
        DeepNetwork(coupling_fn=jax.nn.leaky_relu, update_type="unbounded")
        .add_layer(size=1, kind="binary")
        .add_layer(
            size=width,
            add_constant_input=True,
            tonic_volatility=-4.0,
            volatility_parent=False,
        )
    )
    if scan:
        net = net.add_layer_stack(
            layer_sizes=[width] * depth,
            add_constant_input=True,
            tonic_volatility=-4.0,
            tonic_volatility_vol=-8.0,
            volatility_parent=True,
        )
    else:
        for _ in range(depth):
            net = net.add_layer(
                size=width,
                add_constant_input=True,
                tonic_volatility=-4.0,
                tonic_volatility_vol=-8.0,
                volatility_parent=True,
            )
    net = net.add_layer(size=2, add_constant_input=False, coupling_fn=lambda x: x)
    return net.weight_initialisation("he", key=jax.random.key(seed))


def test_phase8_parity_predict_before_training():
    """Scanned and unrolled networks produce byte-identical predictions at init."""
    u, s = _phase8_build(scan=False), _phase8_build(scan=True)
    rng = np.random.default_rng(0)
    X = jnp.array(rng.standard_normal((16, 2)), dtype=jnp.float32)
    diff = float(np.abs(np.asarray(u.predict(X)) - np.asarray(s.predict(X))).max())
    assert diff < 1e-6, f"prediction parity broke: max abs diff = {diff}"


def test_phase8_parity_fit_one_epoch():
    """Scanned and unrolled networks produce byte-identical fits over one epoch."""
    u, s = _phase8_build(scan=False), _phase8_build(scan=True)
    rng = np.random.default_rng(1)
    X = jnp.array(rng.standard_normal((20, 2)), dtype=jnp.float32)
    y = jnp.array((rng.uniform(size=(20,)) > 0.5).astype(np.float32).reshape(-1, 1))
    u.fit(X, y, optimizer=optax.sgd(0.05), learning_kind="standard", time_step=1e-2)
    s.fit(X, y, optimizer=optax.sgd(0.05), learning_kind="standard", time_step=1e-2)

    # Predictions during the scan step (per-sample output collected by run_scan).
    pred_diff = float(
        np.abs(np.asarray(u.predictions) - np.asarray(s.predictions)).max()
    )
    assert pred_diff < 1e-6, (
        f"per-step predictions diverged: max abs diff = {pred_diff}"
    )

    # Forward pass after training on a fresh input — also parity-clean.
    fwd_diff = float(np.abs(np.asarray(u.predict(X)) - np.asarray(s.predict(X))).max())
    assert fwd_diff < 1e-6, f"post-fit predictions diverged: max abs diff = {fwd_diff}"


def test_phase8_parity_record_trajectory():
    """Recording trajectories from a scanned network matches the unrolled one."""
    from pyhgf.typing.vectorised import RECORD_ALL

    depth = 5
    u = _phase8_build(scan=False, depth=depth, width=6)
    s = _phase8_build(scan=True, depth=depth, width=6)
    rng = np.random.default_rng(2)
    X = jnp.array(rng.standard_normal((8, 2)), dtype=jnp.float32)
    y = jnp.array((rng.uniform(size=(8,)) > 0.5).astype(np.float32).reshape(-1, 1))
    u.fit(
        X,
        y,
        optimizer=optax.sgd(0.0),
        learning_kind="standard",
        record=RECORD_ALL,
        time_step=1e-2,
    )
    s.fit(
        X,
        y,
        optimizer=optax.sgd(0.0),
        learning_kind="standard",
        record=RECORD_ALL,
        time_step=1e-2,
    )

    # Unrolled: depth+3 elements [binary, transition, h0..h{depth-1}, input]
    # all (T, n_nodes). Scanned: 4 elements [binary, transition, stack, input];
    # the stack is (T, N=depth, n_nodes).
    em_u = u.trajectories["expected_mean"]
    em_s = s.trajectories["expected_mean"]
    assert len(em_u) == depth + 3
    assert len(em_s) == 4
    # Boundary layers match directly.
    assert np.allclose(np.asarray(em_u[0]), np.asarray(em_s[0]), atol=1e-6)
    assert np.allclose(np.asarray(em_u[1]), np.asarray(em_s[1]), atol=1e-6)
    assert np.allclose(np.asarray(em_u[-1]), np.asarray(em_s[-1]), atol=1e-6)
    # The stack at scanned-index 2 unrolls to layers 2 .. depth+1 slice-by-slice.
    stack_traj = np.asarray(em_s[2])  # shape (T, N=depth, n_nodes)
    for k in range(depth):
        unrolled_idx = 2 + k
        assert np.allclose(
            stack_traj[:, k, :], np.asarray(em_u[unrolled_idx]), atol=1e-6
        ), f"stack slice {k} != unrolled layer {unrolled_idx}"


def test_phase8_auto_scan_skips_ineligible_configs():
    """When eligibility fails, ``add_layer_stack`` silently falls back to unrolled."""
    # Mixed widths: cannot collapse into a uniform-stack PyTree.
    net = DeepNetwork().add_layer(size=4).add_layer_stack(layer_sizes=[4, 8, 8, 8, 8])
    assert net.scan_blocks == []

    # Width mismatch with the layer immediately below.
    net = DeepNetwork().add_layer(size=4).add_layer_stack(layer_sizes=[6] * 5)
    assert net.scan_blocks == []

    # Stack would sit directly above a binary leaf — scan body uses the
    # value-prediction kernel and would corrupt the binary update.
    net = (
        DeepNetwork()
        .add_layer(size=1, kind="binary")
        .add_layer_stack(layer_sizes=[1] * 5)
    )
    assert net.scan_blocks == []

    # Below the threshold even with all other constraints met.
    net = DeepNetwork().add_layer(size=4).add_layer_stack(layer_sizes=[4] * 4)
    assert net.scan_blocks == []
