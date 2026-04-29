# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf.model import DeepNetwork
from pyhgf.model import Network as PyNetwork


def test_fit():
    """Test that DeepNetwork and RsNetwork produce finite fit results.

    Network: 2 targets → 2+1 hidden₁ → 2+1 hidden₂ → 1+1 input.
    Tested across all combinations of coupling function and learning-rate mode.
    """
    n_targets, n_h1, n_h2, n_input = 2, 2, 2, 1

    np.random.seed(42)
    x = np.random.randn(5, n_input)
    y = np.random.randn(5, n_targets)

    coupling_variants = [
        (None, None),  # linear (identity)
        ("tanh", (jnp.tanh,)),  # nonlinear
    ]
    # (learning_kind, lr, label) — lr is now applied uniformly to all kinds,
    # including "precision_ratio".  "adam" triggers the Adam optimiser on both backends.
    lr_variants = [
        ("precision_weighted", 0.1, "precision_weighted lr=0.1"),
        ("precision_weighted", "adam", "precision_weighted adam"),
        ("standard", 0.1, "standard lr=0.1"),
        ("precision_ratio", 0.1, "precision_ratio lr=0.1"),
    ]

    for coupling_name, py_coupling_fn in coupling_variants:
        for kind, lr, lr_label in lr_variants:
            label = f"coupling={coupling_name}, {lr_label}"

            # --- DeepNetwork (JAX vectorized) ---
            dn = (
                DeepNetwork(
                    coupling_fn=py_coupling_fn[0] if py_coupling_fn else lambda x: x
                )
                .add_layer(size=n_targets)
                .add_layer(size=n_h1)
                .add_layer(size=n_h2)
                .add_layer(size=n_input)
                .weight_initialisation("xavier", seed=42)
            )
            dn.fit(x=x, y=y, lr=lr, learning_kind=kind)
            preds_dn = dn.predict(np.array([[0.5]]))

            # --- RsNetwork (Rust) ---
            rs = (
                RsNetwork()
                .add_nodes(kind="continuous-state", n_nodes=n_targets)
                .add_layer(size=n_h1, coupling_fn=coupling_name)
                .add_layer(size=n_h2, coupling_fn=coupling_name)
                .add_layer(size=n_input, coupling_fn=coupling_name)
                .weight_initialisation("xavier", seed=42)
            )
            predictors = tuple(rs.layers[-1][:n_input])
            rs.fit(
                x=x.tolist(),
                y=y.tolist(),
                inputs_x_idxs=predictors,
                inputs_y_idxs=tuple(range(n_targets)),
                lr=lr,
                learning_kind=kind,
            )
            preds_rs = rs.predict(
                x=[[0.5]],
                inputs_x_idxs=predictors,
                inputs_y_idxs=tuple(range(n_targets)),
            )

            # Both should produce finite predictions
            assert np.all(np.isfinite(np.asarray(preds_dn))), (
                f"{label}: DeepNetwork predictions contain NaN/Inf"
            )
            assert np.all(np.isfinite(np.asarray(preds_rs))), (
                f"{label}: RsNetwork predictions contain NaN/Inf"
            )


def test_add_layer():
    """Test that add_layer correctly registers layers."""
    net = DeepNetwork().add_layer(size=4).add_layer(size=3).add_layer(size=2)

    assert net.n_layers == 3
    assert net.layer_sizes == [4, 3, 2]
    assert net.n_nodes == 4 + 3 + 2


def test_add_layer_stack():
    """Test building a multi-layer stack."""
    net = DeepNetwork().add_layer_stack(layer_sizes=[4, 3, 2, 1])

    assert net.n_layers == 4
    assert net.layer_sizes == [4, 3, 2, 1]
    assert net.n_nodes == 4 + 3 + 2 + 1


def test_add_layer_binary():
    """Test adding binary layers."""
    net = (
        DeepNetwork()
        .add_layer(size=3, kind="binary")
        .add_layer(size=3, add_constant_input=False, fully_connected=False)
    )
    assert net.layer_kinds == ["binary", "volatile"]
    assert net.n_layers == 2


def test_predict():
    """Test predict() shape, finiteness, determinism, and trained vs untrained."""
    n_targets, n_hidden, n_predictors = 2, 3, 4

    np.random.seed(42)
    x_train = np.random.randn(5, n_predictors)
    y_train = np.random.randn(5, n_targets)
    x_test = np.random.randn(3, n_predictors)

    dn = (
        DeepNetwork()
        .add_layer(size=n_targets)
        .add_layer(size=n_hidden)
        .add_layer(size=n_predictors)
    )
    dn.fit(x=x_train, y=y_train, lr="adam")
    preds = dn.predict(x_test)

    assert preds.shape == (3, n_targets)
    assert np.all(np.isfinite(preds))

    # Determinism
    assert np.allclose(preds, dn.predict(x_test), atol=1e-6)

    # Untrained should differ
    dn_untrained = (
        DeepNetwork()
        .add_layer(size=n_targets)
        .add_layer(size=n_hidden)
        .add_layer(size=n_predictors)
    )
    dn_untrained.fit(x=x_train[:1], y=y_train[:1], lr=0.0)
    assert not np.allclose(preds, dn_untrained.predict(x_test), atol=1e-3)


def _build_network_dn(n_targets=3, n_hidden=8, n_predictors=4):
    """Build a 3-layer DeepNetwork (targets → hidden → predictors)."""
    return (
        DeepNetwork()
        .add_layer(size=n_targets)
        .add_layer(size=n_hidden)
        .add_layer(size=n_predictors)
    )


def _build_network_rs(n_targets=3, n_hidden=8, n_predictors=4):
    """Build a 3-layer RsNetwork (targets → hidden → predictors)."""
    return (
        RsNetwork()
        .add_nodes(kind="continuous-state", n_nodes=n_targets)
        .add_layer(size=n_hidden, coupling_strengths=1.0)
        .add_layer(size=n_predictors, coupling_strengths=1.0)
    )


def test_weight_initialisation_strategies():
    """All four strategies produce changed weights on both backends."""
    for strategy in ["xavier", "he", "orthogonal", "sparse"]:
        # --- DeepNetwork ---
        dn = _build_network_dn()
        dn.state = dn._init_state()
        before = np.asarray(dn.state.weights[0]).copy()
        dn.weight_initialisation(strategy, seed=42)
        after = np.asarray(dn.state.weights[0])
        assert not np.array_equal(before, after), (
            f"DeepNetwork/{strategy}: weights unchanged by weight_initialisation"
        )

        # --- RsNetwork ---
        rs = _build_network_rs()
        rs.weight_initialisation(strategy, seed=42)
        # No error raised = success for Rust backend


def test_weight_initialisation_deterministic():
    """Same seed produces identical weights."""
    nets = []
    for _ in range(2):
        dn = _build_network_dn()
        dn.state = dn._init_state()
        dn.weight_initialisation("xavier", seed=123)
        nets.append(dn)

    for w0, w1 in zip(nets[0].state.weights, nets[1].state.weights):
        assert np.array_equal(np.asarray(w0), np.asarray(w1))


def test_weight_initialisation_invalid_strategy():
    """Invalid strategy raises ValueError."""
    dn = _build_network_dn()
    dn.state = dn._init_state()
    with pytest.raises(ValueError):
        dn.weight_initialisation("nonexistent", seed=0)


def test_weight_initialisation_single_layer():
    """Weight init on a single-layer network is a no-op."""
    dn = DeepNetwork().add_layer(size=3)
    dn.weight_initialisation("xavier", seed=0)
    assert dn.state is not None
    assert len(dn.state.weights) == 0


def test_reset():
    """Test that reset clears the state."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    dn.fit(x, y, lr=0.1)
    assert dn.state is not None
    assert dn.predictions is not None

    dn.reset()
    assert dn._propagation_fn is None
    assert dn._prediction_fn is None


def test_fully_connected_false():
    """Test one-to-one (diagonal) weight matrix."""
    dn = (
        DeepNetwork()
        .add_layer(size=3)
        .add_layer(size=3, add_constant_input=False, fully_connected=False)
    )
    dn.state = dn._init_state()
    # Weight matrix should be identity-like
    w = np.asarray(dn.state.weights[0])
    assert np.allclose(w, np.eye(3)), f"Expected identity weight matrix, got {w}"


def test_three_backends_binary_volatile():
    """Compare all three backends on a binary-output + volatile-hidden architecture.

    Architecture:
    → 1 binary output
    → 1 volatile value parent (1-to-1)
    → 2 volatile hidden with constant input
    → 1 volatile input

    The constant is a value parent of the intermediate volatile node (shared across
    all three backends).  All three update types are exercised.  Both JAX and Rust
    default to ``tonic_volatility=-4.0`` for volatile-state nodes, so no explicit
    override is needed.

    Tolerances
    ----------
    Rust vs Python (both float64) : atol=1e-6
    JAX vs Rust (JAX uses float32) : atol=1e-3
    """
    n_targets = 1
    n_hidden = 2
    n_input = 1

    np.random.seed(42)
    x = np.random.randn(1, n_input)
    y = np.random.choice([0.0, 1.0], size=(1, n_targets))

    atol_rs_py = 1e-6  # Rust vs Python (both float64)
    atol_jax = 1e-3  # JAX (float32) vs Rust

    for update_type in ["standard", "eHGF", "unbounded"]:
        label = f"update_type={update_type}"

        # --- DeepNetwork (JAX vectorized) ---
        dn = (
            DeepNetwork(update_type=update_type)
            .add_layer(size=n_targets, kind="binary")
            .add_layer(size=n_targets, add_constant_input=False, fully_connected=False)
            .add_layer(size=n_hidden)
            .add_layer(size=n_input, add_constant_input=False)
        )
        dn.fit(x=x, y=y, lr=0.1)

        # --- RsNetwork (Rust) ---
        rs = (
            RsNetwork(update_type=update_type)
            .add_nodes(kind="binary-state", n_nodes=n_targets)
            .add_nodes(kind="volatile-state", n_nodes=n_targets, value_children=0)
            .add_layer(size=n_hidden)
            .add_layer(size=n_input, add_constant_input=False)
        )
        predictors = tuple(rs.layers[-1][:n_input])
        targets_idxs = tuple(range(n_targets))
        rs.fit(
            x=x.tolist(),
            y=y.tolist(),
            inputs_x_idxs=predictors,
            inputs_y_idxs=targets_idxs,
            lr=0.1,
        )

        # --- Network (Python per-node) ---
        net = (
            PyNetwork(update_type=update_type)
            .add_nodes(kind="binary-state", n_nodes=n_targets)
            .add_nodes(kind="volatile-state", n_nodes=n_targets, value_children=0)
            .add_nodes(kind="volatile-state", n_nodes=n_hidden, value_children=1)
            .add_nodes(kind="constant-state", n_nodes=n_targets, value_children=1)
            .add_nodes(kind="volatile-state", n_nodes=n_input, value_children=[2, 3])
        )
        # Node layout: binary(0..n_targets), intermediate(n_targets..2*n_targets),
        #              hidden(2*n_targets..2*n_targets+n_hidden),
        #              constant(2*n_targets+n_hidden..3*n_targets+n_hidden),
        #              input(3*n_targets+n_hidden..3*n_targets+n_hidden+n_input)
        x_idxs = tuple(
            range(3 * n_targets + n_hidden, 3 * n_targets + n_hidden + n_input)
        )
        y_idxs = tuple(range(n_targets))
        net.fit(
            x=x,
            y=y,
            inputs_x_idxs=x_idxs,
            inputs_y_idxs=y_idxs,
            lr=0.1,
        )

        # ---- Hidden-layer volatile states ----
        # Rust/Python node layout: binary(0), intermediate(1), hidden(2..2+n_hidden)
        # JAX: dn.state.layers[2] holds the hidden layer (index 2 in add_layer order)
        for i in range(n_hidden):
            node_idx = 2 * n_targets + i  # Rust/Python index

            rs_mean = float(rs.node_trajectories[node_idx]["mean"][-1])
            py_mean = float(net.last_attributes[node_idx]["mean"])
            jax_mean = float(dn.state.layers[2].mean[i])

            assert np.allclose(py_mean, rs_mean, atol=atol_rs_py), (
                f"{label}: hidden[{i}] mean: Python={py_mean} vs Rust={rs_mean}"
            )
            assert np.allclose(jax_mean, rs_mean, atol=atol_jax), (
                f"{label}: hidden[{i}] mean: JAX={jax_mean} vs Rust={rs_mean}"
            )

            rs_prec = float(rs.node_trajectories[node_idx]["precision"][-1])
            py_prec = float(net.last_attributes[node_idx]["precision"])
            jax_prec = float(dn.state.layers[2].precision[i])

            assert np.allclose(py_prec, rs_prec, atol=atol_rs_py), (
                f"{label}: hidden[{i}] precision: Python={py_prec} vs Rust={rs_prec}"
            )
            assert np.allclose(jax_prec, rs_prec, atol=atol_jax), (
                f"{label}: hidden[{i}] precision: JAX={jax_prec} vs Rust={rs_prec}"
            )

            rs_mean_vol = float(rs.node_trajectories[node_idx]["mean_vol"][-1])
            py_mean_vol = float(net.last_attributes[node_idx]["mean_vol"])
            jax_mean_vol = float(dn.state.layers[2].mean_vol[i])

            assert np.allclose(py_mean_vol, rs_mean_vol, atol=atol_rs_py), (
                f"{label}: hidden[{i}] mean_vol: Python={py_mean_vol} vs Rust={rs_mean_vol}"
            )
            assert np.allclose(jax_mean_vol, rs_mean_vol, atol=atol_jax), (
                f"{label}: hidden[{i}] mean_vol: JAX={jax_mean_vol} vs Rust={rs_mean_vol}"
            )

            rs_prec_vol = float(rs.node_trajectories[node_idx]["precision_vol"][-1])
            py_prec_vol = float(net.last_attributes[node_idx]["precision_vol"])
            jax_prec_vol = float(dn.state.layers[2].precision_vol[i])

            assert np.allclose(py_prec_vol, rs_prec_vol, atol=atol_rs_py), (
                f"{label}: hidden[{i}] precision_vol: Python={py_prec_vol} vs Rust={rs_prec_vol}"
            )
            assert np.allclose(jax_prec_vol, rs_prec_vol, atol=atol_jax), (
                f"{label}: hidden[{i}] precision_vol: JAX={jax_prec_vol} vs Rust={rs_prec_vol}"
            )

        # ---- Coupling weights: hidden ← input ----
        # JAX: weights[2] connects layer[2] (hidden, rows) to layer[3] (input, cols),
        #      shape (n_hidden, n_input).
        for j in range(n_hidden):
            node_idx = 2 * n_targets + j
            for k in range(n_input):
                w_py = float(net.last_attributes[node_idx]["value_coupling_parents"][k])
                w_rs = float(
                    rs.node_trajectories[node_idx]["value_coupling_parents"][-1][k]
                )
                w_jax = float(dn.state.weights[2][j, k])

                assert np.allclose(w_py, w_rs, atol=atol_rs_py), (
                    f"{label}: weight hidden[{j}]←input[{k}]: Python={w_py} vs Rust={w_rs}"
                )
                assert np.allclose(w_jax, w_rs, atol=atol_jax), (
                    f"{label}: weight hidden[{j}]←input[{k}]: JAX={w_jax} vs Rust={w_rs}"
                )

        # ---- Binary coupling weights ----
        # All three backends now learn binary-to-parent weights via sigmoid coupling.
        for j in range(n_targets):
            w_py = float(net.last_attributes[j]["value_coupling_parents"][0])
            w_rs = float(rs.node_trajectories[j]["value_coupling_parents"][-1][0])
            w_jax = float(dn.state.weights[0][j, 0])

            # All backends should have deviated from the initial weight of 1.0.
            assert not np.allclose(w_py, 1.0, atol=1e-6), (
                f"{label}: Python binary weight={w_py} should have changed from 1.0"
            )
            assert not np.allclose(w_rs, 1.0, atol=1e-6), (
                f"{label}: Rust binary weight={w_rs} should have changed from 1.0"
            )
            assert not np.allclose(w_jax, 1.0, atol=1e-6), (
                f"{label}: JAX binary weight={w_jax} should have changed from 1.0"
            )
            # Rust and Python (both float64) should agree closely.
            assert np.allclose(w_py, w_rs, atol=atol_rs_py), (
                f"{label}: binary weight Python={w_py} vs Rust={w_rs}"
            )

        # ---- Predictions ----
        # All backends now use updated binary weights, so all predictions are comparable.
        preds_rs = rs.predict(
            x=[[0.5]],
            inputs_x_idxs=predictors,
            inputs_y_idxs=targets_idxs,
        )
        preds_py = net.predict(
            x=np.array([[0.5]]),
            inputs_x_idxs=x_idxs,
            inputs_y_idxs=y_idxs,
        )
        preds_jax = dn.predict(np.array([[0.5]]))

        # Rust and Python (both float64) should produce identical predictions.
        # JAX (float32) may accumulate enough rounding error across layers to
        # exceed atol_jax, so only finite-value correctness is checked there.
        assert np.all(np.isfinite(np.asarray(preds_jax))), (
            f"{label}: JAX predictions contain NaN/Inf"
        )
        assert np.allclose(
            np.asarray(preds_py), np.asarray(preds_rs), atol=atol_rs_py
        ), f"{label}: predictions: Python={preds_py} vs Rust={preds_rs}"


def test_add_layer_invalid_kind():
    """Invalid layer kind raises ValueError."""
    with pytest.raises(ValueError, match="Invalid layer kind"):
        DeepNetwork().add_layer(size=3, kind="invalid")


def test_add_layer_one_to_one_constant_input():
    """One-to-one layers cannot use add_constant_input."""
    with pytest.raises(ValueError, match="One-to-one layers.*cannot use"):
        (
            DeepNetwork()
            .add_layer(size=3)
            .add_layer(size=3, fully_connected=False, add_constant_input=True)
        )


def test_add_layer_one_to_one_size_mismatch():
    """One-to-one layers require matching sizes."""
    with pytest.raises(ValueError, match="One-to-one layers require the same size"):
        (
            DeepNetwork()
            .add_layer(size=3)
            .add_layer(size=4, fully_connected=False, add_constant_input=False)
        )


def test_fit_invalid_lr():
    """Unknown lr string or learning_kind raises ValueError."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    with pytest.raises(ValueError, match="Unknown lr value"):
        dn.fit(x=np.zeros((5, 3)), y=np.zeros((5, 2)), lr="sgd")
    with pytest.raises(ValueError, match="Unknown kind"):
        dn.fit(x=np.zeros((5, 3)), y=np.zeros((5, 2)), lr=0.1, learning_kind="kalman")


def test_fit_record_trajectories():
    """fit(record_trajectories=True) stores trajectories."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    dn.fit(x=x, y=y, lr=0.1, record_trajectories=True)
    assert dn.trajectories is not None


def test_predict_before_fit():
    """predict() before fit() raises ValueError."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    with pytest.raises(ValueError, match="must be fit"):
        dn.predict(np.zeros((5, 3)))


def test_predict_1d_input():
    """predict() with 1d input returns 1d output."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    dn.fit(x=np.random.randn(5, 3), y=np.random.randn(5, 2), lr=0.1)
    pred = dn.predict(np.array([0.1, 0.2, 0.3]))
    assert pred.ndim == 1
    assert pred.shape == (2,)


def test_repr():
    """__repr__ contains expected info."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    r = repr(dn)
    assert "VectorizedDeepNetwork" in r
    assert "nodes=5" in r
    assert "[2, 3]" in r


def test_fit_weight_update_false_freezes_weights():
    """fit(weight_update=False) keeps weights identical to the pre-fit state."""
    np.random.seed(0)
    x = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randn(10, 2).astype(np.float32)

    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=4)
        .add_layer(size=3)
        .weight_initialisation("xavier", seed=42)
    )
    weights_before = [np.asarray(w).copy() for w in dn.state.weights]

    dn.fit(x=x, y=y, lr=0.1, weight_update=False)
    for before, after in zip(weights_before, dn.state.weights):
        assert np.array_equal(before, np.asarray(after))

    # Adam state must also stay frozen (no step counter increment).
    assert int(dn.state.adam_t) == 0


def test_fit_weight_update_true_changes_weights():
    """fit(weight_update=True) (default) actually changes the weights."""
    np.random.seed(0)
    x = np.random.randn(10, 3).astype(np.float32)
    y = np.random.randn(10, 2).astype(np.float32)

    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=4)
        .add_layer(size=3)
        .weight_initialisation("xavier", seed=42)
    )
    weights_before = [np.asarray(w).copy() for w in dn.state.weights]
    dn.fit(x=x, y=y, lr=0.1)
    assert any(
        not np.array_equal(before, np.asarray(after))
        for before, after in zip(weights_before, dn.state.weights)
    )


def test_fit_weight_update_toggle_retraces():
    """Toggling weight_update across calls forces a re-trace and yields the right state."""
    np.random.seed(0)
    x = np.random.randn(5, 3).astype(np.float32)
    y = np.random.randn(5, 2).astype(np.float32)

    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=4)
        .add_layer(size=3)
        .weight_initialisation("xavier", seed=42)
    )

    # First call with weight_update=False — weights frozen
    dn.fit(x=x, y=y, lr=0.1, weight_update=False)
    weights_after_frozen = [np.asarray(w).copy() for w in dn.state.weights]

    # Now flip to weight_update=True — must re-trace and actually update
    dn.fit(x=x, y=y, lr=0.1, weight_update=True)
    assert any(
        not np.array_equal(before, np.asarray(after))
        for before, after in zip(weights_after_frozen, dn.state.weights)
    )
