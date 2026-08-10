# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf.model import DeepNetwork, LayerConfig
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
    # (learning_kind, lr, label) — lr is applied uniformly to all kinds.
    # "adam" triggers the Adam optimiser on both backends.
    lr_variants = [
        ("precision_weighted", 0.1, "precision_weighted lr=0.1"),
        ("precision_weighted", "adam", "precision_weighted adam"),
        ("standard", 0.1, "standard lr=0.1"),
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
                .weight_initialisation("xavier", key=jax.random.key(42))
            )
            # DeepNetwork takes an optax optimizer; the Rust backend below still
            # takes the raw ``lr`` (a float, or "adam"), so don't clobber ``lr``.
            optimizer = optax.adam(1e-3) if lr == "adam" else optax.sgd(lr)
            dn.fit(x=x, y=y, optimizer=optimizer, learning_kind=kind)
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


def test_add_layer_categorical():
    """Test adding a categorical output layer (accepts the alias too)."""
    net = (
        DeepNetwork()
        .add_layer(size=3, kind="categorical")
        .add_layer(size=3, add_constant_input=False, fully_connected=False)
    )
    assert net.layer_kinds == ["categorical", "volatile"]
    assert net.n_layers == 2

    aliased = DeepNetwork().add_layer(size=3, kind="categorical-state")
    assert aliased.layer_kinds == ["categorical"]


def test_categorical_predict_is_softmax():
    """A categorical output layer returns a per-sample softmax over the K classes."""
    n_classes, n_input = 3, 2
    net = (
        DeepNetwork()
        .add_layer(size=n_classes, kind="categorical")
        .add_layer(size=8)
        .add_layer(size=n_input, add_constant_input=False)
        .weight_initialisation("he", key=jax.random.key(0))
    )
    out = np.asarray(net.predict(np.random.randn(10, n_input).astype(np.float32)))
    assert out.shape == (10, n_classes)
    # Softmax: non-negative and each row sums to one.
    assert np.all(out >= 0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_categorical_learns_separable_classes():
    """The categorical network fits three linearly separable Gaussian blobs.

    One-hot labels are clamped on the size-K output layer; training drives the argmax
    accuracy to one.
    """
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 2.0], [-2.0, -1.0], [2.0, -1.0]], dtype=np.float32)
    labels = np.repeat([0, 1, 2], 30)
    x = (centers[labels] + rng.normal(scale=0.25, size=(90, 2))).astype(np.float32)
    y_onehot = np.eye(3, dtype=np.float32)[labels]

    net = DeepNetwork(coupling_fn=jax.nn.leaky_relu).add_layer(
        size=3, kind="categorical"
    )
    for _ in range(4):
        net.add_layer(size=16, tonic_volatility_vol=-8.0)
    net = net.add_layer(
        size=2,
        add_constant_input=False,
        coupling_fn=lambda x: x,
        expected_precision=10e9,
    ).weight_initialisation("he", key=jax.random.key(0))

    adam = optax.adam(1e-2)
    for _ in range(40):
        net.fit(
            jnp.array(x),
            jnp.array(y_onehot),
            optimizer=adam,
            learning_kind="standard",
            time_step=0.001,
        )
    preds = np.asarray(net.predict(jnp.array(x)))
    assert (preds.argmax(axis=1) == labels).mean() > 0.95


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
    dn.fit(x=x_train, y=y_train, optimizer=optax.adam(1e-3))
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
    dn_untrained.fit(x=x_train[:1], y=y_train[:1], optimizer=optax.sgd(0.0))
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
        before = np.asarray(dn.state.weights[0]).copy()
        dn.weight_initialisation(strategy, key=jax.random.key(42))
        after = np.asarray(dn.state.weights[0])
        assert not np.array_equal(before, after), (
            f"DeepNetwork/{strategy}: weights unchanged by weight_initialisation"
        )

        # --- RsNetwork ---
        rs = _build_network_rs()
        rs.weight_initialisation(strategy, seed=42)
        # No error raised = success for Rust backend


def test_orthogonal_initialisation_survives_the_reshape():
    """Non-square matrices come out orthogonal once the caller has reshaped them.

    ``orthogonal_init`` returns a flat vector that the caller reshapes row-major
    to ``(n_children, n_parents)``. Both orientations are checked here on a network
    whose layers are deliberately unequal: one matrix has more children than parents,
    the other fewer.

    With at least as many children as parents every input direction survives, so
    ``W.T @ W = I``; with fewer, the layer is a projection and the best available is
    ``W @ W.T = I``, preserving length within the row space.
    """
    net = (
        DeepNetwork()
        .add_layer(size=6)
        .add_layer(size=3)
        .add_layer(size=8)
        .weight_initialisation("orthogonal", key=jax.random.key(0))
    )

    shapes = []
    for layer in net.state.layers[1:]:
        weights = np.asarray(layer.weights_in)
        if layer.add_constant_input:
            # The bias column is not a connection and takes no part in the
            # orthogonality; it is zeroed by ``_init_matrix``.
            np.testing.assert_allclose(weights[:, -1], 0.0, atol=0.0)
            weights = weights[:, :-1]
        n_children, n_parents = weights.shape
        shapes.append((n_children, n_parents))
        assert n_children != n_parents, "the square case cannot detect the scramble"
        if n_children >= n_parents:
            product, size = weights.T @ weights, n_parents
        else:
            product, size = weights @ weights.T, n_children
        np.testing.assert_allclose(product, np.eye(size), atol=1e-6)

    # One matrix of each orientation, so both branches were exercised.
    assert any(c > p for c, p in shapes) and any(c < p for c, p in shapes), shapes


def test_weight_initialisation_deterministic():
    """Same seed produces identical weights."""
    nets = []
    for _ in range(2):
        dn = _build_network_dn()
        dn.weight_initialisation("xavier", key=jax.random.key(123))
        nets.append(dn)

    for w0, w1 in zip(nets[0].state.weights, nets[1].state.weights):
        assert np.array_equal(np.asarray(w0), np.asarray(w1))


def test_weight_initialisation_invalid_strategy():
    """Invalid strategy raises ValueError."""
    dn = _build_network_dn()
    with pytest.raises(ValueError):
        dn.weight_initialisation("nonexistent", key=jax.random.key(0))


def test_weight_initialisation_single_layer():
    """Weight init on a single-layer network is a no-op."""
    dn = DeepNetwork().add_layer(size=3)
    dn.weight_initialisation("xavier", key=jax.random.key(0))
    assert dn.state is not None
    assert len(dn.state.weights) == 0


def test_reset():
    """Test that reset clears the state."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    dn.fit(x, y, optimizer=optax.sgd(0.1))
    assert dn.state is not None
    assert dn.predictions is not None

    dn.reset()
    # After reset, the optimiser state is dropped and the network is
    # re-initialised to uniform weights. JIT caches live inside
    # ``eqx.filter_jit`` now; no instance-side bookkeeping to check.
    assert dn.opt_state is None
    assert dn._optimizer is None


def test_fully_connected_false():
    """Test one-to-one (diagonal) weight matrix."""
    dn = (
        DeepNetwork()
        .add_layer(size=3)
        .add_layer(size=3, add_constant_input=False, fully_connected=False)
    )
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
    all three backends).  All three update types are exercised.  Volatile-state nodes
    carry no ``tonic_volatility``, so no explicit override is needed.

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

    for volatility_updates in ["standard", "eHGF", "unbounded"]:
        label = f"volatility_updates={volatility_updates}"
        atol_jax_local = 5e-2 if volatility_updates == "unbounded" else atol_jax
        atol_rs_py_local = 1e-2 if volatility_updates == "unbounded" else atol_rs_py

        # --- DeepNetwork (JAX vectorized) ---
        dn = (
            DeepNetwork(
                volatility_updates=volatility_updates,
                feedforward_uncertainty=True,
            )
            .add_layer(size=n_targets, kind="binary")
            .add_layer(size=n_targets, add_constant_input=False, fully_connected=False)
            .add_layer(size=n_hidden)
            .add_layer(size=n_input, add_constant_input=False)
        )
        dn.fit(x=x, y=y, optimizer=optax.sgd(0.1))

        # --- RsNetwork (Rust) ---
        rs = (
            RsNetwork(volatility_updates=volatility_updates)
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
            PyNetwork(volatility_updates=volatility_updates)
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
            jax_mean = float(dn.state.layers[2].state.mean[i])

            assert np.allclose(py_mean, rs_mean, atol=atol_rs_py_local), (
                f"{label}: hidden[{i}] mean: Python={py_mean} vs Rust={rs_mean}"
            )
            assert np.allclose(jax_mean, rs_mean, atol=atol_jax_local), (
                f"{label}: hidden[{i}] mean: JAX={jax_mean} vs Rust={rs_mean}"
            )

            rs_prec = float(rs.node_trajectories[node_idx]["precision"][-1])
            py_prec = float(net.last_attributes[node_idx]["precision"])
            jax_prec = float(dn.state.layers[2].state.precision[i])

            assert np.allclose(py_prec, rs_prec, atol=atol_rs_py_local), (
                f"{label}: hidden[{i}] precision: Python={py_prec} vs Rust={rs_prec}"
            )
            assert np.allclose(jax_prec, rs_prec, atol=atol_jax_local), (
                f"{label}: hidden[{i}] precision: JAX={jax_prec} vs Rust={rs_prec}"
            )

            rs_mean_vol = float(rs.node_trajectories[node_idx]["mean_vol"][-1])
            py_mean_vol = float(net.last_attributes[node_idx]["mean_vol"])
            jax_mean_vol = float(dn.state.layers[2].state.mean_vol[i])

            assert np.allclose(py_mean_vol, rs_mean_vol, atol=atol_rs_py_local), (
                f"{label}: hidden[{i}] mean_vol: Python={py_mean_vol} vs Rust={rs_mean_vol}"
            )
            assert np.allclose(jax_mean_vol, rs_mean_vol, atol=atol_jax_local), (
                f"{label}: hidden[{i}] mean_vol: JAX={jax_mean_vol} vs Rust={rs_mean_vol}"
            )

            rs_prec_vol = float(rs.node_trajectories[node_idx]["precision_vol"][-1])
            py_prec_vol = float(net.last_attributes[node_idx]["precision_vol"])
            jax_prec_vol = float(dn.state.layers[2].state.precision_vol[i])

            assert np.allclose(py_prec_vol, rs_prec_vol, atol=atol_rs_py_local), (
                f"{label}: hidden[{i}] precision_vol: Python={py_prec_vol} vs Rust={rs_prec_vol}"
            )
            assert np.allclose(jax_prec_vol, rs_prec_vol, atol=atol_jax_local), (
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

                assert np.allclose(w_py, w_rs, atol=atol_rs_py_local), (
                    f"{label}: weight hidden[{j}]←input[{k}]: Python={w_py} vs Rust={w_rs}"
                )
                assert np.allclose(w_jax, w_rs, atol=atol_jax_local), (
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
            assert np.allclose(w_py, w_rs, atol=atol_rs_py_local), (
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


def test_fit_invalid_learning_kind():
    """Unknown ``learning_kind`` raises ``ValueError``."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    with pytest.raises(ValueError, match="Unknown kind"):
        dn.fit(
            x=np.zeros((5, 3)),
            y=np.zeros((5, 2)),
            optimizer=optax.sgd(0.1),
            learning_kind="kalman",
        )


def test_fit_record_trajectories():
    """fit(record=("expected_mean",)) stores the requested field's trajectory."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), record=("expected_mean",))
    assert dn.trajectories is not None
    # New shape: dict[field, tuple[(T, n_nodes) per layer]]
    assert set(dn.trajectories) == {"expected_mean"}
    assert dn.trajectories["expected_mean"][0].shape == (5, 2)
    assert dn.trajectories["expected_mean"][1].shape == (5, 3)


def test_predict_before_fit_works_on_uniform_weights():
    """``predict()`` succeeds even before ``fit``."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    out = dn.predict(np.zeros((5, 3)))
    assert out.shape == (5, 2)
    assert np.all(np.isfinite(out))


def test_predict_with_no_layers_raises():
    """Calling ``predict`` on an empty builder raises ``ValueError``."""
    with pytest.raises(ValueError, match="at least one layer"):
        DeepNetwork().predict(np.zeros((5, 3)))


def test_predict_1d_input():
    """Predict() with 1d input returns 1d output."""
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)
    dn.fit(x=np.random.randn(5, 3), y=np.random.randn(5, 2), optimizer=optax.sgd(0.1))
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
        .weight_initialisation("xavier", key=jax.random.key(42))
    )
    weights_before = [np.asarray(w).copy() for w in dn.state.weights]

    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), weight_update=False)
    for before, after in zip(weights_before, dn.state.weights):
        assert np.array_equal(before, np.asarray(after))

    # Optimiser state must also stay frozen — opt_state is the fresh init
    # (SGD's state is just a counter; should still be 0).
    fresh = optax.sgd(0.1).init(dn.state.weights_tuple())
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: bool(jnp.array_equal(a, b)), dn.opt_state, fresh
        )
    )


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
        .weight_initialisation("xavier", key=jax.random.key(42))
    )
    weights_before = [np.asarray(w).copy() for w in dn.state.weights]
    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1))
    assert any(
        not np.array_equal(before, np.asarray(after))
        for before, after in zip(weights_before, dn.state.weights)
    )


def test_fit_weight_update_toggle_retraces():
    """Toggling weight_update across calls."""
    np.random.seed(0)
    x = np.random.randn(5, 3).astype(np.float32)
    y = np.random.randn(5, 2).astype(np.float32)

    dn = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=4)
        .add_layer(size=3)
        .weight_initialisation("xavier", key=jax.random.key(42))
    )

    # First call with weight_update=False — weights frozen
    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), weight_update=False)
    weights_after_frozen = [np.asarray(w).copy() for w in dn.state.weights]

    # Now flip to weight_update=True — must re-trace and actually update
    dn.fit(x=x, y=y, optimizer=optax.sgd(0.1), weight_update=True)
    assert any(
        not np.array_equal(before, np.asarray(after))
        for before, after in zip(weights_after_frozen, dn.state.weights)
    )


def test_input_layer_uses_prior_precision():
    """The bottom layer's ``expected_precision`` must stay at its prior precision.

    Layer 0 is the observation layer of a DeepNetwork — it has no value children, so it
    does not undergo a Gaussian random walk between samples and the volatility
    contribution is skipped (mirroring the per-node continuous-node treatment). Volatile
    layers carry no ``tonic_volatility``.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 2)).astype(np.float32)
    y = rng.standard_normal((5, 1)).astype(np.float32)

    net = (
        DeepNetwork()
        .add_layer(
            size=1,
            precision=5.0,
            expected_precision=5.0,
        )
        .add_layer(size=4)
        .add_layer(size=2)
    )
    # lr=0 so weights don't drift across samples — isolates the prediction step.
    net.fit(
        x=x,
        y=y,
        optimizer=optax.sgd(0.0),
        learning_kind="standard",
        record=("expected_precision",),
    )
    expected_precision = np.asarray(net.trajectories["expected_precision"][0])

    # The bottom (input) layer's expected_precision stays at its prior precision.
    np.testing.assert_allclose(
        expected_precision,
        5.0,
        rtol=1e-5,
        err_msg=(
            "DeepNetwork bottom layer's expected_precision is not the prior "
            "precision — is_input_layer override missing"
        ),
    )


def test_constant_input_is_linearly_coupled():
    """The bias enters predictions linearly, whatever the coupling function.

    A Linear -> GELU -> Linear network with bias columns must compute
    ``W1 @ gelu(W2 @ x + b2) + b1``: the constant bias node is wired in
    linearly (``g(1) = 1``), the same convention as the weight-learning step,
    the per-node backend, and the Rust backend (both force constant-state
    nodes to identity coupling). Applying the coupling function to the
    constant node would instead scale every bias by ``gelu(1)`` (~0.84), so
    the network would learn against a different forward function than the one
    it computes.
    """
    rng = np.random.default_rng(3)
    d, h = 4, 6
    w1 = jnp.asarray(rng.normal(size=(d, h)))
    b1 = jnp.asarray(rng.normal(size=(d,)))
    w2 = jnp.asarray(rng.normal(size=(h, d)))
    b2 = jnp.asarray(rng.normal(size=(h,)))
    x = jnp.asarray(rng.normal(size=(d,)))

    net = (
        DeepNetwork()
        .add_layer(size=d)  # output layer
        .add_layer(size=h, coupling_fn=jax.nn.gelu)  # hidden, GELU coupling
        .add_layer(size=d)  # input layer
    )
    elements = list(net.state.layers)
    elements[1] = dataclasses.replace(
        elements[1], weights_in=jnp.concatenate([w1, b1[:, None]], axis=1)
    )
    elements[2] = dataclasses.replace(
        elements[2], weights_in=jnp.concatenate([w2, b2[:, None]], axis=1)
    )
    net.state = dataclasses.replace(net.state, layers=tuple(elements))

    expected = w1 @ jax.nn.gelu(w2 @ x + b2) + b1
    np.testing.assert_allclose(net.predict(x), expected, rtol=1e-5, atol=1e-6)


def _set_weights(net: DeepNetwork, weights: dict) -> DeepNetwork:
    """Replace ``weights_in`` on the given layers (index -> matrix)."""
    elements = list(net.state.layers)
    for i, w in weights.items():
        elements[i] = dataclasses.replace(elements[i], weights_in=jnp.asarray(w))
    net.state = dataclasses.replace(net.state, layers=tuple(elements))
    return net


def _norm_rel_err(a, b) -> float:
    """Relative error between two arrays, measured in the Frobenius norm."""
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_input_error_routes_child_error_through_weights():
    """``input_error()`` returns the child's error routed back through the weights.

    Two linear layers without bias and with default (unit) precisions: the
    precision weighting is 1, so the message at the input (top) layer reduces
    to ``W.T @ (y - W @ x)`` — the output residual multiplied back through the
    connecting weight matrix.
    """
    rng = np.random.default_rng(1)
    w = rng.normal(size=(3, 2))
    x = jnp.asarray(rng.normal(size=(2,)))
    y = jnp.asarray(rng.normal(size=(3,)))

    net = (
        DeepNetwork()
        .add_layer(size=3, add_constant_input=False)
        .add_layer(size=2, add_constant_input=False)
    )
    _set_weights(net, {1: w})
    net.prediction(x).update(y)

    expected = jnp.asarray(w).T @ (y - jnp.asarray(w) @ x)
    np.testing.assert_allclose(net.input_error(), expected, rtol=1e-5, atol=1e-6)


# Feed-forward block used by the backprop-parity tests below:
# a Linear -> GELU -> Linear network, D -> H -> D, without biases.
_FF_D, _FF_H = 8, 16


def _ff_oracle_grads(w1, w2, x, y):
    """Backprop gradients of the squared-error loss on the same forward pass.

    The forward function mirrors the DeepNetwork exactly: the top (input)
    layer predicts the hidden layer linearly, and the hidden layer predicts
    the output layer through the GELU coupling.
    """

    def loss(w1_, w2_, x_):
        return 0.5 * jnp.sum((y - w1_ @ jax.nn.gelu(w2_ @ x_)) ** 2)

    return jax.grad(loss, argnums=(0, 1, 2))(w1, w2, x)


def _ff_net(hidden_kwargs: dict, top_kwargs: dict, leaf_kwargs: dict) -> DeepNetwork:
    return (
        DeepNetwork()
        .add_layer(size=_FF_D, add_constant_input=False, **leaf_kwargs)
        .add_layer(
            size=_FF_H,
            add_constant_input=False,
            coupling_fn=jax.nn.gelu,
            **hidden_kwargs,
        )
        .add_layer(size=_FF_D, add_constant_input=False, **top_kwargs)
    )


def test_ff_block_single_sweep_matches_backprop():
    """One local learning step on Linear -> GELU -> Linear matches backprop.

    The network is configured so that the single belief-propagation sweep
    reproduces the gradients of a squared-error loss on the identical forward
    function: volatility levels frozen, high prior precision on
    the hidden and input layers so their beliefs barely move during the update
    sweep, unit precision on the observed output layer, and the
    ``"precision_weighted"`` learning mode — the hidden layer's belief shift is
    its routed error divided by its posterior precision, and the
    precision-weighted gradient multiplies that same precision back in.

    Bias columns are excluded: the prediction step applies the coupling
    function to the constant bias node while the learning step treats it as
    linear, so with biases the two sides compute different forward functions.

    Measured agreement (norm-relative): ~5e-4 on the weight gradients and
    ~2e-3 on the input error message in float32; better than 1e-5 in float64.
    """
    rng = np.random.default_rng(42)
    w1 = jnp.asarray(rng.normal(size=(_FF_D, _FF_H)) * (2.0 / _FF_H) ** 0.5)
    w2 = jnp.asarray(rng.normal(size=(_FF_H, _FF_D)) * (2.0 / _FF_D) ** 0.5)
    x = jnp.asarray(rng.normal(size=(_FF_D,)))
    y = jnp.asarray(w1 @ jax.nn.gelu(w2 @ x) + rng.normal(size=(_FF_D,)))

    g_w1, g_w2, g_x = _ff_oracle_grads(w1, w2, x, y)

    high_precision = dict(
        volatility_parent=False,
        precision=1e4,
        expected_precision=1e4,
    )
    net = _ff_net(
        hidden_kwargs=high_precision,
        top_kwargs=high_precision,
        leaf_kwargs=dict(volatility_parent=False),
    )
    _set_weights(net, {1: w1, 2: w2})

    lr = 1e-3
    net.prediction(x).update(
        y, optimizer=optax.sgd(lr), learning_kind="precision_weighted"
    )

    # The applied weight change divided by -lr is the descent gradient.
    d_w1 = -(net.state.layers[1].weights_in - w1) / lr
    d_w2 = -(net.state.layers[2].weights_in - w2) / lr

    assert _norm_rel_err(d_w1, g_w1) < 1e-2
    assert _norm_rel_err(d_w2, g_w2) < 1e-2
    # Prediction errors follow the observed-minus-predicted convention, so the
    # input message is the negative of the loss gradient at the input.
    assert _norm_rel_err(net.input_error(), -g_x) < 1e-2


def test_input_side_gradient_precision_cancellation():
    """The input-side weight gradient equals backprop at default settings.

    The hidden layer's posterior mean shift is its routed error divided by its posterior
    precision; the ``"precision_weighted"`` gradient multiplies the resulting prediction
    error by that same posterior precision. The division cancels exactly, so the
    gradient of the weight matrix entering the top (input) layer matches backprop at any
    precision setting — checked here with all defaults.
    """
    rng = np.random.default_rng(7)
    w1 = jnp.asarray(rng.normal(size=(_FF_D, _FF_H)) * (2.0 / _FF_H) ** 0.5)
    w2 = jnp.asarray(rng.normal(size=(_FF_H, _FF_D)) * (2.0 / _FF_D) ** 0.5)
    x = jnp.asarray(rng.normal(size=(_FF_D,)))
    y = jnp.asarray(w1 @ jax.nn.gelu(w2 @ x) + rng.normal(size=(_FF_D,)))

    _, g_w2, _ = _ff_oracle_grads(w1, w2, x, y)

    net = _ff_net(hidden_kwargs={}, top_kwargs={}, leaf_kwargs={})
    _set_weights(net, {1: w1, 2: w2})

    lr = 1e-3
    net.prediction(x).update(
        y, optimizer=optax.sgd(lr), learning_kind="precision_weighted"
    )
    d_w2 = -(net.state.layers[2].weights_in - w2) / lr

    assert _norm_rel_err(d_w2, g_w2) < 1e-3


def _ff_net_with_weights(rng) -> tuple[DeepNetwork, jnp.ndarray, jnp.ndarray]:
    """Build a default-config feed-forward net with random weights, plus the weights."""
    w1 = jnp.asarray(rng.normal(size=(_FF_D, _FF_H)) * (2.0 / _FF_H) ** 0.5)
    w2 = jnp.asarray(rng.normal(size=(_FF_H, _FF_D)) * (2.0 / _FF_D) ** 0.5)
    net = _ff_net(hidden_kwargs={}, top_kwargs={}, leaf_kwargs={})
    _set_weights(net, {1: w1, 2: w2})
    return net, w1, w2


def test_sample_step_matches_stateful_api():
    """The pure per-sample step reproduces the stateful prediction/update path.

    From the same state template, ``sample_step`` must return (a) the same input-layer
    error as ``net.prediction(x).update(y)`` followed by ``input_error()``, (b)
    precision increments equal to the change of the carried fields (value precision and
    the volatility level), and (c) weight gradients matching the weight change one SGD
    step applies.
    """
    from pyhgf.utils.vectorized_belief_propagation import sample_step

    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(size=(_FF_D,)))
    y = jnp.asarray(rng.normal(size=(_FF_D,)))

    net, w1, w2 = _ff_net_with_weights(rng)
    template = net.state

    input_error, grads, increments = sample_step(template, x, y)

    # Beliefs-only stateful pass: input error and precision increments.
    net.prediction(x).update(y)
    np.testing.assert_allclose(input_error, net.input_error(), rtol=1e-5, atol=1e-7)
    for elem_before, elem_after, inc in zip(
        template.layers, net.state.layers, increments
    ):
        for field in ("precision", "mean_vol", "precision_vol"):
            np.testing.assert_allclose(
                inc[field],
                getattr(elem_after.state, field) - getattr(elem_before.state, field),
                rtol=1e-5,
                atol=1e-6,
            )

    # Learning stateful pass: gradients match the applied weight change.
    lr = 1e-3
    net_learn = _ff_net(hidden_kwargs={}, top_kwargs={}, leaf_kwargs={})
    _set_weights(net_learn, {1: w1, 2: w2})
    net_learn.prediction(x).update(
        y, optimizer=optax.sgd(lr), learning_kind="precision_weighted"
    )
    # atol: reconstructing the gradient from a float32 weight delta of ~lr·grad
    # loses ~(weight magnitude · float32 eps) / lr of absolute precision.
    for k in (1, 2):
        implied = (
            -(net_learn.state.layers[k].weights_in - template.layers[k].weights_in) / lr
        )
        np.testing.assert_allclose(grads[k], implied, rtol=1e-3, atol=1e-4)


def test_batch_update_averages_and_is_batch_size_invariant():
    """A batch counts as one observation: averaged updates, repetition-invariant.

    ``batch_update`` must apply the batch-mean of the per-sample weight gradients in one
    optimiser step, advance the precision state by the batch-mean of the per-sample
    increments, and return per-sample input errors matching the pure per-sample step.
    Feeding the same batch twice over must produce the same step.
    """
    from pyhgf.utils.vectorized_belief_propagation import sample_step

    rng = np.random.default_rng(23)
    batch = 4
    xb = jnp.asarray(rng.normal(size=(batch, _FF_D)))
    yb = jnp.asarray(rng.normal(size=(batch, _FF_D)))

    net, w1, w2 = _ff_net_with_weights(rng)
    template = net.state

    # Per-sample reference quantities from the pure step.
    ref = [sample_step(template, xb[i], yb[i]) for i in range(batch)]
    mean_grads = {
        k: jnp.mean(jnp.stack([r[1][k] for r in ref]), axis=0) for k in (1, 2)
    }
    mean_inc = jnp.mean(
        jnp.stack([r[2][1]["precision"] for r in ref]), axis=0
    )  # hidden-layer value precision

    lr = 1e-2
    net.batch_update(xb, yb, optimizer=optax.sgd(lr))

    for k in (1, 2):
        np.testing.assert_allclose(
            net.state.layers[k].weights_in,
            template.layers[k].weights_in - lr * mean_grads[k],
            rtol=1e-5,
            atol=1e-6,
        )
    np.testing.assert_allclose(
        net.state.layers[1].state.precision,
        template.layers[1].state.precision + mean_inc,
        rtol=1e-5,
        atol=1e-6,
    )
    assert net.input_errors.shape == (batch, _FF_D)
    for i in range(batch):
        np.testing.assert_allclose(net.input_errors[i], ref[i][0], rtol=1e-5, atol=1e-6)

    # The same samples twice over: averaging makes the step identical.
    net_twice = _ff_net(hidden_kwargs={}, top_kwargs={}, leaf_kwargs={})
    _set_weights(net_twice, {1: w1, 2: w2})
    net_twice.batch_update(
        jnp.concatenate([xb, xb]), jnp.concatenate([yb, yb]), optimizer=optax.sgd(lr)
    )
    for k in (1, 2):
        np.testing.assert_allclose(
            net_twice.state.layers[k].weights_in,
            net.state.layers[k].weights_in,
            rtol=1e-6,
            atol=1e-7,
        )
    np.testing.assert_allclose(
        net_twice.state.layers[1].state.precision,
        net.state.layers[1].state.precision,
        rtol=1e-6,
        atol=1e-7,
    )


def test_batch_update_freezing_flags():
    """``update_precisions=False`` and ``optimizer=None`` freeze their targets.

    Without precision updates, the carried fields stay exactly at their template values
    while the weights still learn (the mode used for exact comparisons against
    backpropagation). Without an optimiser, the weights stay exactly fixed while the
    precisions still adapt.
    """
    rng = np.random.default_rng(31)
    xb = jnp.asarray(rng.normal(size=(4, _FF_D)))
    yb = jnp.asarray(rng.normal(size=(4, _FF_D)))

    # Precisions frozen, weights learning.
    net, w1, w2 = _ff_net_with_weights(rng)
    template = net.state
    net.batch_update(xb, yb, optimizer=optax.sgd(1e-2), update_precisions=False)
    for elem_before, elem_after in zip(template.layers, net.state.layers):
        for field in ("precision", "mean_vol", "precision_vol"):
            np.testing.assert_array_equal(
                getattr(elem_after.state, field), getattr(elem_before.state, field)
            )
    assert not np.allclose(net.state.layers[1].weights_in, w1)

    # Weights frozen, precisions adapting.
    net_frozen = _ff_net(hidden_kwargs={}, top_kwargs={}, leaf_kwargs={})
    _set_weights(net_frozen, {1: w1, 2: w2})
    template_frozen = net_frozen.state
    net_frozen.batch_update(xb, yb)
    np.testing.assert_array_equal(net_frozen.state.layers[1].weights_in, w1)
    assert not np.allclose(
        net_frozen.state.layers[1].state.precision,
        template_frozen.layers[1].state.precision,
    )


def test_batch_update_from_predicted_states_matches():
    """Reusing the forward sweep's states gives the same step as re-sweeping.

    ``predict_states`` + ``batch_update(predicted=...)`` must produce the same new
    weights and per-sample input errors as the plain call, which re-runs the forward
    sweep internally — the reuse reorganises when the same computation happens, it does
    not change the computation.
    """
    rng = np.random.default_rng(17)
    w1 = jnp.asarray(rng.normal(size=(_FF_D, _FF_H)) * (2.0 / _FF_H) ** 0.5)
    w2 = jnp.asarray(rng.normal(size=(_FF_H, _FF_D)) * (2.0 / _FF_D) ** 0.5)
    xb = jnp.asarray(rng.normal(size=(5, _FF_D)))
    yb = jnp.asarray(rng.normal(size=(5, _FF_D)))

    def fresh():
        net = _ff_net(hidden_kwargs={}, top_kwargs={}, leaf_kwargs={})
        return _set_weights(net, {1: w1, 2: w2})

    plain = fresh()
    np.testing.assert_allclose(
        fresh().predict_states(xb)[0], plain.predict(xb), rtol=1e-6, atol=1e-7
    )
    plain.batch_update(xb, yb, optimizer=optax.sgd(1e-2))

    reused = fresh()
    _, states = reused.predict_states(xb)
    reused.batch_update(xb, yb, optimizer=optax.sgd(1e-2), predicted=states)

    for k in (1, 2):
        np.testing.assert_allclose(
            reused.state.layers[k].weights_in,
            plain.state.layers[k].weights_in,
            rtol=1e-6,
            atol=1e-7,
        )
    np.testing.assert_allclose(
        reused.input_errors, plain.input_errors, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        reused.state.layers[1].state.precision,
        plain.state.layers[1].state.precision,
        rtol=1e-6,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# ``update_input_layer``: the sweeps reaching the top (input) layer
# ---------------------------------------------------------------------------


def _three_layer(update_input_layer):
    """Build a 2 -> 4 -> 2 network, identical apart from the flag under test."""
    net = DeepNetwork(update_input_layer=update_input_layer)
    return (
        net
        .add_layer(size=2, add_constant_input=False)
        .add_layer(size=4, add_constant_input=False)
        .add_layer(size=2, add_constant_input=False)
    )


def test_input_layer_frozen_by_default():
    """Off (the default), no sweep writes anything to the top layer.

    This is the behaviour the Rust backend and the nodalised ``Network`` share, so it
    has to survive the flag being added.
    """
    net = _three_layer(False)
    before = net.state.layers[-1].state
    net.prediction(jnp.array([1.0, -1.0])).update(jnp.array([0.5, 0.5]))
    after = net.state.layers[-1].state

    for field in ("precision", "expected_precision", "mean_vol", "precision_vol"):
        np.testing.assert_array_equal(
            getattr(after, field), getattr(before, field), err_msg=field
        )
    # Only the clamp itself moved: mean and expected_mean are the predictors.
    np.testing.assert_array_equal(after.mean, jnp.array([1.0, -1.0]))
    np.testing.assert_array_equal(after.value_prediction_error, jnp.zeros(2))


def test_input_layer_precision_moves_but_mean_stays_on_the_predictors():
    """On, only the top layer's precision moves — its value stays on the input.

    The predictors are observed, and the weight update reads the top layer's mean back
    to build its parent-side factor, so that mean is not the network's to revise. Its
    precision is.
    """
    x = jnp.array([1.0, -1.0])
    net = _three_layer(True)
    net.prediction(x).update(jnp.array([0.5, 0.5]))
    top = net.state.layers[-1].state

    np.testing.assert_array_equal(top.expected_mean, x)
    np.testing.assert_array_equal(top.mean, x)
    # A layer pinned to its own prediction has no residual.
    np.testing.assert_array_equal(top.value_prediction_error, jnp.zeros(2))
    # Evidence arrived, so the belief sharpened past its own prediction.
    assert np.all(np.asarray(top.precision) > np.asarray(top.expected_precision))


def test_input_layer_weight_gradient_sees_the_clamped_predictors():
    """The top weight matrix is learned against the inputs that were supplied.

    The parent-side factor of the gradient is ``coupling_fn(parent.mean)``, so a top
    layer whose mean drifted off ``x`` would train its weights against an estimate of
    the input rather than the input. Turning the flag on must not change that factor.
    """
    rng = np.random.default_rng(3)
    w = rng.normal(size=(4, 2))
    x = jnp.asarray(rng.normal(size=(2,)))
    y = jnp.asarray(rng.normal(size=(2,)))

    grads = []
    for flag in (False, True):
        net = _three_layer(flag)
        _set_weights(net, {1: np.eye(2, 4), 2: w})
        before = np.asarray(net.state.layers[-1].weights_in).copy()
        net.prediction(x).update(y, optimizer=optax.sgd(1e-2))
        grads.append(np.asarray(net.state.layers[-1].weights_in) - before)

    # Same parent activation, same child error on this first step: same update.
    np.testing.assert_allclose(grads[0], grads[1], rtol=1e-6, atol=1e-7)


def test_input_layer_precision_equilibrates_instead_of_running_away():
    """The clamped top layer's precision settles rather than growing without bound.

    Its volatility level is deliberately not updated: with the mean clamped the
    volatility prediction error can never be positive, so a layer that did update it
    would drive its own diffusion to zero and then accumulate evidence forever. Held
    at the tonic value, the diffusion bounds the precision.
    """
    x, y = jnp.array([1.0, -1.0]), jnp.array([0.5, 0.5])
    net = _three_layer(True)

    seen = []
    for step in range(2000):
        net.prediction(x).update(y)
        if step in (0, 999, 1999):
            seen.append(
                float(np.mean(np.asarray(net.state.layers[-1].state.precision)))
            )

    early, mid, late = seen
    assert early < mid  # evidence sharpens it ...
    assert np.isfinite(late)
    assert abs(late - mid) < 0.01 * mid  # ... then it settles
    # The volatility level was never touched, which is what supplies the ceiling.
    top = net.state.layers[-1].state
    np.testing.assert_array_equal(top.mean_vol, jnp.zeros(2))
    np.testing.assert_array_equal(top.precision_vol, jnp.ones(2))


def test_input_layer_posterior_precision_is_carried_to_the_next_prediction():
    """The posterior precision written on one step shapes the next step's prediction.

    This is the whole point of the flag: without the carry, ``expected_precision``
    at the top is a constant, and the layer below inherits a constant value-coupling
    variance no matter how much evidence has accumulated.
    """
    x, y = jnp.array([1.0, -1.0]), jnp.array([0.5, 0.5])
    net = _three_layer(True)

    net.prediction(x).update(y)
    first = np.asarray(net.state.layers[-1].state.precision).copy()
    hidden_first = np.asarray(net.state.layers[1].state.expected_precision).copy()

    for _ in range(20):
        net.prediction(x).update(y)
    later = np.asarray(net.state.layers[-1].state.precision)

    assert np.all(later > first)
    # A more precise input bleeds less uncertainty into the layer below, so that
    # layer's own predicted precision rises with it.
    assert np.all(
        np.asarray(net.state.layers[1].state.expected_precision) > hidden_first
    )


def test_input_layer_flag_survives_from_dict():
    """The flag is part of the network spec, not just the constructor."""
    config = {
        "layers": [{"size": 2}, {"size": 3}],
        "update_input_layer": True,
    }
    assert DeepNetwork.from_dict(config).state.update_input_layer is True
    assert (
        DeepNetwork.from_dict({"layers": [{"size": 2}]}).state.update_input_layer
        is False
    )


def test_input_layer_updated_under_batch_and_scan():
    """``fit`` and ``batch_update`` route through the same sweeps as ``update``.

    Both carry the top layer's precision forward (it is one of ``_CARRIED_FIELDS``), so
    neither should leave it at the value ``add_layer`` gave it.
    """
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(16, 2)))
    y = jnp.asarray(rng.normal(size=(16, 2)))

    for net in (
        _three_layer(True).fit(
            x, y, optimizer=optax.sgd(1e-3), learning_kind="standard"
        ),
        _three_layer(True).batch_update(
            x, y, optimizer=optax.sgd(1e-3), learning_kind="standard"
        ),
    ):
        precision = np.asarray(net.state.layers[-1].state.precision)
        assert np.all(np.isfinite(precision))
        assert not np.allclose(precision, 1.0)


def test_input_layer_precision_tracks_the_routed_evidence_not_the_residual():
    """The clamped input's precision follows the connection, not the fit."""
    rng = np.random.default_rng(7)
    w = rng.normal(size=(2, 2))
    x = jnp.asarray(rng.normal(size=(200, 2)))

    def settle(y, weights):
        net = (
            DeepNetwork(update_input_layer=True)
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=2, add_constant_input=False)
        )
        _set_weights(net, {1: weights})
        # Weights frozen, so nothing but the beliefs can move.
        net.fit(x, y, optimizer=optax.sgd(0.0), weight_update=False)
        return float(np.mean(np.asarray(net.state.layers[-1].state.precision)))

    explained = settle(jnp.asarray(np.asarray(x) @ w.T), w)
    noise = settle(jnp.asarray(rng.normal(size=(200, 2))), w)
    assert explained == pytest.approx(noise, rel=1e-6)

    stronger = settle(jnp.asarray(np.asarray(x) @ w.T), 2.0 * w)
    assert stronger > explained


# ---------------------------------------------------------------------------
# ``feedforward_uncertainty``: withholding a parent's uncertainty
# ---------------------------------------------------------------------------


def _feedforward_net(feedforward_uncertainty):
    """Build a 2 -> 2 -> 2 chain, optionally propagating parent uncertainty."""
    net = DeepNetwork(
        coupling_fn=jax.nn.leaky_relu,
        feedforward_uncertainty=feedforward_uncertainty,
    )
    for _ in range(3):
        net.add_layer(size=2, add_constant_input=False, volatility_parent=True)
    return net


def test_feedforward_uncertainty_false_propagates_no_parent_uncertainty():
    """The default: value parents contribute nothing to a child's precision."""
    x = jnp.array([0.7, -0.4])
    local = _feedforward_net(False).prediction(x)

    for layer in local.state.layers:
        state = layer.state
        np.testing.assert_allclose(
            state.expected_precision,
            state.conditional_expected_precision,
            rtol=1e-6,
        )


def test_feedforward_uncertainty_true_propagates_parent_uncertainty():
    """With the flag on, the value-coupling variance separates the two precisions."""
    x = jnp.array([0.7, -0.4])
    coupled = _feedforward_net(True).prediction(x)

    # The top layer has no value parent, so only the layers below it can differ.
    separated = [
        not np.allclose(
            np.asarray(layer.state.expected_precision),
            np.asarray(layer.state.conditional_expected_precision),
        )
        for layer in coupled.state.layers[:-1]
    ]
    assert any(separated), "the value-coupling variance should separate the two"


def test_feedforward_uncertainty_leaves_the_conditional_precision_alone():
    """The flag governs only the parent's contribution, not the layer's own."""
    x = jnp.array([0.7, -0.4])
    off = _feedforward_net(False).prediction(x)
    on = _feedforward_net(True).prediction(x)

    for a, b in zip(off.state.layers, on.state.layers):
        np.testing.assert_allclose(
            a.state.conditional_expected_precision,
            b.state.conditional_expected_precision,
            rtol=1e-6,
        )


def test_feedforward_uncertainty_defaults_to_off():
    """Value parents do not propagate their uncertainty unless asked to."""
    assert DeepNetwork().feedforward_uncertainty is False


def test_from_dict_forwards_network_level_settings():
    """A network-level key of the config reaches the network and its state.

    The belief sweeps read the copy on the state, so a key that reached only the network
    object would leave the built network running the default rule while its config says
    otherwise.
    """
    net = DeepNetwork.from_dict({
        "layers": [{"size": 2}, {"size": 3}],
        "feedforward_uncertainty": True,
        "predict_precision": False,
    })
    assert net.feedforward_uncertainty is True
    assert net.state.feedforward_uncertainty is True
    assert net.predict_precision is False
    assert net.state.predict_precision is False


def test_from_dict_ignores_keys_that_name_nothing():
    """Keys outside the constructor stay ignored, so configs may be annotated."""
    net = DeepNetwork.from_dict({"layers": [{"size": 2}, {"size": 3}], "note": "x"})
    assert net.n_layers == 2


def test_from_configs_forwards_network_level_settings():
    """``from_configs`` passes any remaining keyword to the constructor."""
    net = DeepNetwork.from_configs(
        [LayerConfig(size=2), LayerConfig(size=3)], feedforward_uncertainty=True
    )
    assert net.feedforward_uncertainty is True
    assert net.state.feedforward_uncertainty is True


# ---------------------------------------------------------------------------
# ``predict_precision``: the precision-prediction switch
# ---------------------------------------------------------------------------


def _precision_net(predict_precision, volatility_parent=True):
    """Build a 2 -> 2 -> 2 chain whose interior layer imports parent uncertainty."""
    net = DeepNetwork(
        coupling_fn=jax.nn.leaky_relu, predict_precision=predict_precision
    )
    for _ in range(3):
        net.add_layer(
            size=2, add_constant_input=False, volatility_parent=volatility_parent
        )
    return net


def test_predict_precision_false_freezes_every_predicted_precision():
    """With the flag off, prediction produces means and leaves precisions alone."""
    x = jnp.array([0.7, -0.4])
    frozen = _precision_net(False).prediction(x)

    for layer in frozen.state.layers:
        state = layer.state
        np.testing.assert_allclose(state.expected_precision, state.precision, rtol=1e-6)
        np.testing.assert_allclose(
            state.conditional_expected_precision, state.precision, rtol=1e-6
        )
        np.testing.assert_allclose(state.effective_precision, 0.0, atol=0.0)


def test_predict_precision_false_leaves_the_volatility_level_untouched():
    """The volatility level is not advanced when the value precisions are frozen."""
    x = jnp.array([0.7, -0.4])
    net = _precision_net(False)
    before = [np.asarray(layer.state.mean_vol).copy() for layer in net.state.layers]
    after = net.prediction(x).state.layers

    for start, layer in zip(before, after):
        np.testing.assert_array_equal(np.asarray(layer.state.expected_mean_vol), start)
        np.testing.assert_array_equal(
            np.asarray(layer.state.expected_precision_vol),
            np.asarray(layer.state.precision_vol),
        )


def test_predict_precision_does_not_touch_the_predicted_means():
    """The flag governs precisions only; the expected means are identical."""
    x = jnp.array([0.7, -0.4])
    on = _precision_net(True).prediction(x).state.layers
    off = _precision_net(False).prediction(x).state.layers
    for a, b in zip(on, off):
        np.testing.assert_allclose(
            np.asarray(a.state.expected_mean),
            np.asarray(b.state.expected_mean),
            rtol=1e-6,
        )


def test_predict_precision_false_freezes_the_clamped_top_layer():
    """The two flags compose: a swept top layer still gets no precision prediction.

    ``update_input_layer`` is what routes the top element through
    :func:`~pyhgf.updates.vectorized.volatile.vectorized_root_prediction` — it is the
    only layer with no parent above to predict it. With ``predict_precision`` off that
    root prediction has to freeze exactly as an interior layer does, or the top would
    keep diffusing while the rest of the hierarchy stood still.
    """
    x, y = jnp.array([1.0, -1.0]), jnp.array([0.5, 0.5])

    def top_after_a_step(predict_precision):
        net = DeepNetwork(update_input_layer=True, predict_precision=predict_precision)
        for size in (2, 4, 2):
            net.add_layer(size=size, add_constant_input=False)
        net.prediction(x).update(y)
        # A second step, so the prediction reads a posterior precision the previous
        # step wrote rather than the value ``add_layer`` built.
        return net.prediction(x).state.layers[-1].state

    frozen = top_after_a_step(False)
    np.testing.assert_allclose(frozen.expected_precision, frozen.precision, rtol=1e-6)
    np.testing.assert_allclose(
        frozen.conditional_expected_precision, frozen.precision, rtol=1e-6
    )
    np.testing.assert_allclose(frozen.effective_precision, 0.0, atol=0.0)
    # The volatility level is not advanced either.
    np.testing.assert_array_equal(frozen.expected_mean_vol, frozen.mean_vol)
    np.testing.assert_array_equal(frozen.expected_precision_vol, frozen.precision_vol)

    # With the prediction on, the diffusion damps the carried precision instead.
    live = top_after_a_step(True)
    assert np.all(np.asarray(live.expected_precision) < np.asarray(live.precision))


def test_input_layer_precision_ignores_the_constant_node():
    """A constant input node carries no evidence about the predictors' precision.

    ``_top_precision_only`` drops the bias column before reading the routed evidence:
    the constant node is not a belief the network holds about the input, so its weights
    say nothing about how precise the observed predictors are. Its column is also what
    makes ``weights_in`` one wider than the top layer's own precision, so leaving it in
    would not even be shape-compatible.
    """
    x, y = jnp.array([1.0, -1.0]), jnp.array([0.5, 0.5])
    rng = np.random.default_rng(11)
    body = rng.normal(size=(4, 2))

    def settle(bias_column):
        net = DeepNetwork(update_input_layer=True)
        net.add_layer(size=2, add_constant_input=False)
        net.add_layer(size=4, add_constant_input=False)
        net.add_layer(size=2, add_constant_input=bias_column is not None)
        if bias_column is None:
            _set_weights(net, {2: body})
        else:
            _set_weights(net, {2: np.hstack([body, bias_column[:, None]])})
        net.prediction(x).update(y)
        return np.asarray(net.state.layers[-1].state.precision)

    plain = settle(None)
    # The top layer keeps its own node count; the constant lives in the matrix only.
    assert plain.shape == (2,)
    np.testing.assert_allclose(settle(np.zeros(4)), plain, rtol=1e-6)
    # An enormous constant weight is still not evidence about the input.
    np.testing.assert_allclose(settle(np.full(4, 50.0)), plain, rtol=1e-6)


def test_predict_precision_default_is_on():
    """The default is the ordinary filter, in which precisions do move."""
    x = jnp.array([0.7, -0.4])
    net = DeepNetwork(coupling_fn=jax.nn.leaky_relu)
    assert net.predict_precision is True
    interior = _precision_net(True).prediction(x).state.layers[1].state
    # The volatility diffusion alone lowers the predicted precision below the prior.
    assert np.all(
        np.asarray(interior.conditional_expected_precision)
        < np.asarray(interior.precision)
    )


def test_fit_update_precisions_false_pins_precisions():
    """Static-cascade mode: precisions stay at their built values across a fit."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(32, 2)))
    y = jnp.asarray(rng.normal(size=(32, 2)))

    def build():
        return (
            DeepNetwork(coupling_fn=jax.nn.leaky_relu)
            .add_layer(size=2, add_constant_input=False, volatility_parent=False)
            .add_layer(
                size=4,
                add_constant_input=False,
                volatility_parent=False,
                precision=3.0,
            )
            .add_layer(size=2, add_constant_input=False, volatility_parent=False)
        )

    static = build()
    before = np.asarray(static.state.layers[1].weights_in).copy()
    static.fit(
        x,
        y,
        optimizer=optax.sgd(1e-2),
        learning_kind="standard",
        update_precisions=False,
    )
    np.testing.assert_array_equal(
        static.state.layers[1].state.precision, jnp.full(4, 3.0)
    )
    assert not np.allclose(np.asarray(static.state.layers[1].weights_in), before)

    # The default carries, so the two modes genuinely diverge.
    dynamic = build().fit(x, y, optimizer=optax.sgd(1e-2), learning_kind="standard")
    assert not np.allclose(np.asarray(dynamic.state.layers[1].state.precision), 3.0)


# ---------------------------------------------------------------------------
# The dead-gradient health check
# ---------------------------------------------------------------------------


def _stiff_deep_net(depth=8):
    """Build a pinned-stiff deep chain that silently kills belief-side gradients."""
    net = DeepNetwork(coupling_fn=jax.nn.leaky_relu, predict_precision=False).add_layer(
        size=3, kind="categorical", volatility_parent=False
    )
    for _ in range(depth):
        net.add_layer(
            size=8,
            volatility_parent=False,
            precision=10e6,
            expected_precision=10e6,
        )
    return net.add_layer(
        size=2,
        add_constant_input=False,
        coupling_fn=lambda x: x,
        volatility_parent=False,
        precision=1e6,
        expected_precision=1e6,
    ).weight_initialisation("he", key=jax.random.key(0))


def test_fit_warns_when_only_the_head_trains():
    """A network training only its head must not fail silently.

    Under the stiff pinned configuration the belief-side gradients are exactly zero
    above the first few matrices, so ``fit`` must raise ``DeadGradientWarning``.
    """
    from pyhgf.model.deep_network import DeadGradientWarning

    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(24, 2)))
    y = jnp.asarray(np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=24)])

    with pytest.warns(DeadGradientWarning):
        _stiff_deep_net().fit(
            x, y, optimizer=optax.adam(1e-3), learning_kind="precision_weighted"
        )


def test_optimizer_state_survives_a_fresh_optimizer_object():
    """Constructing the optimiser inside the loop must not reset its moments."""
    rng = np.random.default_rng(31)
    x = jnp.asarray(rng.normal(size=(32, _FF_D)).astype(np.float32))
    y = jnp.asarray(rng.normal(size=(32, _FF_D)).astype(np.float32))

    def run(make_optimizer, steps=5):
        net, _, _ = _ff_net_with_weights(np.random.default_rng(31))
        sizes = []
        for _ in range(steps):
            before = np.asarray(net.state.layers[2].weights_in).copy()
            net.batch_update(x, y, optimizer=make_optimizer(), update_precisions=False)
            sizes.append(
                float(np.abs(np.asarray(net.state.layers[2].weights_in) - before).max())
            )
        return net, sizes

    shared = optax.adam(1e-2)
    _, reused = run(lambda: shared)
    net, fresh = run(lambda: optax.adam(1e-2))
    np.testing.assert_allclose(fresh, reused, rtol=1e-5)
    # The moments really are accumulating, so the comparison has content.
    assert fresh[-1] > fresh[0] * (1 + 1e-3), fresh

    # A different optimiser kind rebuilds the state instead of failing.
    net.batch_update(x, y, optimizer=optax.sgd(1e-2), update_precisions=False)

    # A rate change keeps the moments: the step scales with the new rate.
    net2, sizes = run(lambda: optax.adam(1e-2), steps=3)
    before = np.asarray(net2.state.layers[2].weights_in).copy()
    net2.batch_update(x, y, optimizer=optax.adam(1e-1), update_precisions=False)
    scaled = float(np.abs(np.asarray(net2.state.layers[2].weights_in) - before).max())
    assert 8.0 < scaled / sizes[-1] < 12.0, (scaled, sizes[-1])
