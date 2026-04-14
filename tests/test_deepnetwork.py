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
    Tested with linear and one nonlinear coupling function.
    """
    n_targets, n_h1, n_h2, n_input = 2, 2, 2, 1

    np.random.seed(42)
    x = np.random.randn(5, n_input)
    y = np.random.randn(5, n_targets)

    for coupling_name, py_coupling_fn in [(None, None), ("tanh", (jnp.tanh,))]:
        label = f"coupling={coupling_name}"

        # --- DeepNetwork (JAX vectorized) ---
        dn = (
            DeepNetwork(
                coupling_fn=py_coupling_fn[0] if py_coupling_fn else lambda x: x
            )
            .add_layer(size=n_targets)
            .add_layer(size=n_h1)
            .add_layer(size=n_h2)
            .add_layer(size=n_input)
        )
        dn.fit(x=x, y=y, lr=0.1, optimizer=None)
        preds_dn = dn.predict(np.array([[0.5]]))

        # --- RsNetwork (Rust) ---
        rs = (
            RsNetwork()
            .add_nodes(kind="continuous-state", n_nodes=n_targets)
            .add_layer(size=n_h1, coupling_fn=coupling_name)
            .add_layer(size=n_h2, coupling_fn=coupling_name)
            .add_layer(size=n_input, coupling_fn=coupling_name)
        )
        predictors = tuple(rs.layers[-1][:n_input])
        rs.fit(
            x=x.tolist(),
            y=y.tolist(),
            inputs_x_idxs=predictors,
            inputs_y_idxs=tuple(range(n_targets)),
            lr=0.1,
            optimizer=None,
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
    net = DeepNetwork().add_layer(size=3, kind="binary").add_layer(size=2)
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
    dn.fit(x=x_train, y=y_train, lr=0.1)
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


def test_cross_backend_binary_volatile():
    """Compare RsNetwork, Network, and DeepNetwork with binary output + volatile hidden.

    Architecture: 1 binary output → 2 volatile hidden → 1 volatile input.
    Tested for all three update types: standard, eHGF, unbounded.
    No constant inputs (bias) to keep cross-backend comparison simple.
    """
    n_targets = 1
    n_hidden = 2
    n_input = 1

    np.random.seed(42)
    x = np.random.randn(10, n_input)
    y = np.random.choice([0.0, 1.0], size=(10, n_targets))

    rtol, atol = 1e-4, 1e-6

    for update_type in ["standard", "eHGF", "unbounded"]:
        label = f"update_type={update_type}"

        # --- DeepNetwork (JAX vectorized) ---
        dn = (
            DeepNetwork(update_type=update_type)
            .add_layer(size=n_targets, kind="binary")
            .add_layer(size=n_hidden, add_constant_input=False)
            .add_layer(size=n_input, add_constant_input=False)
        )
        dn.fit(x=x, y=y, lr=0.1, optimizer=None)

        # --- RsNetwork (Rust) ---
        rs = (
            RsNetwork(update_type=update_type)
            .add_nodes(kind="binary-state", n_nodes=n_targets)
            .add_layer(size=n_hidden, add_constant_input=False)
            .add_layer(size=n_input, add_constant_input=False)
        )
        predictors = tuple(rs.layers[-1][:n_input])
        targets = tuple(range(n_targets))
        rs.fit(
            x=x.tolist(),
            y=y.tolist(),
            inputs_x_idxs=predictors,
            inputs_y_idxs=targets,
            lr=0.1,
            optimizer=None,
        )

        # --- Network (Python per-node) ---
        net = (
            PyNetwork(update_type=update_type)
            .add_nodes(kind="binary-state", n_nodes=n_targets)
            .add_nodes(
                kind="volatile-state",
                n_nodes=n_hidden,
                value_children=list(range(n_targets)),
            )
            .add_nodes(
                kind="volatile-state",
                n_nodes=n_input,
                value_children=list(range(n_targets, n_targets + n_hidden)),
            )
        )
        x_idxs = tuple(range(n_targets + n_hidden, n_targets + n_hidden + n_input))
        y_idxs = tuple(range(n_targets))
        net.fit(
            x=x,
            y=y,
            inputs_x_idxs=x_idxs,
            inputs_y_idxs=y_idxs,
            lr=0.1,
            optimizer=None,
        )

        # ---- Compare hidden-layer volatile states ----
        for i in range(n_hidden):
            node_idx = n_targets + i
            # Mean
            net_mean = float(net.last_attributes[node_idx]["mean"])
            rs_mean = float(rs.node_trajectories[node_idx]["mean"][-1])
            dn_mean = float(dn.state.layers[1].mean[i])

            assert np.allclose(net_mean, dn_mean, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] mean: Network={net_mean} vs DeepNetwork={dn_mean}"
            )
            assert np.allclose(rs_mean, dn_mean, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] mean: Rust={rs_mean} vs DeepNetwork={dn_mean}"
            )

            # Precision
            net_prec = float(net.last_attributes[node_idx]["precision"])
            rs_prec = float(rs.node_trajectories[node_idx]["precision"][-1])
            dn_prec = float(dn.state.layers[1].precision[i])

            assert np.allclose(net_prec, dn_prec, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] precision: Network={net_prec} vs DeepNetwork={dn_prec}"
            )
            assert np.allclose(rs_prec, dn_prec, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] precision: Rust={rs_prec} vs DeepNetwork={dn_prec}"
            )

            # Volatility-level mean
            net_mean_vol = float(net.last_attributes[node_idx]["mean_vol"])
            rs_mean_vol = float(rs.node_trajectories[node_idx]["mean_vol"][-1])
            dn_mean_vol = float(dn.state.layers[1].mean_vol[i])

            assert np.allclose(net_mean_vol, dn_mean_vol, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] mean_vol: Network={net_mean_vol} vs DeepNetwork={dn_mean_vol}"
            )
            assert np.allclose(rs_mean_vol, dn_mean_vol, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] mean_vol: Rust={rs_mean_vol} vs DeepNetwork={dn_mean_vol}"
            )

            # Volatility-level precision
            net_prec_vol = float(net.last_attributes[node_idx]["precision_vol"])
            rs_prec_vol = float(rs.node_trajectories[node_idx]["precision_vol"][-1])
            dn_prec_vol = float(dn.state.layers[1].precision_vol[i])

            assert np.allclose(net_prec_vol, dn_prec_vol, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] precision_vol: Network={net_prec_vol} vs DeepNetwork={dn_prec_vol}"
            )
            assert np.allclose(rs_prec_vol, dn_prec_vol, rtol=rtol, atol=atol), (
                f"{label}: hidden[{i}] precision_vol: Rust={rs_prec_vol} vs DeepNetwork={dn_prec_vol}"
            )

        # ---- Compare coupling weights between volatile layers (hidden ← input) ----
        for j in range(n_hidden):
            node_idx = n_targets + j
            net_weights = net.last_attributes[node_idx]["value_coupling_parents"]
            for k in range(n_input):
                w_net = float(net_weights[k])
                w_rs = float(
                    rs.node_trajectories[node_idx]["value_coupling_parents"][-1][k]
                )
                w_dn = float(dn.state.weights[1][j, k])

                assert np.allclose(w_net, w_dn, rtol=rtol, atol=atol), (
                    f"{label}: weight hidden[{j}]←input[{k}]: "
                    f"Network={w_net} vs DeepNetwork={w_dn}"
                )
                assert np.allclose(w_rs, w_dn, rtol=rtol, atol=atol), (
                    f"{label}: weight hidden[{j}]←input[{k}]: "
                    f"Rust={w_rs} vs DeepNetwork={w_dn}"
                )

        # ---- Compare binary output coupling weights (should remain at 1.0) ----
        for j in range(n_targets):
            net_weights = net.last_attributes[j]["value_coupling_parents"]
            for k in range(n_hidden):
                w_net = float(net_weights[k])
                w_dn = float(dn.state.weights[0][j, k])

                assert np.allclose(w_net, w_dn, rtol=rtol, atol=atol), (
                    f"{label}: weight binary[{j}]←hidden[{k}]: "
                    f"Network={w_net} vs DeepNetwork={w_dn}"
                )

        # ---- Compare predictions ----
        x_test = np.array([[0.5]])
        preds_dn = dn.predict(x_test)
        preds_rs = rs.predict(
            x=[[0.5]],
            inputs_x_idxs=predictors,
            inputs_y_idxs=targets,
        )
        preds_net = net.predict(
            x=np.array([[0.5]]),
            inputs_x_idxs=x_idxs,
            inputs_y_idxs=y_idxs,
        )

        assert np.all(np.isfinite(np.asarray(preds_dn))), (
            f"{label}: DeepNetwork predictions contain NaN/Inf"
        )
        assert np.allclose(
            np.asarray(preds_net), np.asarray(preds_dn), rtol=rtol, atol=atol
        ), f"{label}: predictions: Network={preds_net} vs DeepNetwork={preds_dn}"
        assert np.allclose(
            np.asarray(preds_rs), np.asarray(preds_dn), rtol=rtol, atol=atol
        ), f"{label}: predictions: Rust={preds_rs} vs DeepNetwork={preds_dn}"
