# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests for the Rust ``DeepNetwork`` class.

Two complementary suites cover ``pyhgf.rshgf.DeepNetwork``:

* API surface: builder validation, input handling, error paths, and weight
  management of the Rust class itself.
* Numerical parity: ``pyhgf.model.DeepNetwork`` (JAX) and the Rust class
  implement the same vectorised deep predictive coding network; these tests
  build matched architectures, inject identical weights, and assert that the
  forward pass and the weight learning trajectories agree.

JAX is put in float64 mode for the duration of this module so it matches the
Rust ``f64`` backend; without this, JAX's default float32 would swamp any
meaningful tolerance. The flag is restored on teardown so the other test
modules keep their default dtype behaviour. The two backends order their
matrix product summations differently, so agreement is asserted to a tight
floating point tolerance rather than bit equality.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork

from pyhgf.model import DeepNetwork as JaxDeepNetwork


@pytest.fixture(autouse=True, scope="module")
def _float64_jax():
    """Enable float64 for this module's tests and restore the flag after."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _build_rs(sizes, volatility_updates="eHGF", coupling_fn="linear"):
    net = RsDeepNetwork(volatility_updates=volatility_updates, coupling_fn=coupling_fn)
    for s in sizes:
        net = net.add_layer(s)
    return net


def _build_jax(sizes, volatility_updates="eHGF", coupling_fn=None):
    kwargs = {"volatility_updates": volatility_updates}
    if coupling_fn is not None:
        kwargs["coupling_fn"] = coupling_fn
    net = JaxDeepNetwork(**kwargs)
    for s in sizes:
        net = net.add_layer(size=s)
    return net


def _random_weights(rs_net, rng, scale=0.3):
    """Random weights matching each of the Rust net's weight matrices."""
    return [rng.normal(scale=scale, size=w.shape) for w in rs_net.get_weights()]


def _inject_jax_weights(net, weights):
    """Write the given weight matrices into the JAX network's layers 1..n."""
    layers = list(net.state.layers)
    new_layers = [layers[0]] + [
        dataclasses.replace(layer, weights_in=jnp.asarray(w))
        for layer, w in zip(layers[1:], weights)
    ]
    net.state = dataclasses.replace(net.state, layers=tuple(new_layers))


def _matched_pair(sizes, rng, volatility_updates="eHGF", coupling=None):
    """Build a Rust and a JAX network with identical topology and weights.

    ``coupling`` is the name of a ``jnp`` function (for example ``"tanh"``); ``None``
    uses the identity coupling on both sides.
    """
    rs = _build_rs(
        sizes, volatility_updates, "linear" if coupling is None else coupling
    )
    jx = _build_jax(
        sizes, volatility_updates, None if coupling is None else getattr(jnp, coupling)
    )
    weights = _random_weights(rs, rng)
    rs.set_weights(weights)
    _inject_jax_weights(jx, weights)
    return rs, jx


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_constructor_validation():
    """Reject invalid constructor arguments with a ValueError."""
    with pytest.raises(ValueError, match="Invalid volatility update"):
        RsDeepNetwork(volatility_updates="nope")
    with pytest.raises(ValueError, match="Unknown coupling function"):
        RsDeepNetwork(coupling_fn="sigmiod")
    with pytest.raises(ValueError, match="precision_clipping_value"):
        RsDeepNetwork(precision_clipping_value=0.7)


def test_methods_require_a_layer():
    """Error on every method that needs a built network when none exists."""
    net = RsDeepNetwork()
    for call in (
        lambda: net.predict(np.zeros((1, 1))),
        lambda: net.fit(np.zeros((1, 1)), np.zeros((1, 1))),
        lambda: net.get_weights(),
        lambda: net.weight_initialisation("he"),
    ):
        with pytest.raises(ValueError, match="add at least one layer"):
            call()


def test_add_layer_chaining_and_getters():
    """Chain add_layer calls and read the layer getters."""
    net = RsDeepNetwork()
    net.add_layer(2).add_layer(3, kind="volatile-state").add_layer(4)
    assert net.n_layers == 3
    assert net.layer_sizes == [2, 3, 4]
    assert net.n_nodes == 9


def test_add_layer_validation():
    """Reject unknown kinds and overrides; accept valid overrides."""
    with pytest.raises(ValueError, match="Invalid layer kind"):
        RsDeepNetwork().add_layer(2, kind="nope")
    with pytest.raises(ValueError, match="Unknown layer override"):
        RsDeepNetwork().add_layer(2, not_a_field=1.0)
    # LayerParams and LayerState overrides are accepted.
    RsDeepNetwork().add_layer(2, tonic_volatility=-2.0, expected_precision=1e10)


def test_categorical_only_at_bottom():
    """Accept a categorical bottom layer and reject one above it."""
    RsDeepNetwork().add_layer(3, kind="categorical")
    net = RsDeepNetwork().add_layer(2)
    with pytest.raises(ValueError, match="only supported as the output"):
        net.add_layer(3, kind="categorical")


def test_failed_add_layer_is_not_retained():
    """Drop the offending config when add_layer fails."""
    net = RsDeepNetwork().add_layer(3)
    with pytest.raises(ValueError, match="cannot use"):
        net.add_layer(3, fully_connected=False, add_constant_input=True)
    # The offending config is dropped; the network stays usable.
    assert net.n_layers == 1
    net.add_layer(3, fully_connected=False, add_constant_input=False)
    assert net.n_layers == 2


def test_predict_input_forms():
    """Accept 2D arrays, 1D samples, and nested lists in predict."""
    net = _build_rs([2, 3])
    rng = np.random.default_rng(0)
    x2 = rng.normal(size=(4, 3))
    out2 = net.predict(x2)
    assert out2.shape == (4, 2)
    # A 1D sample returns a 1D output and matches the matching batch row.
    out1 = net.predict(x2[0])
    assert out1.shape == (2,)
    np.testing.assert_allclose(out1, out2[0], rtol=1e-12)
    # Nested Python lists are accepted.
    out_list = net.predict(x2.tolist())
    np.testing.assert_allclose(out_list, out2, rtol=1e-12)


def test_predict_input_errors():
    """Reject mismatched feature counts and ragged inputs."""
    net = _build_rs([2, 3])
    with pytest.raises(ValueError, match="feature column"):
        net.predict(np.zeros((4, 5)))
    with pytest.raises(ValueError, match="ragged"):
        net.predict([[1.0, 2.0, 3.0], [1.0, 2.0]])


def test_fit_1d_disambiguation():
    """Treat a 1D array as n samples when the layer has one node."""
    # With single node layers a 1D array is n samples of one feature.
    net = _build_rs([1, 1])
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    out = net.fit(x, y)
    assert out.shape == (5, 1)


def test_fit_validation_errors():
    """Reject mismatched sample counts and unknown fit options."""
    net = _build_rs([2, 3])
    x = np.zeros((4, 3))
    y = np.zeros((4, 2))
    with pytest.raises(ValueError, match="same number of samples"):
        net.fit(x, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="Unknown optimizer"):
        net.fit(x, y, optimizer="nope")
    with pytest.raises(ValueError, match="Unknown learning_kind"):
        net.fit(x, y, learning_kind="nope")


def test_set_weights_roundtrip():
    """Return the same matrices from get_weights after set_weights."""
    net = _build_rs([2, 3])
    rng = np.random.default_rng(1)
    weights = _random_weights(net, rng)
    net.set_weights(weights)
    for got, expected in zip(net.get_weights(), weights):
        np.testing.assert_allclose(got, expected)


def test_set_weights_validation_is_atomic():
    """Leave every weight untouched when set_weights rejects its input."""
    net = _build_rs([2, 3, 2])
    good = _random_weights(net, np.random.default_rng(2))
    with pytest.raises(ValueError, match="expected 2 weight"):
        net.set_weights(good[:1])
    # A shape error anywhere leaves every weight untouched.
    with pytest.raises(ValueError, match="shape mismatch"):
        net.set_weights([good[0], np.zeros((9, 9))])
    for w in net.get_weights():
        assert (np.asarray(w) == 1.0).all()


def test_weight_initialisation():
    """Draw deterministic weights per seed; reset them on rebuild."""
    a = _build_rs([2, 3, 3]).weight_initialisation("he", seed=42)
    b = _build_rs([2, 3, 3]).weight_initialisation("he", seed=42)
    for wa, wb in zip(a.get_weights(), b.get_weights()):
        np.testing.assert_array_equal(wa, wb)
        assert (np.asarray(wa) != 1.0).any()
    with pytest.raises(ValueError):
        a.weight_initialisation("nope")
    # A later add_layer rebuilds the network and resets the weights.
    a.add_layer(2)
    assert (np.asarray(a.get_weights()[0]) == 1.0).all()


def test_fit_weight_update_false_freezes_weights():
    """Keep the weights unchanged when weight_update is False."""
    net = _build_rs([2, 3])
    rng = np.random.default_rng(3)
    weights = _random_weights(net, rng)
    net.set_weights(weights)
    net.fit(rng.normal(size=(6, 3)), rng.normal(size=(6, 2)), weight_update=False)
    for got, expected in zip(net.get_weights(), weights):
        np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Parity with the JAX backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("volatility_updates", ["eHGF", "standard", "unbounded"])
def test_predict_parity(volatility_updates):
    """Match the JAX forward pass for every volatility scheme."""
    rng = np.random.default_rng(10)
    rs, jx = _matched_pair([2, 4, 3], rng, volatility_updates=volatility_updates)
    x = rng.normal(size=(5, 3))
    np.testing.assert_allclose(
        rs.predict(x), np.asarray(jx.predict(x)), rtol=1e-8, atol=1e-10
    )


def test_predict_parity_tanh_coupling():
    """Match the JAX forward pass under a tanh coupling function."""
    rng = np.random.default_rng(11)
    rs, jx = _matched_pair([2, 4, 3], rng, coupling="tanh")
    x = rng.normal(size=(5, 3))
    np.testing.assert_allclose(
        rs.predict(x), np.asarray(jx.predict(x)), rtol=1e-8, atol=1e-10
    )


def test_predict_parity_single_sample():
    """Match the JAX forward pass for a single 1D sample."""
    rng = np.random.default_rng(12)
    rs, jx = _matched_pair([2, 3], rng)
    x = rng.normal(size=3)
    out_rs = rs.predict(x)
    out_jx = np.asarray(jx.predict(x))
    assert out_rs.shape == out_jx.shape == (2,)
    np.testing.assert_allclose(out_rs, out_jx, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    ("optimizer", "make_optax"),
    [("sgd", lambda: optax.sgd(0.05)), ("adam", lambda: optax.adam(1e-2))],
)
def test_fit_trajectory_parity(optimizer, make_optax):
    """Per sample predictions and final weights agree over a training run."""
    rng = np.random.default_rng(13)
    rs, jx = _matched_pair([2, 4, 3], rng)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 2))

    preds_rs = rs.fit(
        x, y, optimizer=optimizer, learning_rate=0.05 if optimizer == "sgd" else 1e-2
    )
    jx.fit(x, y, make_optax())

    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )
    jx_weights = [np.asarray(layer.weights_in) for layer in jx.state.layers[1:]]
    for w_rs, w_jx in zip(rs.get_weights(), jx_weights):
        np.testing.assert_allclose(w_rs, w_jx, rtol=1e-6, atol=1e-9)


def test_fit_parity_standard_learning_kind():
    """Match the JAX training run under the standard gradient mode."""
    rng = np.random.default_rng(14)
    rs, jx = _matched_pair([2, 3, 3], rng)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=(6, 2))
    preds_rs = rs.fit(
        x, y, optimizer="sgd", learning_rate=0.1, learning_kind="standard"
    )
    jx.fit(x, y, optax.sgd(0.1), learning_kind="standard")
    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )


def test_fit_parity_time_step():
    """Match the JAX training run under a non-unit time step."""
    rng = np.random.default_rng(15)
    rs, jx = _matched_pair([2, 3], rng)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=(6, 2))
    preds_rs = rs.fit(x, y, optimizer="sgd", learning_rate=0.05, time_step=0.7)
    jx.fit(x, y, optax.sgd(0.05), time_step=0.7)
    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )


def test_fit_parity_weight_update_false():
    """With frozen weights the belief updates alone must also agree."""
    rng = np.random.default_rng(16)
    rs, jx = _matched_pair([2, 3], rng)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=(6, 2))
    preds_rs = rs.fit(x, y, weight_update=False)
    jx.fit(x, y, optax.sgd(0.1), weight_update=False)
    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("volatility_updates", ["standard", "unbounded"])
def test_fit_parity_volatility_updates(volatility_updates):
    """The belief updates of every volatility scheme agree during training."""
    rng = np.random.default_rng(19)
    rs, jx = _matched_pair([2, 3], rng, volatility_updates=volatility_updates)
    x = rng.normal(size=(6, 3))
    y = rng.normal(size=(6, 2))
    preds_rs = rs.fit(x, y, optimizer="sgd", learning_rate=0.05)
    jx.fit(x, y, optax.sgd(0.05))
    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )


def test_fit_parity_binary_output():
    """Match the JAX training run with a binary output layer."""
    rng = np.random.default_rng(20)
    rs = RsDeepNetwork(volatility_updates="eHGF")
    rs.add_layer(2, kind="binary").add_layer(3)
    jx = JaxDeepNetwork(volatility_updates="eHGF")
    jx.add_layer(size=2, kind="binary").add_layer(size=3)
    weights = _random_weights(rs, rng)
    rs.set_weights(weights)
    _inject_jax_weights(jx, weights)
    x = rng.normal(size=(8, 3))
    y = rng.integers(0, 2, size=(8, 2)).astype(float)
    preds_rs = rs.fit(x, y, optimizer="sgd", learning_rate=0.05)
    jx.fit(x, y, optax.sgd(0.05))
    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )
    jx_weights = [np.asarray(layer.weights_in) for layer in jx.state.layers[1:]]
    for w_rs, w_jx in zip(rs.get_weights(), jx_weights):
        np.testing.assert_allclose(w_rs, w_jx, rtol=1e-6, atol=1e-9)


def test_fit_parity_categorical_output():
    """Match the JAX training run with a categorical output layer."""
    rng = np.random.default_rng(21)
    rs = RsDeepNetwork(volatility_updates="eHGF")
    rs.add_layer(3, kind="categorical").add_layer(4)
    jx = JaxDeepNetwork(volatility_updates="eHGF")
    jx.add_layer(size=3, kind="categorical").add_layer(size=4)
    weights = _random_weights(rs, rng)
    rs.set_weights(weights)
    _inject_jax_weights(jx, weights)
    x = rng.normal(size=(8, 4))
    y = np.eye(3)[rng.integers(0, 3, size=8)]
    preds_rs = rs.fit(x, y, optimizer="sgd", learning_rate=0.05)
    jx.fit(x, y, optax.sgd(0.05))
    np.testing.assert_allclose(
        preds_rs, np.asarray(jx.predictions), rtol=1e-6, atol=1e-9
    )


def test_predict_parity_binary_output():
    """Match the JAX forward pass with a binary output layer."""
    rng = np.random.default_rng(17)
    rs = RsDeepNetwork(volatility_updates="eHGF")
    rs.add_layer(2, kind="binary").add_layer(3)
    jx = JaxDeepNetwork(volatility_updates="eHGF")
    jx.add_layer(size=2, kind="binary").add_layer(size=3)
    weights = _random_weights(rs, rng)
    rs.set_weights(weights)
    _inject_jax_weights(jx, weights)
    x = rng.normal(size=(5, 3))
    np.testing.assert_allclose(
        rs.predict(x), np.asarray(jx.predict(x)), rtol=1e-8, atol=1e-10
    )


def test_predict_parity_categorical_output():
    """Match the JAX forward pass with a categorical output layer."""
    rng = np.random.default_rng(18)
    rs = RsDeepNetwork(volatility_updates="eHGF")
    rs.add_layer(3, kind="categorical").add_layer(4)
    jx = JaxDeepNetwork(volatility_updates="eHGF")
    jx.add_layer(size=3, kind="categorical").add_layer(size=4)
    weights = _random_weights(rs, rng)
    rs.set_weights(weights)
    _inject_jax_weights(jx, weights)
    x = rng.normal(size=(5, 4))
    out_rs = rs.predict(x)
    # A categorical output layer predicts a softmax; rows sum to one.
    np.testing.assert_allclose(out_rs.sum(axis=1), np.ones(5), rtol=1e-12)
    np.testing.assert_allclose(out_rs, np.asarray(jx.predict(x)), rtol=1e-8, atol=1e-10)
