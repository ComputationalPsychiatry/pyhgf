# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests of the exact coupling moments in the relaxed updates.

With ``coupling_moments="exact"``, the relaxed updates replace the coupling function of
a value parent by its statistical linearisation under the predicted belief of the
parent. The tests compare the four backends (nodalised and vectorised, JAX and Rust)
with hand-computed updates and with each other.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.coupling import resolve_coupling_fn
from pyhgf.model import DeepNetwork as JaxDeepNetwork
from pyhgf.model import Network as PyNetwork
from pyhgf.updates.prediction.continuous import (
    predict_precision,
    predict_value_coupling_variance,
)
from pyhgf.updates.vectorised.learning import (
    evidence_pullback,
    learning_weights_vectorised,
)
from pyhgf.updates.vectorised.volatile.prediction import vectorised_layer_prediction
from pyhgf.utils.vectorised_belief_propagation import (
    carry_precisions,
    prediction_sweep,
    update_sweep,
)

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64

JAX_COUPLINGS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "sigmoid": jax.nn.sigmoid,
}


def _coupling_argument(cls, name):
    """Coupling function argument of ``add_nodes`` for the JAX or the Rust class."""
    return (JAX_COUPLINGS[name],) if cls is PyNetwork else name


def _linearised_update(coupling, mean, variance, observation, noise_variance):
    """Posterior of a parent observed through a statistically linearised coupling.

    The parent has the prior N(mean, variance) and the observation is
    u = g(x) + noise. With the moments (m, v, A) of g under the prior, the linear
    function is m + A (x - mean) + e, with Var[e] = v - A^2 variance. Returns the
    posterior mean and precision of the Kalman update of this linear model, with the
    predicted mean and precision of the observation.
    """
    expected_value, coupling_variance, slope = (
        float(moment) for moment in coupling.moments(mean, variance)
    )
    residual_variance = coupling_variance - slope**2 * variance
    innovation_variance = coupling_variance + noise_variance
    gain = slope * variance / innovation_variance
    return {
        "mean": mean + gain * (observation - expected_value),
        "precision": 1.0 / (variance - gain * slope * variance),
        "observation_expected_mean": expected_value,
        "observation_expected_precision": 1.0 / (noise_variance + residual_variance),
    }


# ----------------------------------------------------------------------------------
# Nodalised backends
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [PyNetwork, RsNetwork], ids=["jax", "rust"])
@pytest.mark.parametrize("name", ["leaky_relu", "relu", "tanh"])
@pytest.mark.parametrize("prior_mean", [-1.0, 0.3])
def test_nodalised_update_is_the_linearised_kalman_update(cls, name, prior_mean):
    """One observation updates a parent as the statistically linearised model does.

    For a leaky rectified linear unit with a prior mean of -1 and a unit variance, the
    first-order Taylor expansion barely moves the parent (the slope at the mean is
    0.01), whereas the mean slope under the prior moves it.
    """
    observation, noise_precision = 2.0, 4.0

    def run(coupling_moments):
        net = (
            cls(volatility_updates="standard", coupling_moments=coupling_moments)
            .add_nodes(precision=noise_precision)
            .add_nodes(
                value_children=0,
                coupling_fn=_coupling_argument(cls, name),
                mean=prior_mean,
                precision=1.0,
                # the predicted precision is the prior precision
                tonic_volatility=-30.0,
            )
            .input_data(input_data=np.array([observation]))
        )
        return {
            "mean": float(net.node_trajectories[1]["mean"][0]),
            "precision": float(net.node_trajectories[1]["precision"][0]),
            "observation_expected_mean": float(
                net.node_trajectories[0]["expected_mean"][0]
            ),
            "observation_expected_precision": float(
                net.node_trajectories[0]["expected_precision"][0]
            ),
        }

    expected = _linearised_update(
        resolve_coupling_fn(name), prior_mean, 1.0, observation, 1.0 / noise_precision
    )
    exact = run("exact")
    for key, value in expected.items():
        np.testing.assert_allclose(exact[key], value, rtol=1e-4, err_msg=key)

    # the default is the first-order Taylor expansion at the mean of the parent
    taylor = run("taylor")
    np.testing.assert_allclose(
        taylor["observation_expected_mean"],
        float(resolve_coupling_fn(name)(prior_mean)),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        taylor["observation_expected_precision"], noise_precision, rtol=1e-6
    )
    if name == "leaky_relu" and prior_mean < 0:
        assert abs(taylor["mean"] - prior_mean) < 0.1
        assert exact["mean"] - prior_mean > 0.9


def test_predicted_variance_carries_the_variance_of_the_coupling():
    r"""The value parents add :math:`(t \alpha)^2 \mathrm{Var}[g(x_b)]` to a prediction.

    The slope and the residual variances of
    :func:`pyhgf.updates.prediction.continuous.predict_value_coupling_variance` sum
    to this variance, the residual variance is counted in the conditional predicted
    precision, and it is 0 under the first-order Taylor expansion.
    """
    time_step, alpha = 0.7, 1.3
    net = (
        PyNetwork(coupling_moments="exact")
        .add_nodes(precision=1e4)
        .add_nodes(value_children=0, tonic_volatility=-2.0)
        .add_nodes(
            value_children=([1], [alpha]),
            coupling_fn=(jax.nn.relu,),
            tonic_volatility=-1.0,
        )
    )
    attributes = net.attributes
    attributes[-1]["time_step"] = time_step
    attributes[2]["expected_mean"] = 0.4
    attributes[2]["expected_precision"] = 0.8

    _, coupling_variance, slope = resolve_coupling_fn("relu").moments(0.4, 1.0 / 0.8)
    slope_variance, residual_variance = predict_value_coupling_variance(
        attributes, net.edges, 1, "exact"
    )
    np.testing.assert_allclose(
        slope_variance, (time_step * alpha * slope) ** 2 / 0.8, rtol=1e-5
    )
    np.testing.assert_allclose(
        slope_variance + residual_variance,
        (time_step * alpha) ** 2 * coupling_variance,
        rtol=1e-5,
    )
    assert float(residual_variance) > 0.0

    expected_precision, conditional_expected_precision, _ = predict_precision(
        attributes, net.edges, 1, "exact"
    )
    own_variance = 1.0 / attributes[1]["precision"] + time_step * np.exp(-2.0)
    np.testing.assert_allclose(
        1.0 / conditional_expected_precision,
        own_variance + residual_variance,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        1.0 / expected_precision,
        own_variance + (time_step * alpha) ** 2 * coupling_variance,
        rtol=1e-5,
    )

    # first-order Taylor expansion: the slope at the mean, and no residual
    slope_variance, residual_variance = predict_value_coupling_variance(
        attributes, net.edges, 1
    )
    np.testing.assert_allclose(slope_variance, (time_step * alpha) ** 2 / 0.8)
    assert float(residual_variance) == 0.0


def _nodalised_network(cls, name, volatility_updates, topology, coupling_moments):
    """Build and run a small network whose value coupling is non-linear."""
    timeseries = np.asarray(load_data("continuous"))[:120]
    coupling_fn = _coupling_argument(cls, name)
    net = cls(volatility_updates=volatility_updates, coupling_moments=coupling_moments)
    if topology == "leaf":
        # input <- parent
        net = net.add_nodes(precision=4.0).add_nodes(
            value_children=0, coupling_fn=coupling_fn, tonic_volatility=-2.0
        )
    elif topology == "chain":
        # input <- state <- parent, with a volatility parent on the state
        net = (
            net
            .add_nodes(precision=4.0)
            .add_nodes(value_children=0, tonic_volatility=-3.0)
            .add_nodes(value_children=1, coupling_fn=coupling_fn, tonic_volatility=-2.0)
            .add_nodes(volatility_children=1, tonic_volatility=-3.0)
        )
    elif topology == "noise":
        # input with a noise parent <- parent
        net = (
            net
            .add_nodes()
            .add_nodes(value_children=0, coupling_fn=coupling_fn, tonic_volatility=-2.0)
            .add_nodes(volatility_children=0, mean=1.0, tonic_volatility=-4.0)
        )
    else:
        # a parent with a value child and a volatility child
        net = (
            net
            .add_nodes(precision=4.0)
            .add_nodes(value_children=0, tonic_volatility=-3.0)
            .add_nodes(precision=4.0)
            .add_nodes(value_children=2, tonic_volatility=-3.0)
            .add_nodes(
                value_children=1,
                volatility_children=3,
                coupling_fn=coupling_fn,
                tonic_volatility=-2.0,
            )
        )
        timeseries = np.stack([timeseries, np.roll(timeseries, 3)], axis=1)
    return net.input_data(input_data=timeseries)


@pytest.mark.parametrize("topology", ["leaf", "chain", "noise", "mixed"])
@pytest.mark.parametrize("name", sorted(JAX_COUPLINGS))
@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF", "unbounded"])
def test_nodalised_backends_match(topology, name, volatility_updates):
    """The JAX and the Rust nodalised backends agree with the exact moments."""
    with enable_x64(True):
        py_net = _nodalised_network(
            PyNetwork, name, volatility_updates, topology, "exact"
        )
        rs_net = _nodalised_network(
            RsNetwork, name, volatility_updates, topology, "exact"
        )
        for node_idx in range(len(py_net.edges)):
            for key in ("mean", "precision", "expected_mean", "expected_precision"):
                expected = np.asarray(py_net.node_trajectories[node_idx][key])
                assert np.all(np.isfinite(expected))
                np.testing.assert_allclose(
                    np.asarray(rs_net.node_trajectories[node_idx][key], dtype=float),
                    expected,
                    rtol=1e-7,
                    atol=1e-9,
                    err_msg=f"node {node_idx}, {key}",
                )


@pytest.mark.parametrize("cls", [PyNetwork, RsNetwork], ids=["jax", "rust"])
def test_nodalised_moments_change_a_curved_coupling_only(cls):
    """The two settings differ for a curved coupling and agree for a linear one."""
    timeseries = np.asarray(load_data("continuous"))[:60]

    def posterior_means(coupling_moments, **coupling_kwargs):
        net = (
            cls(coupling_moments=coupling_moments)
            .add_nodes(precision=4.0)
            .add_nodes(value_children=0, tonic_volatility=-2.0, **coupling_kwargs)
            .input_data(input_data=timeseries)
        )
        return np.asarray(net.node_trajectories[1]["mean"], dtype=float)

    np.testing.assert_allclose(
        posterior_means("exact"), posterior_means("taylor"), rtol=1e-6
    )
    curved = {"coupling_fn": _coupling_argument(cls, "relu")}
    assert (
        np.max(
            np.abs(
                posterior_means("exact", **curved) - posterior_means("taylor", **curved)
            )
        )
        > 1e-2
    )


def test_noise_prediction_error_takes_the_posterior_moments():
    r"""The noise prediction error is the expected squared noise under the posterior.

    With the exact moments, the noise prediction error of an input node :math:`u` is
    :math:`\hat{\pi}_u (\mathrm{Var}[g(x_b)] + (u - \mathbb{E}[g(x_b)])^2) - 1` under
    the posterior of its value parent, with :math:`\hat{\pi}_u` the precision of the
    observation noise alone.
    """
    with enable_x64(True):
        net = (
            PyNetwork(coupling_moments="exact")
            .add_nodes()
            .add_nodes(
                value_children=0, coupling_fn=(jax.nn.relu,), tonic_volatility=-2.0
            )
            .add_nodes(volatility_children=0, mean=1.0, tonic_volatility=-4.0)
        )
        observation = 0.8
        net.input_data(input_data=np.array([observation]))
        attributes = net.last_attributes

        posterior_value, posterior_variance, _ = resolve_coupling_fn("relu").moments(
            attributes[1]["mean"], 1.0 / attributes[1]["precision"]
        )
        noise_precision = np.exp(-attributes[2]["expected_mean"])
        np.testing.assert_allclose(
            attributes[0]["temp"]["volatility_prediction_error"],
            noise_precision
            * (posterior_variance + (observation - posterior_value) ** 2)
            - 1.0,
            rtol=1e-8,
        )
        # the predicted precision of the input also counts the residual variance
        assert float(attributes[0]["expected_precision"]) < float(noise_precision)


# ----------------------------------------------------------------------------------
# Vectorised backends
# ----------------------------------------------------------------------------------


def test_vectorised_update_is_the_linearised_kalman_update():
    """Two layers of leaky rectified linear units follow the linearised model.

    The top layer holds the predictor with a unit precision, the hidden layer is
    predicted through the moments of the coupling under this belief, and the hidden
    layer is updated from the observation as in the nodalised backend.
    """
    with enable_x64(True):
        coupling = resolve_coupling_fn("leaky_relu")
        net = JaxDeepNetwork(
            coupling_fn="leaky_relu",
            feedforward_uncertainty=True,
            coupling_moments="exact",
            volatility_updates="standard",
        )
        for precision in (4.0, 1.0, 1.0):
            net = net.add_layer(
                1,
                add_constant_input=False,
                volatility_parent=False,
                precision=precision,
            )
        for predictor in (-1.0, 0.4):
            state = update_sweep(
                prediction_sweep(net.state, jnp.array([predictor])), jnp.array([2.0])
            )
            hidden, output = state.layers[1].state, state.layers[0].state

            # prediction of the hidden layer from the predictor, N(predictor, 1)
            top_value, top_variance, top_slope = (
                float(moment) for moment in coupling.moments(predictor, 1.0)
            )
            top_residual = top_variance - top_slope**2
            np.testing.assert_allclose(hidden.expected_mean, top_value, rtol=1e-10)
            np.testing.assert_allclose(
                hidden.conditional_expected_precision, 1.0 / (1.0 + top_residual)
            )
            np.testing.assert_allclose(
                hidden.expected_precision, 1.0 / (1.0 + top_variance), rtol=1e-10
            )

            # update of the hidden layer from the observation
            expected = _linearised_update(
                coupling, top_value, 1.0 + top_variance, 2.0, 0.25
            )
            np.testing.assert_allclose(hidden.mean, expected["mean"], rtol=1e-9)
            np.testing.assert_allclose(
                hidden.precision, expected["precision"], rtol=1e-9
            )
            np.testing.assert_allclose(
                output.expected_mean, expected["observation_expected_mean"], rtol=1e-9
            )
            np.testing.assert_allclose(
                output.expected_precision,
                expected["observation_expected_precision"],
                rtol=1e-9,
            )


def _vectorised_pair(sizes, rng, coupling, volatility_updates, bottom_kind):
    """Build a Rust and a JAX network with the exact moments and the same weights."""
    kwargs = {
        "volatility_updates": volatility_updates,
        "coupling_fn": coupling,
        "feedforward_uncertainty": True,
        "coupling_moments": "exact",
    }
    rs, jx = RsDeepNetwork(**kwargs), JaxDeepNetwork(**kwargs)
    for position, size in enumerate(sizes):
        kind = bottom_kind if position == 0 else "volatile"
        rs = rs.add_layer(size) if kind == "volatile" else rs.add_layer(size, kind=kind)
        jx = jx.add_layer(size=size, kind=kind)
    weights = [rng.normal(scale=0.3, size=w.shape) for w in rs.get_weights()]
    rs.set_weights(weights)
    layers = list(jx.state.layers)
    jx.state = dataclasses.replace(
        jx.state,
        layers=tuple(
            [layers[0]]
            + [
                dataclasses.replace(layer, weights_mean=jnp.asarray(w))
                for layer, w in zip(layers[1:], weights)
            ]
        ),
    )
    return rs, jx


@pytest.mark.parametrize("coupling", ["leaky_relu", "relu", "tanh", "sigmoid"])
@pytest.mark.parametrize("volatility_updates", ["eHGF", "unbounded"])
@pytest.mark.parametrize("bottom_kind", ["volatile", "binary", "categorical"])
def test_vectorised_backends_match(coupling, volatility_updates, bottom_kind):
    """The JAX and the Rust vectorised backends agree with the exact moments.

    Both the sample-by-sample training and the batch update are compared, with the
    predictions, the prediction errors routed to the input layer and the weights.
    """
    rng = np.random.default_rng(3)
    n_outputs = 3 if bottom_kind == "categorical" else 2
    sizes = [n_outputs, 4, 4, 3]
    x = rng.normal(size=(8, 3))
    y = {
        "volatile": rng.normal(size=(8, 2)),
        "binary": rng.integers(0, 2, size=(8, 2)).astype(float),
        "categorical": np.eye(3)[rng.integers(0, 3, size=8)],
    }[bottom_kind]
    # tolerance of the single-precision engines
    tolerance = {"rtol": 3e-3, "atol": 1e-4}

    rs, jx = _vectorised_pair(sizes, rng, coupling, volatility_updates, bottom_kind)
    predictions = rs.fit(x, y, optimiser="sgd", learning_rate=0.05)
    jx.fit(x, y, optax.sgd(0.05))
    assert np.all(np.isfinite(predictions))
    np.testing.assert_allclose(predictions, np.asarray(jx.predictions), **tolerance)
    for w_rs, layer in zip(rs.get_weights(), jx.state.layers[1:]):
        np.testing.assert_allclose(w_rs, np.asarray(layer.weights_mean), **tolerance)
    # the read-only forward pass, over a batch and for one sample, at two time steps
    for time_step in (1.0, 0.1):
        np.testing.assert_allclose(
            rs.predict(x, time_step=time_step),
            np.asarray(jx.predict(x, time_step=time_step)),
            **tolerance,
        )
    np.testing.assert_allclose(
        rs.predict(x[0], time_step=0.1),
        np.asarray(jx.predict(x[0], time_step=0.1)),
        **tolerance,
    )

    rs, jx = _vectorised_pair(sizes, rng, coupling, volatility_updates, bottom_kind)
    optimiser = optax.sgd(0.05)
    for _ in range(3):
        x = rng.normal(size=(8, 3))
        errors = rs.batch_update(x, y, optimiser="sgd", learning_rate=0.05)
        jx.batch_update(x, y, optimiser=optimiser)
        np.testing.assert_allclose(errors, np.asarray(jx.input_errors), **tolerance)
    for w_rs, layer in zip(rs.get_weights(), jx.state.layers[1:]):
        np.testing.assert_allclose(w_rs, np.asarray(layer.weights_mean), **tolerance)


@pytest.mark.parametrize("cls", [JaxDeepNetwork, RsDeepNetwork], ids=["jax", "rust"])
def test_vectorised_moments_change_a_curved_coupling_only(cls):
    """The two settings differ for a curved coupling and agree for a linear one."""
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))

    def predictions(coupling, coupling_moments):
        net = cls(
            coupling_fn=coupling,
            feedforward_uncertainty=True,
            coupling_moments=coupling_moments,
        )
        net = net.add_layer(2).add_layer(4).add_layer(3)
        if cls is JaxDeepNetwork:
            net.fit(x, y, optax.sgd(0.05))
            return np.asarray(net.predictions)
        return np.asarray(net.fit(x, y, optimiser="sgd", learning_rate=0.05))

    np.testing.assert_allclose(
        predictions("identity", "exact"),
        predictions("identity", "taylor"),
        rtol=1e-4,
        atol=1e-5,
    )
    assert (
        np.max(np.abs(predictions("relu", "exact") - predictions("relu", "taylor")))
        > 1e-2
    )


def test_carried_precision_leaves_out_the_residual_variance():
    r"""A layer carries its evidence and its predicted precision without the residual.

    With the exact moments, the prediction stores the variance of the residuals of
    the linearised coupling functions, :math:`\sum_b \alpha_b^2 R_b`, and counts it
    in the conditional predicted variance. The precision carried to the next sample
    is :math:`\pi_y + (1/\bar{\pi}_a - \sum_b \alpha_b^2 R_b)^{-1}`, since the next
    prediction adds this variance afresh. The observation layer is left as it is,
    and without the exact moments the stored variance is zero.
    """
    rng = np.random.default_rng(19)
    x, y = jnp.asarray(rng.normal(size=3)), jnp.asarray(rng.normal(size=2))
    coupling = resolve_coupling_fn("leaky_relu")
    with enable_x64(True):
        for coupling_moments in ("exact", "taylor"):
            net = JaxDeepNetwork(
                coupling_fn="leaky_relu",
                feedforward_uncertainty=True,
                coupling_moments=coupling_moments,
            )
            net = net.add_layer(2).add_layer(4).add_layer(4).add_layer(3)
            net = net.weight_initialisation("he")
            swept = update_sweep(prediction_sweep(net.state, x), y)
            carried = carry_precisions(swept)
            for index in (1, 2):
                state, parent = swept.layers[index].state, swept.layers[index + 1]
                canonical = (
                    state.precision
                    + state.conditional_expected_precision
                    - state.expected_precision
                )
                if coupling_moments == "taylor":
                    np.testing.assert_array_equal(state.residual_variance, 0.0)
                    np.testing.assert_allclose(
                        carried.layers[index].state.precision, canonical, rtol=1e-12
                    )
                    continue

                _, variance, slope = coupling.moments(
                    parent.state.expected_mean, 1.0 / parent.state.expected_precision
                )
                residual = (parent.weights_mean[:, :-1] ** 2) @ (
                    variance - slope**2 / parent.state.expected_precision
                )
                assert np.all(np.asarray(residual) > 1e-4)
                np.testing.assert_allclose(state.residual_variance, residual, rtol=1e-9)
                own_precision = 1.0 / (
                    1.0 / state.conditional_expected_precision - residual
                )
                np.testing.assert_allclose(
                    carried.layers[index].state.precision,
                    state.precision + own_precision - state.expected_precision,
                    rtol=1e-9,
                )
                # more precise than carrying the conditional precision as it is
                assert np.all(carried.layers[index].state.precision > canonical)

            # the observation layer keeps the precision of its observation noise
            np.testing.assert_array_equal(
                carried.layers[0].state.precision, swept.layers[0].state.precision
            )


@pytest.mark.parametrize("cls", [JaxDeepNetwork, RsDeepNetwork], ids=["jax", "rust"])
@pytest.mark.parametrize("coupling_moments", ["taylor", "exact"])
def test_predict_takes_the_time_step_of_the_sweep(cls, coupling_moments):
    """The forward pass at the training time step is the next training prediction.

    With the exact moments, the predicted means depend on the predicted precisions,
    and so on the time step: the forward pass at another time step differs. With the
    first-order Taylor expansion, the predicted means do not depend on it.
    """
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(30, 3)), rng.normal(size=(30, 2))
    time_step = 0.01
    net = cls(
        coupling_fn="leaky_relu",
        feedforward_uncertainty=True,
        coupling_moments=coupling_moments,
    )
    net = net.add_layer(2).add_layer(5).add_layer(5).add_layer(3)
    net.weight_initialisation("he")

    def train(inputs, targets, **kwargs):
        if cls is JaxDeepNetwork:
            net.fit(inputs, targets, optax.sgd(0.01), time_step=time_step, **kwargs)
            return np.asarray(net.predictions)
        return np.asarray(
            net.fit(
                inputs,
                targets,
                optimiser="sgd",
                learning_rate=0.01,
                time_step=time_step,
                **kwargs,
            )
        )

    train(x, y)
    at_training_step = np.asarray(net.predict(x[:4], time_step=time_step))
    at_unit_step = np.asarray(net.predict(x[:4]))
    # the next prediction of the training loop, with the weights left as they are
    next_prediction = train(x[:1], y[:1], weight_update=False)[0]
    np.testing.assert_allclose(at_training_step[0], next_prediction, atol=1e-5)
    if coupling_moments == "exact":
        assert np.max(np.abs(at_training_step - at_unit_step)) > 1e-2
    else:
        np.testing.assert_allclose(at_training_step, at_unit_step, atol=1e-6)


def test_weight_factor_is_the_activation_the_prediction_read():
    r"""The parent-side factor is the derivative of the prediction by the weight.

    With the exact moments and a non-linear coupling function, a child is predicted from
    :math:`\sum_i w_{ai} \mathbb{E}[g(x_i)]` under the predicted beliefs of its parents,
    so the factor of the weight gradient is :math:`\mathbb{E}[g(x_i)]`, and 1 for the
    constant input. With the first-order Taylor expansion, and for the identity coupling
    under both settings, it is the coupling function at the posterior mean of the
    parent.
    """
    rng = np.random.default_rng(11)
    for name, coupling_moments in (
        ("leaky_relu", "exact"),
        ("leaky_relu", "taylor"),
        ("identity", "exact"),
    ):
        coupling = resolve_coupling_fn(name)
        net = JaxDeepNetwork(
            coupling_fn=name,
            feedforward_uncertainty=True,
            coupling_moments=coupling_moments,
        )
        net = net.add_layer(2).add_layer(4).add_layer(3).weight_initialisation("he")
        state = update_sweep(
            prediction_sweep(net.state, jnp.asarray(rng.normal(size=3))),
            jnp.asarray(rng.normal(size=2)),
        )
        parent, child = state.layers[1], state.layers[0]
        _, factor, _ = learning_weights_vectorised(
            parent_state=parent.state,
            child_state=child.state,
            coupling_fn=parent.coupling_fn,
            kind="standard",
            parent_has_constant=parent.add_constant_input,
            coupling_moments=coupling_moments,
        )
        np.testing.assert_allclose(factor[-1], 1.0)
        # the posterior of the parent moved away from its prediction
        assert np.max(np.abs(parent.state.mean - parent.state.expected_mean)) > 1e-3

        if (name, coupling_moments) != ("leaky_relu", "exact"):
            np.testing.assert_allclose(
                factor[:-1], coupling(parent.state.mean), rtol=1e-6
            )
            continue

        def predicted_mean(weights):
            return vectorised_layer_prediction(
                child_state=child.state,
                parent_state=parent.state,
                weights=weights,
                params=child.params,
                time_step=1.0,
                coupling_fn=parent.coupling_fn,
                parent_has_constant=parent.add_constant_input,
                has_volatility_parent=child.has_volatility_parent,
                is_input_layer=child.is_input_layer,
                feedforward_uncertainty=True,
                coupling_moments=coupling_moments,
            ).expected_mean

        # d(expected mean of child a) / d(w_ai): the same row for every child
        jacobian = jax.jacobian(predicted_mean)(parent.weights_mean)
        expected_value, _, _ = coupling.moments(
            parent.state.expected_mean, 1.0 / parent.state.expected_precision
        )
        np.testing.assert_allclose(factor[:-1], expected_value, rtol=1e-5)
        for node in range(2):
            np.testing.assert_allclose(jacobian[node, node], factor, rtol=1e-5)
        # the factor is not the coupling function at the posterior mean
        assert np.max(np.abs(factor[:-1] - coupling(parent.state.mean))) > 1e-2


def test_evidence_pullback_takes_the_mean_slope():
    """The evidence is carried up with the slope that the posterior update uses."""
    rng = np.random.default_rng(13)
    coupling = resolve_coupling_fn("relu")
    net = JaxDeepNetwork(
        coupling_fn="relu", feedforward_uncertainty=True, coupling_moments="exact"
    )
    net = net.add_layer(2).add_layer(4).add_layer(3).weight_initialisation("he")
    state = prediction_sweep(net.state, jnp.asarray(rng.normal(size=3)))
    parent = state.layers[1]
    evidence, noise = jnp.array([2.0, 0.5]), jnp.array([0.1, 0.3])
    softened = evidence / (1.0 + noise * evidence)
    base = jnp.matmul(parent.weights_mean[:, :-1].T ** 2, softened)

    def pullback(coupling_moments):
        return evidence_pullback(
            parent_state=parent.state,
            child_evidence=evidence,
            child_process_noise=noise,
            weights=parent.weights_mean,
            coupling_fn=parent.coupling_fn,
            parent_has_constant=parent.add_constant_input,
            coupling_moments=coupling_moments,
        )

    _, _, slope = coupling.moments(
        parent.state.expected_mean, 1.0 / parent.state.expected_precision
    )
    np.testing.assert_allclose(pullback("exact"), base * slope**2, rtol=1e-5)
    np.testing.assert_allclose(
        pullback("taylor"),
        base * coupling.df(parent.state.expected_mean) ** 2,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "learning_kind", ["standard", "precision_weighted", "synaptic_uncertainty"]
)
def test_learning_kinds_run_with_the_exact_moments(learning_kind):
    """Every learning kind trains with the exact moments and stays finite."""
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
    net = JaxDeepNetwork(
        coupling_fn="leaky_relu", feedforward_uncertainty=True, coupling_moments="exact"
    )
    net = net.add_layer(2).add_layer(5).add_layer(3).weight_initialisation("he")
    kwargs = (
        {"learning_kwargs": {"window": 20}}
        if learning_kind == "synaptic_uncertainty"
        else {}
    )
    before = [np.asarray(layer.weights_mean) for layer in net.state.layers[1:]]
    net.fit(x, y, optax.sgd(0.01), learning_kind=learning_kind, **kwargs)
    assert np.all(np.isfinite(np.asarray(net.predictions)))
    for weights, layer in zip(before, net.state.layers[1:]):
        after = np.asarray(layer.weights_mean)
        assert np.all(np.isfinite(after))
        assert np.max(np.abs(after - weights)) > 0.0


def test_continuous_layers_take_the_exact_moments():
    """A network of continuous layers matches the nodalised network.

    The vectorised continuous layers and the nodalised nodes implement the same updates,
    with the noise parent of the observation layer included.
    """
    timeseries = np.asarray(load_data("continuous"))[:80]
    with enable_x64(True):
        nodalised = (
            PyNetwork(volatility_updates="eHGF", coupling_moments="exact")
            .add_nodes()
            .add_nodes(
                value_children=0, coupling_fn=(jax.nn.relu,), tonic_volatility=-2.0
            )
            .add_nodes(volatility_children=0, mean=1.0, tonic_volatility=-4.0)
            .input_data(input_data=timeseries)
        )
        vectorised = (
            JaxDeepNetwork(
                volatility_updates="eHGF",
                coupling_fn="relu",
                coupling_moments="exact",
            )
            .add_layer(1, kind="continuous")
            .add_layer(1, kind="continuous", tonic_volatility=-2.0)
            .add_layer(
                1,
                kind="continuous",
                value_children=None,
                volatility_children=0,
                mean=1.0,
                tonic_volatility=-4.0,
            )
        )
        vectorised.input_data(
            timeseries[:, None], record=("mean", "precision", "expected_mean")
        )
        for node_idx in (1, 2):
            for key in ("mean", "precision", "expected_mean"):
                np.testing.assert_allclose(
                    np.asarray(vectorised.trajectories[key][node_idx])[:, 0],
                    np.asarray(nodalised.node_trajectories[node_idx][key]),
                    rtol=1e-8,
                    err_msg=f"node {node_idx}, {key}",
                )


# ----------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [PyNetwork, RsNetwork], ids=["jax", "rust"])
def test_nodalised_arguments(cls):
    """The exact moments require the relaxed updates and a valid name."""
    with pytest.raises(ValueError, match="Invalid coupling_moments"):
        cls(coupling_moments="nope")
    with pytest.raises(ValueError, match="mean_field_updates"):
        cls(coupling_moments="exact", mean_field_updates=True)
    cls(coupling_moments="taylor", mean_field_updates=True)


@pytest.mark.parametrize("cls", [JaxDeepNetwork, RsDeepNetwork], ids=["jax", "rust"])
def test_vectorised_arguments(cls):
    """The exact moments require the uncertainty of the value parents."""
    with pytest.raises(ValueError, match="Invalid coupling_moments"):
        cls(coupling_moments="nope")
    with pytest.raises(ValueError, match="feedforward_uncertainty"):
        cls(coupling_moments="exact").add_layer(2)
    with pytest.raises(ValueError, match="predict_precision"):
        cls(
            coupling_moments="exact",
            feedforward_uncertainty=True,
            predict_precision=False,
        ).add_layer(2)
    cls(coupling_moments="exact", feedforward_uncertainty=True).add_layer(2)


def test_jax_deep_network_requires_the_relaxed_updates():
    """The exact moments are incompatible with the mean-field updates."""
    with pytest.raises(ValueError, match="mean_field_updates"):
        JaxDeepNetwork(coupling_moments="exact", mean_field_updates=True)
