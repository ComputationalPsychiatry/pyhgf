# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

r"""The autoconnection strength in the predicted precision of a continuous node.

The carried variance is :math:`\lambda^2 / \pi` (scaling a Gaussian by :math:`\lambda`
scales its variance by :math:`\lambda^2`). These tests pin it at hand-decidable values
and check both backends and both prediction schemes apply it.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.model.deep_network import DeepNetwork
from pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node import (
    precision_update_missing_values,
)
from pyhgf.updates.prediction.continuous import (
    predict_precision,
    predict_precision_mean_field,
)

# The tonic volatility of a fresh state node, so its predicted volatility over a unit
# time step is exp(-4).
TONIC_VOLATILITY = -4.0
OMEGA = float(jnp.exp(TONIC_VOLATILITY))

FIELDS = ("mean", "precision", "expected_mean", "expected_precision")
LAMBDAS = [0.0, 0.2, 0.9, 1.0]


def _one_parent_attributes(autoconnection_strength, tonic_volatility=TONIC_VOLATILITY):
    """Attributes and edges for an input node plus one value parent.

    ``get_network`` leaves ``time_step`` at 0 (which would zero the volatility); set to
    1 here to stand in for the propagation step.
    """
    network = (
        Network()
        .add_nodes()
        .add_nodes(
            value_children=0,
            node_parameters={
                "autoconnection_strength": autoconnection_strength,
                "tonic_volatility": tonic_volatility,
            },
        )
    )
    attributes, edges, _ = network.get_network()
    attributes[-1]["time_step"] = 1.0
    return attributes, edges


def test_predicted_precision_matches_hand_computation():
    """π̂ = 1 / (λ²/π + Ω), checked against hand arithmetic.

    λ = 0.2, π = 1, Ω = 0.1 → carried 0.04, variance 0.14, precision 7.142857;
    dropping λ² would give 1/1.1 = 0.909091 (≈8× less confident than it should be).
    """
    # Ω = t · exp(ω) with t = 1, so ω = log(0.1) gives Ω = 0.1 exactly.
    attributes, edges = _one_parent_attributes(0.2, float(jnp.log(0.1)))
    attributes[1]["precision"] = 1.0

    expected_precision, conditional, _ = predict_precision(attributes, edges, 1)

    assert jnp.isclose(conditional, 1.0 / 0.14, rtol=1e-5)
    # The node's only parent is below it, so there is no value-coupling variance and
    # the marginal predicted precision equals the conditional one.
    assert jnp.isclose(expected_precision, conditional)


@pytest.mark.parametrize(
    "predict",
    [predict_precision, predict_precision_mean_field],
    ids=["piHGF", "mean_field"],
)
def test_zero_autoconnection_keeps_no_variance(predict):
    """At λ = 0 the predicted variance is the predicted volatility alone.

    Both schemes agree (they differ only in how parents enter, and this node has
    none above it): whatever the previous variance, none of it survives.
    """
    attributes, edges = _one_parent_attributes(0.0)

    for precision in (0.25, 1.0, 100.0):
        attributes[1]["precision"] = precision
        _, conditional, _ = predict(attributes, edges, 1)
        assert jnp.isclose(conditional, 1.0 / OMEGA, rtol=1e-5)


@pytest.mark.parametrize(
    "predict",
    [predict_precision, predict_precision_mean_field],
    ids=["piHGF", "mean_field"],
)
def test_unit_autoconnection_is_the_canonical_expression(predict):
    """At λ = 1 the predicted precision is bit-identical to 1 / (1/π + Ω).

    λ² = 1, so the factor cannot perturb the result even in the last bit — this
    bounds the fix's reach: every default-λ network is untouched.
    """
    attributes, edges = _one_parent_attributes(1.0)

    for precision in (0.25, 1.0, 100.0):
        attributes[1]["precision"] = precision
        _, conditional, _ = predict(attributes, edges, 1)
        canonical = 1.0 / (1.0 / jnp.array(precision) + jnp.exp(TONIC_VOLATILITY))
        assert conditional == canonical


@pytest.mark.parametrize("autoconnection_strength", LAMBDAS)
def test_nodalised_and_vectorised_backends_agree(autoconnection_strength):
    """The JAX nodalised and vectorised continuous backends apply the same factor."""
    timeseries = load_data("continuous")

    nodalised = (
        Network()
        .add_nodes()
        .add_nodes(
            value_children=0,
            node_parameters={
                "autoconnection_strength": autoconnection_strength,
                "tonic_volatility": TONIC_VOLATILITY,
            },
        )
        .input_data(input_data=timeseries)
    )
    vectorised = (
        DeepNetwork()
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            autoconnection_strength=autoconnection_strength,
            tonic_volatility=TONIC_VOLATILITY,
        )
        .input_data(timeseries, record=FIELDS)
    )

    for field in FIELDS:
        assert jnp.allclose(
            nodalised.node_trajectories[1][field],
            vectorised.trajectories[field][1][:, 0],
            rtol=1e-5,
            atol=1e-6,
        ), f"λ={autoconnection_strength}: '{field}' mismatch"


@pytest.mark.parametrize("autoconnection_strength", LAMBDAS)
def test_jax_and_rust_nodalised_backends_agree(autoconnection_strength):
    """The JAX and Rust nodalised backends apply the same factor.

    JAX is float32, Rust is f64, so this is a float32 tolerance, not an equality.
    """
    timeseries = load_data("continuous")

    jax_net = (
        Network()
        .add_nodes()
        .add_nodes(
            value_children=0,
            node_parameters={
                "autoconnection_strength": autoconnection_strength,
                "tonic_volatility": TONIC_VOLATILITY,
            },
        )
        .input_data(input_data=timeseries)
    )
    rust_net = (
        RsNetwork()
        .add_nodes()
        .add_nodes(
            value_children=0,
            autoconnection_strength=autoconnection_strength,
            tonic_volatility=TONIC_VOLATILITY,
        )
        .input_data(input_data=timeseries)
    )

    for field in FIELDS:
        assert np.allclose(
            np.asarray(jax_net.node_trajectories[1][field], dtype=np.float64),
            np.asarray(rust_net.node_trajectories[1][field], dtype=np.float64),
            rtol=1e-4,
        ), f"λ={autoconnection_strength}: '{field}' mismatch"


@pytest.mark.parametrize("autoconnection_strength", LAMBDAS)
def test_missing_value_precision_ages_by_the_carried_variance(autoconnection_strength):
    """An unobserved node ages its precision using the same carried variance.

    With nothing observed, the posterior precision is the predicted precision under the
    node's own random walk; the mean ages by λ, so the variance ages by λ².
    """
    attributes, edges = _one_parent_attributes(autoconnection_strength)
    attributes[1]["precision"] = 4.0
    # No child reported an observation at this step.
    attributes[0]["observed"] = 0

    aged = precision_update_missing_values(attributes, edges, 1)

    assert jnp.isclose(aged, 1.0 / (autoconnection_strength**2 / 4.0 + OMEGA))
