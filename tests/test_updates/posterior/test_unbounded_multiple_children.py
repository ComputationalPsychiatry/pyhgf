"""Unbounded volatility update for nodes with several children.

The unbounded update is extended to several volatility children, and to nodes that are
value and volatility parents at the same time, following section 3.6 of Mathys, C.,
Legrand, N., Waade, P. T., Mikuš, N., & Weber, L. A. (2026). Robust volatility updates
for Hierarchical Gaussian Filtering. arXiv:2605.00966.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from pyhgf.model import Network
from pyhgf.updates.posterior.continuous.continuous_node_posterior_update import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_mean_field,
)
from pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded import (  # noqa: E501
    continuous_node_posterior_update_unbounded,
    posterior_update_unbounded,
)

SEED = 123
N_TRIALS = 300


def shared_volatility_parent(mean_field_updates: bool, order=(2, 3)) -> Network:
    """Two inputs, one value parent each, one shared volatility parent (node 4)."""
    return (
        Network(volatility_updates="unbounded", mean_field_updates=mean_field_updates)
        .add_nodes(precision=1e4, n_nodes=2)
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(volatility_children=list(order), mean=4.0)
    )


def mixed_parent(mean_field_updates: bool) -> Network:
    """Node 4 is the value parent of node 2 and the volatility parent of node 3."""
    return (
        Network(volatility_updates="unbounded", mean_field_updates=mean_field_updates)
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(value_children=2, volatility_children=3)
    )


def two_inputs_with_bursts() -> np.ndarray:
    """Gaussian noise with a burst of volatility on each input, at different times."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 1.0, size=(N_TRIALS, 2))
    x[50:100, 0] *= 3.0
    x[150:250, 1] *= 3.0
    return x


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_every_volatility_child_reaches_the_parent(mean_field_updates):
    """A burst seen by the second volatility child only must move the shared parent."""
    x = two_inputs_with_bursts()
    x_without_burst = x.copy()
    x_without_burst[150:250, 1] /= 3.0

    mu = shared_volatility_parent(mean_field_updates).input_data(input_data=x)
    mu = np.asarray(mu.node_trajectories[4]["mean"])
    mu_without_burst = shared_volatility_parent(mean_field_updates).input_data(
        input_data=x_without_burst
    )
    mu_without_burst = np.asarray(mu_without_burst.node_trajectories[4]["mean"])

    assert np.all(np.isfinite(mu))
    # before the burst, both runs are identical
    np.testing.assert_array_equal(mu[:150], mu_without_burst[:150])
    # the burst increases the volatility belief of the shared parent
    assert mu[150:250].mean() - mu_without_burst[150:250].mean() > 0.1


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_volatility_children_order_does_not_matter(mean_field_updates):
    """The posterior is a symmetric function of the volatility children."""
    x = two_inputs_with_bursts()
    a = shared_volatility_parent(mean_field_updates, order=(2, 3)).input_data(
        input_data=x
    )
    b = shared_volatility_parent(mean_field_updates, order=(3, 2)).input_data(
        input_data=x
    )
    for key in ("mean", "precision"):
        # float32 summation order differs between the two networks
        np.testing.assert_allclose(
            a.node_trajectories[4][key],
            b.node_trajectories[4][key],
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_value_child_of_a_mixed_parent_is_used(mean_field_updates):
    """A node that is value and volatility parent uses its value child."""
    rng = np.random.default_rng(SEED)
    x = np.cumsum(rng.normal(0.0, 0.3, size=(N_TRIALS, 2)), axis=0)
    x_shifted = x.copy()
    x_shifted[100:, 0] += 3.0  # seen by the value child (node 2) only

    mu = mixed_parent(mean_field_updates).input_data(input_data=x)
    mu = np.asarray(mu.node_trajectories[4]["mean"])
    mu_shifted = mixed_parent(mean_field_updates).input_data(input_data=x_shifted)
    mu_shifted = np.asarray(mu_shifted.node_trajectories[4]["mean"])

    assert np.all(np.isfinite(mu_shifted))
    np.testing.assert_array_equal(mu[:100], mu_shifted[:100])
    # the parent follows the step transiently, before the value child absorbs it
    assert mu_shifted[100:110].mean() - mu[100:110].mean() > 0.1


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_uninformative_volatility_child_leaves_the_value_update(mean_field_updates):
    """A flat volatility likelihood leaves the regular value-coupling posterior.

    With a very large carried variance, the volatility child's terms of the variational
    energy do not depend on the parent, so every expansion collapses on the effective
    prior built from the value child. This is the posterior of the standard update for a
    pure value parent (relaxed or mean-field).
    """
    network = mixed_parent(mean_field_updates)
    attributes, edges, _ = network.get_network()
    attributes[-1]["time_step"] = 1.0
    # value child: an informative prediction error
    attributes[2]["temp"]["value_prediction_error"] = 1.5
    attributes[2]["mean"] = 1.5
    attributes[2]["precision"] = 3.0
    attributes[2]["expected_precision"] = 2.0
    attributes[2]["temp"]["conditional_expected_precision"] = 2.0
    # volatility child: carried variance so large that Ω(x) / (α + Ω(x)) ≈ 0
    attributes[3]["temp"]["current_variance"] = 1e12
    attributes[3]["mean"] = 0.3
    attributes[3]["precision"] = 1.0

    new = continuous_node_posterior_update_unbounded(
        attributes=attributes,
        node_idx=4,
        edges=edges,
        mean_field_updates=mean_field_updates,
    )
    expected_precision, expected_mean = new[4]["precision"], new[4]["mean"]

    attributes, edges, _ = network.get_network()
    attributes[-1]["time_step"] = 1.0
    attributes[2]["temp"]["value_prediction_error"] = 1.5
    attributes[2]["mean"] = 1.5
    attributes[2]["precision"] = 3.0
    attributes[2]["expected_precision"] = 2.0
    attributes[2]["temp"]["conditional_expected_precision"] = 2.0
    value_only_edges = (
        edges[:4] + (edges[4]._replace(volatility_children=None),) + edges[5:]
    )
    standard_update = (
        continuous_node_posterior_update_mean_field
        if mean_field_updates
        else continuous_node_posterior_update
    )
    reference = standard_update(
        attributes=attributes, node_idx=4, edges=value_only_edges
    )

    assert jnp.isclose(expected_precision, reference[4]["precision"], rtol=1e-5)
    assert jnp.isclose(expected_mean, reference[4]["mean"], rtol=1e-5)


def _set_two_volatility_children(
    attributes, expected_mean, expected_precision, children
) -> None:
    """Set the update inputs of node 4 with consistent child predictions."""
    attributes[-1]["time_step"] = 1.0
    attributes[4]["expected_mean"] = expected_mean
    attributes[4]["expected_precision"] = expected_precision
    for child_idx, (deviation, precision, alpha, omega) in zip((2, 3), children):
        attributes[child_idx]["tonic_volatility"] = omega
        attributes[child_idx]["temp"]["current_variance"] = alpha
        attributes[child_idx]["expected_mean"] = 0.0
        attributes[child_idx]["mean"] = deviation
        attributes[child_idx]["precision"] = precision
        # mean-field prediction of the child, with κ = 1 and t = 1
        attributes[child_idx]["expected_precision"] = 1.0 / (
            alpha + np.exp(omega + expected_mean)
        )


def _kl_to_variational_posterior(
    expected_mean, expected_precision, children, mean, precision
) -> float:
    """KL(p || q), p ∝ exp(I(x)) by quadrature (float64), q = N(mean, 1/precision)."""
    sd = 1.0 / np.sqrt(expected_precision)
    x = np.linspace(expected_mean - 40.0 * sd, expected_mean + 40.0 * sd, 200_001)
    dx = x[1] - x[0]
    log_p = -0.5 * expected_precision * (x - expected_mean) ** 2
    for deviation, child_precision, alpha, omega in children:
        beta = 1.0 / child_precision + deviation**2
        log_denom = np.logaddexp(np.log(alpha), x + omega)
        log_p += -0.5 * log_denom - 0.5 * beta * np.exp(-log_denom)
    log_p -= logsumexp(log_p) + np.log(dx)
    log_q = 0.5 * np.log(precision / (2.0 * np.pi)) - 0.5 * precision * (x - mean) ** 2
    return float(np.sum(np.exp(log_p) * (log_p - log_q)) * dx)


@pytest.mark.parametrize(
    "expected_mean, expected_precision, children",
    [
        # bimodal variational energy, canonical parameters of fig. 3 in the paper
        (-7.0, 0.5, ((1.0, 1e6, 0.05, 0.0), (1.0, 1e6, 0.05, 0.0))),
        # two children with different tonic volatilities and prediction errors
        (1.0, 2.0, ((0.3, 10.0, 0.1, -3.0), (2.5, 10.0, 0.1, -1.0))),
    ],
)
def test_two_children_track_the_variational_posterior(
    expected_mean, expected_precision, children
):
    """The Gaussian posterior is close to exp(I(x)), which includes both children.

    The bound (KL < 0.15) is above the KL of the existing single-child update to its own
    one-child variational posterior in the bimodal regime (0.12, measured when writing
    this test), and the ratio to the first-child-only update documents the size of the
    error that this update removes (KL 1.28 and 0.44 in these two regimes).
    """
    network = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(volatility_children=[2, 3])
    )
    attributes, edges, _ = network.get_network()
    _set_two_volatility_children(
        attributes, expected_mean, expected_precision, children
    )

    precision, mean = posterior_update_unbounded(
        attributes=attributes, node_idx=4, edges=edges
    )
    # the update the node received before both children were used
    first_child_edges = (
        edges[:4] + (edges[4]._replace(volatility_children=(2,)),) + edges[5:]
    )
    first_child_precision, first_child_mean = posterior_update_unbounded(
        attributes=attributes, node_idx=4, edges=first_child_edges
    )
    kl = _kl_to_variational_posterior(
        expected_mean, expected_precision, children, float(mean), float(precision)
    )
    kl_first_child = _kl_to_variational_posterior(
        expected_mean,
        expected_precision,
        children,
        float(first_child_mean),
        float(first_child_precision),
    )
    assert kl < 0.15
    assert kl < kl_first_child / 4.0


def test_zero_coupling_child_is_left_out():
    """A volatility child with a zero coupling is left out, with a finite gradient.

    The terms of this child do not depend on the node, and its Lambert expansion would
    divide by the coupling. The posterior is the one of the node without this child.
    """
    network = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(volatility_children=[2, 3])
    )
    attributes, edges, _ = network.get_network()
    _set_two_volatility_children(
        attributes, 1.0, 2.0, ((0.3, 10.0, 0.1, -3.0), (2.5, 10.0, 0.1, -1.0))
    )
    first_child_edges = (
        edges[:4] + (edges[4]._replace(volatility_children=(2,)),) + edges[5:]
    )
    expected_precision, expected_mean = posterior_update_unbounded(
        attributes=attributes, node_idx=4, edges=first_child_edges
    )

    def posterior(couplings):
        node = {**attributes[4], "volatility_coupling_children": couplings}
        return posterior_update_unbounded(
            attributes={**attributes, 4: node}, node_idx=4, edges=edges
        )

    couplings = jnp.array([1.0, 0.0])
    precision, mean = posterior(couplings)
    assert jnp.isclose(precision, expected_precision, rtol=1e-5)
    assert jnp.isclose(mean, expected_mean, rtol=1e-5)
    for gradient in jax.jacobian(posterior)(couplings):
        assert jnp.all(jnp.isfinite(gradient))


def test_positive_precision_over_a_grid_of_prediction_errors():
    """Posterior precision is finite and positive, as the standard update is not."""
    network = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(volatility_children=[2, 3])
    )
    attributes, edges, _ = network.get_network()
    # canonical units of section 3.5.1 in the paper: α = 0.005, β/α up to 200,
    # γ in [-15, 15]; the second child adds a large prediction error elsewhere
    ratio, gamma = np.meshgrid(
        np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]),
        np.linspace(-15.0, 15.0, 61),
    )
    ratio, gamma = jnp.asarray(ratio.ravel()), jnp.asarray(gamma.ravel())
    attributes[-1]["time_step"] = 1.0
    attributes[4]["expected_mean"] = gamma
    attributes[4]["expected_precision"] = 0.5
    for child_idx, omega, alpha, child_ratio in (
        (2, 0.0, 0.005, ratio),
        (3, -2.0, 0.005, 50.0),
    ):
        attributes[child_idx]["tonic_volatility"] = omega
        attributes[child_idx]["temp"]["current_variance"] = alpha
        attributes[child_idx]["expected_mean"] = 0.0
        attributes[child_idx]["precision"] = 1e6
        attributes[child_idx]["mean"] = jnp.sqrt(alpha * child_ratio)
        attributes[child_idx]["expected_precision"] = 1.0 / (
            alpha + jnp.exp(gamma + omega)
        )

    precision, mean = posterior_update_unbounded(
        attributes=attributes, node_idx=4, edges=edges
    )
    assert precision.shape == gamma.shape
    assert jnp.all(jnp.isfinite(precision)) and jnp.all(jnp.isfinite(mean))
    assert jnp.all(precision > 0.0)
