# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyhgf import load_data
from pyhgf.math import gaussian_surprise
from pyhgf.model import Network
from pyhgf.utils import beliefs_propagation


def test_one_node_hgf():
    """Test the one node HGF with one observation step."""
    # one level HGF and one observation step
    one_node__hgf = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data(input_data=jnp.array([0.2]))
    )

    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.0, 1.0, 0.2, 0.0],
    ):
        assert jnp.isclose(one_node__hgf.node_trajectories[0][key], val)
    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.9820137, 0.98201376, 0.10090748, 0.0],
    ):
        assert jnp.isclose(one_node__hgf.node_trajectories[1][key], val)


def test_two_nodes_hgf():
    r"""Test the two nodes HGF with one observation step.

    The volatility parent of the input (node 2) is its noise parent, and the values
    follow Eqs. 64 to 78 of [1]_. The precision of the observation
    noise is :math:`\hat{\pi}_0 = \exp(-\hat{\mu}_2) = 1` (Eq. 66). The value
    parent is updated first, :math:`\pi_1 = \hat{\pi}_1 + \hat{\pi}_0` and
    :math:`\mu_1 = \hat{\pi}_0 u / \pi_1` (Eqs. 69 and 70), with
    :math:`\hat{\pi}_1 = 1/(1 + e^{-4})`. The noise prediction error reads this
    posterior, :math:`\epsilon = \hat{\pi}_0 / \pi_1 + \hat{\pi}_0 (u - \mu_1)^2 - 1`
    (Eq. 72), and the noise parent takes :math:`\pi_2 = \hat{\pi}_2 + \tfrac12 (1 +
    \epsilon)` and :math:`\mu_2 = \epsilon / (2 \pi_2)` (Eqs. 74 and 78).

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    # one level HGF with a volatility parent and one observation step
    two_nodes__hgf = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
        .input_data(input_data=jnp.array([0.2]))
    )

    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.0, 1.0, 0.2, 0.0],
    ):
        assert jnp.isclose(two_nodes__hgf.node_trajectories[0][key], val)
    assert jnp.isclose(
        two_nodes__hgf.node_trajectories[0]["temp"]["volatility_prediction_error"],
        -0.48564331,
    )
    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.98201379, 0.98201379, 0.10090747, 0.0],
    ):
        assert jnp.isclose(two_nodes__hgf.node_trajectories[1][key], val)
    for key, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.23919213, 0.98201379, -0.19595158, 0.0],
    ):
        assert jnp.isclose(two_nodes__hgf.node_trajectories[2][key], val)


def test_nonlinear_coupling_fn():
    """Tests if the coupling function is passed correctly into the network."""

    # creating a simple coupling function
    def coupling_fn(x):
        return jnp.sin(x)

    def identity_fn(x):
        return x

    # I create a network with a node with 2 value children
    test_HGF = (
        Network()
        .add_nodes(n_nodes=2)
        .add_nodes(value_children=0, coupling_fn=(identity_fn,))
        .add_nodes(value_children=1, coupling_fn=(coupling_fn,))
    )

    # check if the number of coupling fn matches the number of children
    coupling_fn_length = []
    children_number = []
    for node_idx in range(2, len(test_HGF.edges)):
        coupling_fn_length.append(len(test_HGF.edges[node_idx].coupling_fn))
        children_number.append(len(test_HGF.edges[node_idx].value_children))

    assert children_number == coupling_fn_length

    # pass data
    test_HGF.input_data(input_data=jnp.array([[0.2, 0.4], [0.2, 0.4]]))

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [2.9125834, 1.9125834, 0.13492969, 0.10090748],
    ):
        assert jnp.isclose(test_HGF.node_trajectories[2][idx][1], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [2.9124038, 1.9125834, 0.2689424, 0.20181496],
    ):
        assert jnp.isclose(test_HGF.node_trajectories[3][idx][1], val)


def test_continuous_scan_loop():
    """Test the continuous HGF with a scan loop."""
    timeserie = load_data("continuous")

    two_level_hgf = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
        .input_data(input_data=timeserie)
    )

    # Node 2 is the noise parent of the input (see test_two_nodes_hgf above). It
    # learns the observation noise of this series: the predicted noise precision
    # exp(-mu_2) of the input reaches about 2657. The Rust backend gives the same
    # values to about 1e-6.
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.0, 2656.6162, 0.8241, 0.78762567],
    ):
        assert jnp.isclose(two_level_hgf.node_trajectories[0][idx][-1], val)

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [2710.1343, 53.518047, 0.82337976, 0.78762567],
    ):
        assert jnp.isclose(two_level_hgf.node_trajectories[1][idx][-1], val)

    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [5.4241695, 4.9333544, -7.886501, -7.8848085],
    ):
        assert jnp.isclose(two_level_hgf.node_trajectories[2][idx][-1], val)


def test_record_trajectories():
    """Test input_data with record_trajectories=True and False."""
    timeserie = load_data("continuous")

    # record_trajectories=True (default) should store full trajectories
    net_with = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data(input_data=timeserie, record_trajectories=True)
    )
    assert net_with.node_trajectories is not None
    assert len(net_with.node_trajectories[0]["mean"]) == len(timeserie)
    assert net_with.last_attributes is not None

    # record_trajectories=False should skip trajectories but keep last state
    net_without = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data(input_data=timeserie, record_trajectories=False)
    )
    assert net_without.node_trajectories is None
    assert net_without.last_attributes is not None

    # final state should be the same regardless of recording
    for node_idx in [0, 1]:
        for key in ["mean", "precision"]:
            assert jnp.isclose(
                net_with.last_attributes[node_idx][key],
                net_without.last_attributes[node_idx][key],
            )


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_noise_parent_learns_the_observation_noise(mean_field_updates):
    r"""The noise parent of an input recovers the variance of the observation noise.

    The value of the input follows a random walk with step variance 0.5 and the
    observation noise has variance 1, so the value parent adds a predicted variance
    of about 1. The noise prediction error reads the posterior of the value parent,
    so this variance is not attributed to the noise, in the relaxed and in the
    mean-field updates: the predicted noise variance :math:`1/\hat{\pi}_u` converges
    to the variance of the noise.
    """
    rng = np.random.default_rng(1)
    n = 3000
    value = np.cumsum(rng.normal(0.0, np.sqrt(0.5), n))
    noise = rng.normal(0.0, 1.0, n)
    network = (
        Network(mean_field_updates=mean_field_updates)
        .add_nodes()
        .add_nodes(value_children=0, tonic_volatility=np.log(0.5))
        .add_nodes(volatility_children=0, tonic_volatility=-6.0)
        .input_data(input_data=value + noise)
    )
    noise_variance = 1.0 / np.asarray(
        network.node_trajectories[0]["expected_precision"]
    )
    assert np.isclose(
        np.mean(noise_variance[n // 2 :]), np.var(noise[n // 2 :]), rtol=0.1
    )


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_noise_parent_has_finite_gradients(mean_field_updates):
    """The surprise of an input with a noise parent is differentiable.

    The gradient is taken with respect to the prior mean of the noise parent, through
    ten trials, the first of which is an observation equal to its prediction.
    """
    network = (
        Network(mean_field_updates=mean_field_updates)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
    )
    network.create_belief_propagation_fn()
    attributes, edges, update_sequence = network.get_network()
    observations = jnp.concatenate([
        jnp.zeros(1),
        jnp.asarray(np.random.default_rng(7).normal(0.0, 2.0, 9)),
    ])

    def surprise(noise_parent_mean):
        state = jax.tree_util.tree_map(lambda x: x, attributes)
        state[2]["mean"] = noise_parent_mean
        total = 0.0
        for observation in observations:
            state, _ = beliefs_propagation(
                attributes=state,
                inputs=(jnp.array([observation]), jnp.array([1]), 1.0, None),
                update_sequence=update_sequence,
                edges=edges,
                input_idxs=(0,),
            )
            total += gaussian_surprise(
                x=state[0]["mean"],
                expected_mean=state[0]["expected_mean"],
                expected_precision=state[0]["expected_precision"],
            )
        return total

    assert jnp.isfinite(surprise(0.0))
    assert jnp.isfinite(jax.grad(surprise)(0.0))
