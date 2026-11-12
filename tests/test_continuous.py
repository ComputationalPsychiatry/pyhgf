# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.math import gaussian_surprise
from pyhgf.model import Network
from pyhgf.utils import beliefs_propagation

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64


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


# A chain b -> a -> u: the value parent b, its value child a, and the input u (value
# child of a). alpha couples b to a, beta couples a to u.
VALUE_COUPLING_CHAIN = dict(
    b=dict(mean=0.3, precision=2.0, tonic_volatility=-1.0, tonic_drift=0.2),
    a=dict(
        mean=-0.2,
        precision=1.5,
        tonic_volatility=-1.5,
        tonic_drift=-0.1,
        autoconnection_strength=0.7,
    ),
    alpha=1.3,
    beta=0.8,
    input_precision=4.0,
)


def _value_coupling_chain_reference(u, time_steps, p, mean_field_updates):
    r"""Filter the chain :math:`b \to a \to u` with the generative model of [1]_.

    A value parent :math:`b` sets the drift of its value child :math:`a` (Eqs. 8 and
    36 of [1]_),

    .. math::

        x_a^{(k)} \sim \mathcal{N}\left(\lambda_a x_a^{(k-1)}
            + t^{(k)} \left(\rho_a + \alpha \, x_b^{(k)}\right),
            t^{(k)} \exp(\omega_a)\right),

    so the prediction of :math:`a` moves by :math:`t \alpha` per unit of
    :math:`x_b`. The input is generated by its value parent rather than drifted by it
    (Eqs. 9 and 64 of [1]_), :math:`u^{(k)} \sim \mathcal{N}(\beta \, x_a^{(k)},
    1 / \pi_u)`, whatever the time step. Under the relaxed updates, the joint prior of
    the linear chain is Gaussian given the factorised posteriors of the previous step,
    and the reference conditions it on :math:`u` exactly. Under the mean-field
    updates, it applies the canonical updates (Eqs. 38, 39, 69 and 70 of [1]_) with
    the derivative :math:`t \alpha` of the prediction of :math:`a`.

    Returns the trajectories of ``"b"``, ``"a"`` and ``"u"``.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    fields = ("mean", "precision", "expected_mean", "expected_precision")
    b, a = p["b"], p["a"]
    lambda_a = a["autoconnection_strength"]
    alpha, beta, pi_u = p["alpha"], p["beta"], p["input_precision"]
    mu_b, pi_b = b["mean"], b["precision"]
    mu_a, pi_a = a["mean"], a["precision"]
    trajectories = {node: {field: [] for field in fields} for node in ("b", "a", "u")}

    for observation, t in zip(u, time_steps):
        # Prediction step (Eqs. 34 to 37). The drift of a takes t, as does its
        # derivative with respect to x_b.
        muhat_b = mu_b + t * b["tonic_drift"]
        pihat_b = 1 / (1 / pi_b + t * math.exp(b["tonic_volatility"]))
        c = t * alpha
        muhat_a = lambda_a * mu_a + t * a["tonic_drift"] + c * muhat_b
        pibar_a = 1 / (lambda_a**2 / pi_a + t * math.exp(a["tonic_volatility"]))
        pihat_a = pibar_a if mean_field_updates else 1 / (1 / pibar_a + c**2 / pihat_b)
        # The input is generated by its value parent, without the time step.
        muhat_u = beta * muhat_a

        if mean_field_updates:
            # Canonical updates (Eqs. 69 and 70, then 38 and 39).
            pi_a = pihat_a + beta**2 * pi_u
            mu_a = muhat_a + beta * pi_u / pi_a * (observation - muhat_u)
            pi_b = pihat_b + c**2 * pihat_a
            mu_b = muhat_b + c * pihat_a / pi_b * (mu_a - muhat_a)
        else:
            # Exact conditioning of the joint Gaussian prior of (x_b, x_a, u).
            var_u = beta**2 / pihat_a + 1 / pi_u
            innovation = observation - muhat_u
            cov_a_u = beta / pihat_a
            cov_b_u = beta * c / pihat_b
            mu_a = muhat_a + cov_a_u / var_u * innovation
            pi_a = 1 / (1 / pihat_a - cov_a_u**2 / var_u)
            mu_b = muhat_b + cov_b_u / var_u * innovation
            pi_b = 1 / (1 / pihat_b - cov_b_u**2 / var_u)

        for node, values in (
            ("b", (mu_b, pi_b, muhat_b, pihat_b)),
            ("a", (mu_a, pi_a, muhat_a, pihat_a)),
            ("u", (observation, pi_u, muhat_u, pi_u)),
        ):
            for field, value in zip(fields, values):
                trajectories[node][field].append(value)

    return {
        node: {field: np.asarray(values) for field, values in node_fields.items()}
        for node, node_fields in trajectories.items()
    }


@pytest.mark.parametrize("cls", [Network, RsNetwork], ids=["jax", "rust"])
@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_value_coupling_with_irregular_time_steps(cls, mean_field_updates):
    r"""Value coupling follows the generative model of [1]_ with irregular time steps.

    The chain of :func:`_value_coupling_chain_reference` runs 60 trials with time steps
    between 0.2 and 3. The time step scales the update of a value parent of a state
    node, but not the prediction of an input. The Rust backend takes unit coupling
    strengths from Python.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    p = dict(VALUE_COUPLING_CHAIN)
    if cls is RsNetwork:
        p.update(alpha=1.0, beta=1.0)
    rng = np.random.default_rng(3)
    time_steps = rng.choice([0.2, 0.5, 1.0, 1.7, 3.0], size=60)
    u = np.cumsum(rng.normal(0.0, 0.5, 60)) + rng.normal(0.0, 0.3, 60)
    reference = _value_coupling_chain_reference(u, time_steps, p, mean_field_updates)

    def coupling(child_idx, strength):
        return child_idx if cls is RsNetwork else ([child_idx], [strength])

    with enable_x64(True):
        net = (
            cls(volatility_updates="standard", mean_field_updates=mean_field_updates)
            .add_nodes(precision=p["input_precision"])
            .add_nodes(value_children=coupling(0, p["beta"]), **p["a"])
            .add_nodes(value_children=coupling(1, p["alpha"]), **p["b"])
        )
        net.input_data(input_data=u, time_steps=time_steps)
        traj = net.node_trajectories
        for node, idx in (("u", 0), ("a", 1), ("b", 2)):
            for field, expected in reference[node].items():
                assert np.allclose(
                    np.asarray(traj[idx][field]), expected, rtol=1e-9, atol=1e-12
                ), f"{node} {field}"
