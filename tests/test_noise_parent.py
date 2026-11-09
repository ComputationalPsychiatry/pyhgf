# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Noise parents of continuous input nodes.

A volatility parent of an input node is a noise parent (see the appendix of [1]_ on the
noise parents of continuous input nodes, Eqs. 64 to 78). The reference
:func:`_weber_filter` transcribes the equations of [1]_ trial by trial, for an input
with a value parent :math:`x` and a noise parent :math:`a`, each with an optional
volatility parent: the prediction step (Eqs. 34 to 37, without drift), the update of a
volatility parent (Eqs. 42 to 45), and the input node with its value and noise parents
(Eqs. 64 to 78).

References
----------
.. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
   Mathys, C. (2026). The generalized hierarchical Gaussian filter.
   doi:10.7554/elife.110174.1
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf.math import gaussian_surprise
from pyhgf.model import Network

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64

VOLATILITY_UPDATES = ["standard", "eHGF", "unbounded"]
FIELDS = ("mean", "precision", "expected_mean", "expected_precision")

# value parent x and noise parent a of the input, with their volatility parents x2
# and a2 (``None`` leaves a parent out)
PARAMETERS = dict(
    x=dict(mu=0.5, pi=1.0, omega=-2.0),
    x2=dict(mu=1.0, pi=1.0, omega=-6.0),
    a=dict(mu=0.0, pi=1.0, omega=-4.0),
    a2=dict(mu=1.0, pi=1.0, omega=-6.0),
    kappa_u=1.4,
    kappa_x=0.7,
    kappa_a=0.8,
    epsilon_u=0.2,
)


def _weber_filter(u, p):
    r"""Filter the observations ``u`` with the equations of [1]_.

    Every node has an autoconnection strength of 1, no drift and a time step of 1.
    Each trial runs:

    #. the prediction of the state nodes (Eqs. 34, 35 and 37), with
       :math:`\Omega = \exp(\omega + \kappa \mu_{\check{a}}^{(k-1)})` and the
       effective precision :math:`\gamma = \Omega \hat{\pi}` (Eq. 43);
    #. the prediction of the input (Eqs. 64 and 66),
       :math:`\hat{\mu}_u = \hat{\mu}_x` and
       :math:`\hat{\pi}_u = 1 / \exp(\kappa_u \mu_a^{(k-1)} + \epsilon_u)`;
    #. the update of the value parent (Eqs. 68 to 70);
    #. the noise prediction error (Eq. 72), from the posterior of the value parent;
    #. the update of the noise parent (Eqs. 74 and 78, with Eq. 75 for
       :math:`\gamma_u = 1` as a check);
    #. the update of the volatility parents of :math:`x` and :math:`a` from their
       volatility prediction errors (Eqs. 42, 44 and 45).

    Returns the trajectories of every node, and of the input under ``"u"``.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    volatility_parent = {
        child: parent
        for child, parent in (("x", "x2"), ("a", "a2"))
        if p.get(parent) is not None
    }
    coupling = {"x": p["kappa_x"], "a": p["kappa_a"]}
    nodes = ["x", "a", *volatility_parent.values()]
    state = {node: (p[node]["mu"], p[node]["pi"]) for node in nodes}
    trajectories = {node: {field: [] for field in FIELDS} for node in nodes}
    trajectories["u"] = {
        "expected_mean": [],
        "expected_precision": [],
        "noise_prediction_error": [],
        "surprise": [],
    }

    for observation in u:
        # Prediction step (Eqs. 34, 35, 37 and 43), from the posteriors at k - 1
        prediction = {}
        for node in nodes:
            log_volatility = p[node]["omega"]
            if node in volatility_parent:
                log_volatility += coupling[node] * state[volatility_parent[node]][0]
            volatility = math.exp(log_volatility)
            pihat = 1 / (1 / state[node][1] + volatility)
            prediction[node] = (state[node][0], pihat, volatility * pihat)

        # Prediction of the input (Eqs. 64 and 66)
        muhat_u = prediction["x"][0]
        pihat_u = 1 / math.exp(p["kappa_u"] * state["a"][0] + p["epsilon_u"])

        # Update of the value parent (Eqs. 68 to 70)
        delta_u = observation - muhat_u
        pi_x = prediction["x"][1] + pihat_u
        mu_x = prediction["x"][0] + pihat_u / pi_x * delta_u

        # Noise prediction error, from the posterior of the value parent (Eq. 72)
        epsilon = pihat_u / pi_x + pihat_u * (observation - mu_x) ** 2 - 1

        # Update of the noise parent (Eqs. 74 and 78), and Eq. 75 with gamma_u = 1
        kappa_u, gamma_u = p["kappa_u"], 1.0
        pi_a = prediction["a"][1] + 1 / 2 * kappa_u**2 * (1 + epsilon)
        pi_a_eq75 = (
            prediction["a"][1]
            + 1 / 2 * (kappa_u * gamma_u) ** 2
            + (kappa_u * gamma_u) ** 2 * epsilon
            - 1 / 2 * kappa_u**2 * gamma_u * epsilon
        )
        assert math.isclose(pi_a, pi_a_eq75, rel_tol=1e-13)
        mu_a = prediction["a"][0] + 1 / 2 * kappa_u / pi_a * epsilon

        posterior = {"x": (mu_x, pi_x), "a": (mu_a, pi_a)}

        # Update of the volatility parents (Eqs. 42, 44 and 45)
        for child, parent in volatility_parent.items():
            mu_c, pi_c = posterior[child]
            muhat_c, pihat_c, gamma_c = prediction[child]
            vope = pihat_c / pi_c + pihat_c * (mu_c - muhat_c) ** 2 - 1
            kappa = coupling[child]
            pi_v = (
                prediction[parent][1]
                + 1 / 2 * (kappa * gamma_c) ** 2
                + (kappa * gamma_c) ** 2 * vope
                - 1 / 2 * kappa**2 * gamma_c * vope
            )
            mu_v = prediction[parent][0] + 1 / 2 * kappa * gamma_c / pi_v * vope
            posterior[parent] = (mu_v, pi_v)

        for node in nodes:
            for field, value in zip(
                FIELDS,
                (*posterior[node], prediction[node][0], prediction[node][1]),
            ):
                trajectories[node][field].append(value)
        trajectories["u"]["expected_mean"].append(muhat_u)
        trajectories["u"]["expected_precision"].append(pihat_u)
        trajectories["u"]["noise_prediction_error"].append(epsilon)
        # Eq. 71: surprise of the input
        trajectories["u"]["surprise"].append(
            1 / 2 * (math.log(2 * math.pi) - math.log(pihat_u) + pihat_u * delta_u**2)
        )
        state = posterior

    return {
        node: {field: np.asarray(values) for field, values in fields.items()}
        for node, fields in trajectories.items()
    }


def _network(cls, p, volatility_updates, mean_field_updates):
    r"""Build the network of :func:`_weber_filter` with pyhgf.

    Node 0 is the input, followed by :math:`x`, its volatility parent :math:`x_2`,
    :math:`a` and its volatility parent :math:`a_2` (when present). The tonic noise
    :math:`\epsilon_u` is a shift of :math:`\epsilon_u / \kappa_u` of the mean of the
    noise parent :math:`a`, whose updates depend on :math:`\kappa_u \mu_a + \epsilon_u`
    only (see :func:`_noise_shift`). Returns the network and the index of each node.
    """

    def coupling(child_idx, strength):
        # the Rust backend takes unit couplings only
        if cls is RsNetwork:
            assert strength == 1.0
            return child_idx
        return ([child_idx], [float(strength)])

    def parameters(node, shift=0.0):
        return dict(
            mean=float(p[node]["mu"] + shift),
            precision=float(p[node]["pi"]),
            tonic_volatility=float(p[node]["omega"]),
        )

    net = cls(
        volatility_updates=volatility_updates, mean_field_updates=mean_field_updates
    )
    net = net.add_nodes()
    idxs = {"x": 1}
    net = net.add_nodes(value_children=0, **parameters("x"))
    if p.get("x2") is not None:
        idxs["x2"] = 2
        net = net.add_nodes(
            volatility_children=coupling(1, p["kappa_x"]), **parameters("x2")
        )
    idxs["a"] = len(idxs) + 1
    net = net.add_nodes(
        volatility_children=coupling(0, p["kappa_u"]),
        **parameters("a", _noise_shift(p)),
    )
    if p.get("a2") is not None:
        idxs["a2"] = len(idxs) + 1
        net = net.add_nodes(
            volatility_children=coupling(idxs["a"], p["kappa_a"]), **parameters("a2")
        )
    return net, idxs


def _noise_shift(p):
    """Shift of the mean of the noise parent that carries the tonic noise."""
    return p["epsilon_u"] / p["kappa_u"]


def _parameters(hierarchy, unit_couplings):
    """Parameters of the reference, with or without the volatility parents."""
    p = dict(PARAMETERS)
    if not hierarchy:
        p.update(x2=None, a2=None)
    if unit_couplings:
        p.update(kappa_u=1.0, kappa_x=1.0, kappa_a=1.0)
    return p


def _inputs(n_trials=300, seed=7):
    """Draws from a Gaussian whose mean and standard deviation jump."""
    rng = np.random.default_rng(seed)
    trials = np.arange(n_trials)
    means = rng.normal(0.0, 3.0, n_trials // 50 + 1)[trials // 50]
    stds = rng.choice([0.5, 1.0, 2.5], n_trials // 75 + 1)[trials // 75]
    return means + stds * rng.normal(size=n_trials)


def _noise_prediction_error(net):
    """Noise prediction errors recorded on the input node, in either backend."""
    trajectory = net.node_trajectories[0]
    if "temp" in trajectory:
        return trajectory["temp"]["volatility_prediction_error"]
    return trajectory["volatility_prediction_error"]


def _assert_matches(net, idxs, reference, p, rtol, atol):
    """Compare the trajectories of a network with the reference."""
    traj = net.node_trajectories
    for field in ("expected_mean", "expected_precision"):
        assert np.allclose(
            traj[0][field], reference["u"][field], rtol=rtol, atol=atol
        ), f"input {field}"
    assert np.allclose(
        _noise_prediction_error(net),
        reference["u"]["noise_prediction_error"],
        rtol=rtol,
        atol=atol,
    ), "noise prediction error"
    for node, idx in idxs.items():
        for field in FIELDS:
            shift = _noise_shift(p) if node == "a" and "mean" in field else 0.0
            assert np.allclose(
                traj[idx][field], reference[node][field] + shift, rtol=rtol, atol=atol
            ), f"{node} {field}"


@pytest.mark.parametrize("cls", [Network, RsNetwork], ids=["jax", "rust"])
@pytest.mark.parametrize("mean_field_updates", [False, True])
@pytest.mark.parametrize("volatility_updates", VOLATILITY_UPDATES)
def test_input_with_value_and_noise_parent_follows_the_equations(
    cls, volatility_updates, mean_field_updates
):
    r"""An input with a value parent and a noise parent follows Eqs. 64 to 78 of [1]_.

    Neither parent has a parent, so the relaxed and the mean-field predictions
    coincide, and the noise parent takes the update of Eqs. 74 and 78 under every
    scheme: all configurations reproduce the reference in float64 over 300 trials,
    with a coupling :math:`\kappa_u = 1.4` in the JAX backend (the Rust backend takes
    unit couplings from Python). The surprise of the input is Eq. 71.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    p = _parameters(hierarchy=False, unit_couplings=cls is RsNetwork)
    u = _inputs()
    reference = _weber_filter(u, p)
    with enable_x64(True):
        net, idxs = _network(cls, p, volatility_updates, mean_field_updates)
        net.input_data(input_data=np.asarray(u))
        _assert_matches(net, idxs, reference, p, rtol=1e-9, atol=1e-12)
        traj = net.node_trajectories
        surprise = gaussian_surprise(
            x=jnp.asarray(traj[0]["mean"]),
            expected_mean=jnp.asarray(traj[0]["expected_mean"]),
            expected_precision=jnp.asarray(traj[0]["expected_precision"]),
        )
        assert np.allclose(surprise, reference["u"]["surprise"], rtol=1e-9)


@pytest.mark.parametrize("cls", [Network, RsNetwork], ids=["jax", "rust"])
def test_hierarchy_with_noise_parent_follows_the_equations(cls):
    r"""Value and noise parents with volatility parents follow the equations of [1]_.

    The value parent :math:`x` and the noise parent :math:`a` have the volatility
    parents :math:`x_2` and :math:`a_2`. Under the standard and mean-field updates,
    every node follows the reference (Eqs. 34 to 78) in float64 over 300 trials, with
    couplings other than 1 in the JAX backend.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    p = _parameters(hierarchy=True, unit_couplings=cls is RsNetwork)
    u = _inputs()
    reference = _weber_filter(u, p)
    with enable_x64(True):
        net, idxs = _network(cls, p, "standard", True)
        net.input_data(input_data=np.asarray(u))
        _assert_matches(net, idxs, reference, p, rtol=1e-9, atol=1e-12)
