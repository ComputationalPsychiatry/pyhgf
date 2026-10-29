# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

VOLATILITY_UPDATES = ["standard", "eHGF", "unbounded"]
MEAN_FIELD_UPDATES = [False, True]


def _build_network(
    cls, volatility_updates, mean_field_updates, n_levels, timeseries, observed=None
):
    """Build an ``n_levels`` continuous HGF with the given update scheme."""
    net = cls(
        volatility_updates=volatility_updates,
        mean_field_updates=mean_field_updates,
    )
    net = net.add_nodes().add_nodes(value_children=0)
    if n_levels == 3:
        net = net.add_nodes(volatility_children=0)
    return net.input_data(input_data=timeseries, observed=observed)


def _assert_backends_match(py_net, rs_net, n_levels, volatility_updates, label):
    """Assert the JAX and Rust trajectories agree for every node.

    JAX defaults to float32, Rust to f64, so this is a float32 tolerance (under
    ``jax_enable_x64`` they agree to ~1e-14). The unbounded kernel differs at the float
    level between backends, so the tolerance is loosened with a volatility parent.
    """
    # The unbounded posterior kernel differs at the float level between the two
    # backends (mathematically equivalent but distinct implementations), so the
    # cross-backend comparison is loosened when a volatility parent is present.
    if volatility_updates == "unbounded" and n_levels == 3:
        rtol = 1e-1
    else:
        rtol = 1e-4
    for node_idx in range(n_levels):
        for key in ["mean", "expected_mean", "precision", "expected_precision"]:
            assert np.allclose(
                py_net.node_trajectories[node_idx][key],
                rs_net.node_trajectories[node_idx][key],
                rtol=rtol,
            ), f"{label}: node {node_idx}, key '{key}' mismatch"


@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_2_levels(mean_field_updates):
    """Test the 2-level continuous HGF: input node → value parent.

    The value parent has no volatility children, so ``volatility_updates`` does not
    affect its update path and is not parametrised here. The JAX and Rust backends must
    produce identical trajectories for both values of ``mean_field_updates``.
    """
    timeseries = load_data("continuous")
    label = f"mean_field={mean_field_updates}"

    py_net = _build_network(PyNetwork, "standard", mean_field_updates, 2, timeseries)
    rs_net = _build_network(RsNetwork, "standard", mean_field_updates, 2, timeseries)

    _assert_backends_match(py_net, rs_net, 2, "standard", label)


@pytest.mark.parametrize("volatility_updates", VOLATILITY_UPDATES)
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_3_levels(volatility_updates, mean_field_updates):
    """Test the 3-level continuous HGF: input → value parent + volatility parent.

    The JAX and Rust backends must produce identical trajectories for every combination
    of ``volatility_updates`` and ``mean_field_updates``.
    """
    timeseries = load_data("continuous")
    label = f"vol={volatility_updates} mean_field={mean_field_updates}"

    py_net = _build_network(
        PyNetwork, volatility_updates, mean_field_updates, 3, timeseries
    )
    rs_net = _build_network(
        RsNetwork, volatility_updates, mean_field_updates, 3, timeseries
    )

    _assert_backends_match(py_net, rs_net, 3, volatility_updates, label)


@pytest.mark.parametrize("n_levels", [2, 3])
@pytest.mark.parametrize("volatility_updates", VOLATILITY_UPDATES)
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_missing_observations(
    n_levels, volatility_updates, mean_field_updates
):
    """Both backends handle missing inputs in the same way.

    While the input is missing, its value parent ignores the placeholder values and ages
    its precision.
    """
    timeseries = load_data("continuous")
    observed = np.ones(len(timeseries), dtype=int)
    observed[200:300] = 0
    label = f"n_levels={n_levels} vol={volatility_updates} mf={mean_field_updates}"

    py_net, rs_net = (
        _build_network(
            cls, volatility_updates, mean_field_updates, n_levels, timeseries, observed
        )
        for cls in (PyNetwork, RsNetwork)
    )

    _assert_backends_match(py_net, rs_net, n_levels, volatility_updates, label)
    assert np.all(np.diff(rs_net.node_trajectories[1]["precision"][200:300]) < 0.0)


@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_nonlinear_coupling(mean_field_updates):
    """Test a 2-level continuous HGF with a non-linear (tanh) value coupling.

    Exercises the non-linear coupling branches of the posterior precision and mean
    updates (the ``grad`` / ``grad(grad(...))`` terms), for both the relaxed and the
    mean-field schemes. The JAX (``jnp.tanh``) and Rust (``"tanh"``) backends must
    agree.
    """
    import jax.numpy as jnp

    timeseries = load_data("continuous")
    label = f"nonlinear mean_field={mean_field_updates}"

    py_net = (
        PyNetwork(volatility_updates="standard", mean_field_updates=mean_field_updates)
        .add_nodes()
        .add_nodes(value_children=0, coupling_fn=(jnp.tanh,))
        .input_data(input_data=timeseries)
    )
    rs_net = (
        RsNetwork(volatility_updates="standard", mean_field_updates=mean_field_updates)
        .add_nodes()
        .add_nodes(value_children=0, coupling_fn="tanh")
        .input_data(input_data=timeseries)
    )

    _assert_backends_match(py_net, rs_net, 2, "standard", label)


def test_relaxed_precision_curved_value_coupling():
    r"""Test the relaxed posterior precision of a value parent with a tanh coupling.

    In the chain input ← node 1 ← node 2 (tanh coupling), node 1 is a Gaussian
    interior child, so the posterior precision of node 2 follows the posterior-step
    relaxation

    .. math::

        \pi_b = \hat{\pi}_b + \frac{\bar{\pi}_a}{\bar{\pi}_a + \pi_y}
            \left( g'^2 \, \pi_y - g'' \, \pi_a \, \delta_a \right), \qquad
        \pi_y = \pi_a - \hat{\pi}_a,

    with :math:`g = \tanh` differentiated at the previous posterior mean of node 2.
    Both backends must match this closed form, computed from the JAX trajectories.
    """
    import jax
    import jax.numpy as jnp

    timeseries = load_data("continuous")[:50]
    py_net = (
        PyNetwork(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(value_children=1, coupling_fn=(jnp.tanh,))
        .input_data(input_data=timeseries)
    )
    rs_net = (
        RsNetwork(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(value_children=1, coupling_fn="tanh")
        .input_data(input_data=timeseries)
    )
    # The mean of node 2 passes close to zero, so the float32 comparison needs an
    # absolute floor on top of the relative tolerance.
    for node_idx in range(3):
        for key in ["mean", "expected_mean", "precision", "expected_precision"]:
            assert np.allclose(
                py_net.node_trajectories[node_idx][key],
                rs_net.node_trajectories[node_idx][key],
                rtol=1e-4,
                atol=1e-6,
            ), f"node {node_idx}, key '{key}' mismatch"

    child, parent = py_net.node_trajectories[1], py_net.node_trajectories[2]
    pi_a = np.asarray(child["precision"])
    pi_y = pi_a - np.asarray(child["expected_precision"])
    pi_bar_a = np.asarray(child["temp"]["conditional_expected_precision"])
    delta_a = np.asarray(child["temp"]["value_prediction_error"])

    # Previous posterior mean of node 2, starting from its prior mean of 0.
    linearisation_point = jnp.concatenate([jnp.zeros(1), parent["mean"][:-1]])
    g_prime = np.asarray(jax.vmap(jax.grad(jnp.tanh))(linearisation_point))
    g_second = np.asarray(jax.vmap(jax.grad(jax.grad(jnp.tanh)))(linearisation_point))

    expected_precision = np.asarray(parent["expected_precision"]) + pi_bar_a / (
        pi_bar_a + pi_y
    ) * (g_prime**2 * pi_y - g_second * pi_a * delta_a)
    for net in (py_net, rs_net):
        assert np.allclose(
            net.node_trajectories[2]["precision"], expected_precision, rtol=1e-5
        )


@pytest.mark.parametrize("coupling", ["linear", "tanh"])
def test_relaxed_update_observed_leaf(coupling):
    r"""Test the relaxed update of a value parent from an observed leaf.

    The leaf (node 0) has a value parent (node 1) and a volatility parent (node 2).
    It is observed exactly, so node 1 takes the fully observed limit
    :math:`\pi_y \to \infty` of the posterior-step relaxation, in which both
    child-precision factors are the leaf's conditional predicted precision
    :math:`\bar{\pi}_0`:

    .. math::

        \pi_1 = \hat{\pi}_1 + \bar{\pi}_0 \left( g'^2 - g'' \, \delta_0 \right),
            \qquad
        \mu_1 = \hat{\mu}_1 + \frac{g' \, \bar{\pi}_0}{\pi_1} \, \delta_0,

    with :math:`g` differentiated at the previous posterior mean of node 1. The
    leaf's marginal predicted precision :math:`\hat{\pi}_0` also contains node 1's
    own variance :math:`g'^2 / \hat{\pi}_1`, which does not enter this update. Both
    backends must match this closed form, computed from the JAX trajectories.
    """
    import jax
    import jax.numpy as jnp

    if coupling == "tanh":
        py_kwargs, rs_kwargs, coupling_fn = (
            {"coupling_fn": (jnp.tanh,)},
            {"coupling_fn": "tanh"},
            jnp.tanh,
        )
    else:
        py_kwargs, rs_kwargs, coupling_fn = {}, {}, lambda x: x

    timeseries = load_data("continuous")[:50]
    py_net, rs_net = (
        cls(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0, **kwargs)
        .add_nodes(volatility_children=0)
        .input_data(input_data=timeseries)
        for cls, kwargs in ((PyNetwork, py_kwargs), (RsNetwork, rs_kwargs))
    )
    _assert_backends_match(py_net, rs_net, 3, "standard", f"leaf {coupling}")

    leaf, parent = py_net.node_trajectories[0], py_net.node_trajectories[1]
    pi_bar_0 = np.asarray(leaf["temp"]["conditional_expected_precision"])
    delta_0 = np.asarray(leaf["temp"]["value_prediction_error"])

    # The parent's variance makes the marginal differ from the conditional, so the
    # closed form below separates the two.
    assert np.max(pi_bar_0 / np.asarray(leaf["expected_precision"])) > 1.05

    # Previous posterior mean of node 1, starting from its prior mean of 0.
    linearisation_point = jnp.concatenate([jnp.zeros(1), parent["mean"][:-1]])
    g_prime = np.asarray(jax.vmap(jax.grad(coupling_fn))(linearisation_point))
    g_second = np.asarray(
        jax.vmap(jax.grad(jax.grad(coupling_fn)))(linearisation_point)
    )

    expected_precision = np.asarray(parent["expected_precision"]) + pi_bar_0 * (
        g_prime**2 - g_second * delta_0
    )
    expected_mean = (
        np.asarray(parent["expected_mean"])
        + g_prime * pi_bar_0 * delta_0 / expected_precision
    )
    for net in (py_net, rs_net):
        assert np.allclose(
            net.node_trajectories[1]["precision"], expected_precision, rtol=1e-5
        )
        assert np.allclose(
            net.node_trajectories[1]["mean"], expected_mean, rtol=1e-5, atol=1e-6
        )
