# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

VOLATILITY_UPDATES = ["standard", "eHGF", "unbounded"]
MEAN_FIELD_UPDATES = [False, True]


def _build_network(
    cls,
    volatility_updates,
    mean_field_updates,
    n_levels,
    timeseries,
    observed=None,
    input_precision=1.0,
):
    """Build an ``n_levels`` continuous HGF with the given update scheme."""
    net = cls(
        volatility_updates=volatility_updates,
        mean_field_updates=mean_field_updates,
    )
    net = net.add_nodes(precision=input_precision).add_nodes(value_children=0)
    if n_levels == 3:
        net = net.add_nodes(volatility_children=0)
    return net.input_data(input_data=timeseries, observed=observed)


def _assert_backends_match(
    py_net, rs_net, n_levels, volatility_updates, label, atol=0.0
):
    """Assert the JAX and Rust trajectories agree for every node.

    JAX defaults to float32, Rust to f64, so this is a float32 tolerance (under
    ``jax_enable_x64`` they agree to ~1e-14). The unbounded kernel differs at the float
    level between backends, so the tolerance is loosened with a volatility parent.
    ``atol`` adds an absolute floor for trajectories that pass close to zero.
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
                atol=atol,
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
def test_continuous_input_precision(n_levels, volatility_updates, mean_field_updates):
    r"""Both backends weight the observation by the precision of the input node.

    An input node without a volatility parent predicts its own precision,
    :math:`\hat{\pi}_0 = \pi_0`, which both the relaxed and the mean-field updates of
    its value parent use as the observation precision. With three levels, the input
    node has a volatility parent and predicts :math:`1 / (1/\pi_0 + \Omega_0)`.
    """
    timeseries = load_data("continuous")
    label = f"n_levels={n_levels} vol={volatility_updates} mf={mean_field_updates}"

    py_net, rs_net = (
        _build_network(
            cls,
            volatility_updates,
            mean_field_updates,
            n_levels,
            timeseries,
            input_precision=1e4,
        )
        for cls in (PyNetwork, RsNetwork)
    )

    if n_levels == 2:
        input_expected_precision = rs_net.node_trajectories[0]["expected_precision"]
        assert np.all(np.asarray(input_expected_precision) == 1e4)
    _assert_backends_match(py_net, rs_net, n_levels, volatility_updates, label)


@pytest.mark.parametrize(
    "volatility_parent_kwargs",
    [{"tonic_drift": 0.5}, {"autoconnection_strength": 0.5, "mean": -2.0}],
    ids=["drift", "autoconnection"],
)
@pytest.mark.parametrize("volatility_child", [0, 1], ids=["input", "state"])
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_moving_volatility_parent(
    volatility_parent_kwargs, volatility_child, mean_field_updates
):
    r"""Both backends predict the volatility from the parent's predicted mean.

    With a tonic drift or an autoconnection strength other than 1, the predicted mean
    :math:`\hat{\mu}_{\check{a}}^{(k)}` of the volatility parent differs from its
    previous posterior mean :math:`\mu_{\check{a}}^{(k-1)}`. The child's predicted
    volatility uses :math:`\hat{\mu}_{\check{a}}^{(k)}`, so the JAX and Rust
    trajectories must agree.
    """
    timeseries = load_data("continuous")[:200]
    label = f"{volatility_parent_kwargs} child={volatility_child}"

    py_net, rs_net = (
        cls(volatility_updates="standard", mean_field_updates=mean_field_updates)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=volatility_child, **volatility_parent_kwargs)
        .input_data(input_data=timeseries)
        for cls in (PyNetwork, RsNetwork)
    )

    volatility_parent = py_net.node_trajectories[2]
    previous_mean = np.concatenate([[0.0], volatility_parent["mean"][:-1]])
    assert np.max(np.abs(volatility_parent["expected_mean"] - previous_mean)) > 0.1
    _assert_backends_match(py_net, rs_net, 3, "standard", label)


@pytest.mark.parametrize("cls", [PyNetwork, RsNetwork], ids=["jax", "rust"])
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_ehgf_precision_increment_with_two_volatility_parents(cls, mean_field_updates):
    r"""The eHGF precision update re-predicts from every volatility parent of the child.

    Node 1 has two volatility parents, nodes 2 and 3. The eHGF update of node 2
    re-predicts the volatility of node 1 at the posterior mean :math:`\mu_2^{(k)}`,
    with node 3 at its predicted mean :math:`\hat{\mu}_3^{(k)}`,

    .. math::

        \nu = t \exp(\omega_1 + \kappa_2 \mu_2^{(k)} + \kappa_3 \hat{\mu}_3^{(k)}),

    and adds :math:`\max(0, \tfrac12 \kappa_2^2 \gamma (\gamma + w \Delta))` to the
    predicted precision, with :math:`\gamma`, :math:`w` and :math:`\Delta` computed
    from :math:`\nu` and the carried variance :math:`1/\pi_1^{(k-1)}`. The increment
    is recomputed here from the recorded trajectories.
    """
    tonic_volatility = -6.0
    net = (
        cls(volatility_updates="eHGF", mean_field_updates=mean_field_updates)
        .add_nodes(precision=1e4)
        .add_nodes(value_children=0, tonic_volatility=tonic_volatility)
        .add_nodes(volatility_children=1, tonic_volatility=-2.0)
        .add_nodes(volatility_children=1, mean=2.0, tonic_volatility=-2.0)
        .input_data(input_data=load_data("continuous")[:200])
    )
    child, parent, other_parent = (
        {
            key: np.asarray(net.node_trajectories[node_idx][key], dtype=np.float64)
            for key in ("mean", "precision", "expected_mean", "expected_precision")
        }
        for node_idx in (1, 2, 3)
    )

    # Step 0 is left out: its carried variance comes from the initial precision.
    carried_variance = 1.0 / child["precision"][:-1]
    volatility = np.exp(
        tonic_volatility + parent["mean"][1:] + other_parent["expected_mean"][1:]
    )
    expected_precision = 1.0 / (carried_variance + volatility)
    effective_precision = volatility * expected_precision
    volatility_error_weight = (volatility - carried_variance) * expected_precision
    volatility_prediction_error = (
        1.0 / child["precision"][1:]
        + (child["mean"][1:] - child["expected_mean"][1:]) ** 2
    ) * expected_precision - 1.0
    increment = np.maximum(
        0.0,
        0.5
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error),
    )

    assert np.max(np.abs(other_parent["expected_mean"])) > 1.0
    assert np.allclose(
        parent["precision"][1:] - parent["expected_precision"][1:],
        increment,
        rtol=1e-3,
        atol=1e-5,
    )


@pytest.mark.parametrize("cls", [PyNetwork, RsNetwork], ids=["jax", "rust"])
@pytest.mark.parametrize("volatility_updates", VOLATILITY_UPDATES)
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_pinned_second_volatility_parent(cls, volatility_updates, mean_field_updates):
    r"""A pinned second volatility parent shifts the child's tonic volatility.

    Node 1 has two volatility parents: node 2, and node 3 with mean :math:`m`,
    precision :math:`10^8` and tonic volatility :math:`-40`, which keeps its belief at
    :math:`m`. Node 3 then enters the volatility of node 1 as the constant
    :math:`\kappa m`, so the network is equivalent to the same network without node 3
    and with the tonic volatility :math:`\omega_1 + \kappa m` at node 1. In every update
    scheme, node 2 receives the full volatility prediction error of node 1.
    """
    timeseries = load_data("continuous")[:200]
    tonic_volatility, pinned_mean = -6.0, 2.0
    label = f"vol={volatility_updates} mf={mean_field_updates}"

    def network(node_1_tonic_volatility):
        return (
            cls(
                volatility_updates=volatility_updates,
                mean_field_updates=mean_field_updates,
            )
            .add_nodes()
            .add_nodes(value_children=0, tonic_volatility=node_1_tonic_volatility)
            .add_nodes(volatility_children=1, tonic_volatility=-2.0)
        )

    two_parents = (
        network(tonic_volatility)
        .add_nodes(
            volatility_children=1,
            mean=pinned_mean,
            precision=1e8,
            tonic_volatility=-40.0,
        )
        .input_data(input_data=timeseries)
    )
    one_parent = network(tonic_volatility + pinned_mean).input_data(
        input_data=timeseries
    )

    assert np.allclose(two_parents.node_trajectories[3]["mean"], pinned_mean)
    assert np.ptp(np.asarray(one_parent.node_trajectories[2]["mean"])) > 0.1
    for node_idx in range(3):
        for key in ["mean", "expected_mean", "precision", "expected_precision"]:
            assert np.allclose(
                two_parents.node_trajectories[node_idx][key],
                one_parent.node_trajectories[node_idx][key],
                rtol=1e-4,
                atol=1e-6,
            ), f"{label}: node {node_idx}, key '{key}' mismatch"


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


@pytest.mark.parametrize("volatility_updates", VOLATILITY_UPDATES)
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_missing_observations_keep_predictions(
    volatility_updates, mean_field_updates
):
    r"""Without observations, every node above the input keeps its prediction.

    Node 1 is the value parent of the input, with a value parent (node 2) and a
    volatility parent (node 3) of its own. While the input is missing, no node
    receives evidence, so each posterior equals the prediction:
    :math:`\pi^{(k)} = \hat{\pi}^{(k)}` and :math:`\mu^{(k)} = \hat{\mu}^{(k)}`. Node 1
    counts as unobserved, so nodes 2 and 3 receive no prediction error from it.
    """
    timeseries = load_data("continuous")[:300]
    observed = np.ones(len(timeseries), dtype=int)
    observed[200:250] = 0
    label = f"vol={volatility_updates} mf={mean_field_updates}"

    py_net, rs_net = (
        cls(
            volatility_updates=volatility_updates, mean_field_updates=mean_field_updates
        )
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries, observed=observed)
        for cls in (PyNetwork, RsNetwork)
    )

    # The mean of node 2 passes close to zero, so the float32 comparison needs an
    # absolute floor on top of the relative tolerance.
    _assert_backends_match(py_net, rs_net, 4, volatility_updates, label, atol=1e-6)
    for net in (py_net, rs_net):
        for node_idx in (1, 2, 3):
            node = net.node_trajectories[node_idx]
            for key in ("precision", "mean"):
                assert np.allclose(
                    np.asarray(node[key])[200:250],
                    np.asarray(node[f"expected_{key}"])[200:250],
                    rtol=1e-6,
                    atol=1e-9,
                ), f"{label}: node {node_idx}, '{key}' moved without observations"


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
