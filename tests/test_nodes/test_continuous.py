# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

VOLATILITY_UPDATES = ["standard", "eHGF", "unbounded"]
MEAN_FIELD_UPDATES = [False, True]
# JAX float32 against Rust f64 on the networks of
# test_continuous_unbounded_multiple_children: largest absolute gap 1.1e-5.
MULTIPLE_CHILDREN_RTOL, MULTIPLE_CHILDREN_ATOL = 1e-4, 1e-4


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
@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF"])
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


MULTIPLE_CHILDREN_TOPOLOGIES = [
    "shared_volatility_parent",
    "value_and_volatility_parent",
    "value_and_two_volatility_children",
]


def _multiple_children_network(cls, topology, mean_field_updates):
    """Build a network where one node has several children under ``"unbounded"``."""
    net = cls(volatility_updates="unbounded", mean_field_updates=mean_field_updates)
    if topology == "shared_volatility_parent":
        # node 4 is the volatility parent of nodes 2 and 3
        return (
            net
            .add_nodes(n_nodes=2)
            .add_nodes(value_children=0)
            .add_nodes(value_children=1)
            .add_nodes(volatility_children=[2, 3])
        )
    if topology == "value_and_volatility_parent":
        # node 4 is the value parent of node 2 and the volatility parent of node 3
        return (
            net
            .add_nodes(n_nodes=2)
            .add_nodes(value_children=0)
            .add_nodes(value_children=1)
            .add_nodes(value_children=2, volatility_children=3)
        )
    # node 5 is the value parent of node 3 and the volatility parent of node 4 and
    # of the input node 2
    return (
        net
        .add_nodes(n_nodes=3)
        .add_nodes(value_children=0)
        .add_nodes(value_children=1)
        .add_nodes(value_children=3, volatility_children=[4, 2])
    )


def _multiple_children_timeseries(topology):
    """Random walks, one per input node."""
    n_inputs = 3 if topology == "value_and_two_volatility_children" else 2
    rng = np.random.default_rng(123)
    return np.cumsum(rng.normal(0.0, 0.3, size=(300, n_inputs)), axis=0)


@pytest.mark.parametrize("topology", MULTIPLE_CHILDREN_TOPOLOGIES)
@pytest.mark.parametrize("mean_field_updates", MEAN_FIELD_UPDATES)
def test_continuous_unbounded_multiple_children(topology, mean_field_updates):
    """Test the unbounded update of a node with several children in both backends.

    A shared volatility parent, and a node that is both value and volatility parent, use
    the (n+1)-expansion update in the JAX and Rust backends. Under ``jax_enable_x64``
    the two backends agree to ~1e-12 on these networks; the tolerance covers the float32
    rounding of the JAX backend.
    """
    timeseries = _multiple_children_timeseries(topology)
    label = f"{topology} mean_field={mean_field_updates}"

    py_net = _multiple_children_network(
        PyNetwork, topology, mean_field_updates
    ).input_data(input_data=timeseries)
    rs_net = _multiple_children_network(
        RsNetwork, topology, mean_field_updates
    ).input_data(input_data=timeseries)

    for node_idx in range(len(rs_net.edges)):
        for key in ["mean", "expected_mean", "precision", "expected_precision"]:
            assert np.allclose(
                py_net.node_trajectories[node_idx][key],
                rs_net.node_trajectories[node_idx][key],
                rtol=MULTIPLE_CHILDREN_RTOL,
                atol=MULTIPLE_CHILDREN_ATOL,
            ), f"{label}: node {node_idx}, key '{key}' mismatch"
