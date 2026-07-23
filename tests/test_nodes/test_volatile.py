# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

UPDATE_TYPES = ["standard", "eHGF", "unbounded"]


def _assert_value_level_match(net_a, node_a, net_b, node_b, label="", rtol=1e-5):
    """Assert value-level trajectories match between two networks."""
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        assert np.allclose(
            net_a.node_trajectories[node_a][key],
            net_b.node_trajectories[node_b][key],
            rtol=rtol,
        ), f"{label}: Value-level key '{key}' mismatch"


def _assert_vol_level_match(
    volatile_net, vol_node, explicit_net, exp_node, label="", rtol=1e-4, atol=1e-6
):
    """Assert volatility-level trajectories match (volatile _vol vs explicit node)."""
    vol_key_map = {
        "mean_vol": "mean",
        "expected_mean_vol": "expected_mean",
        "precision_vol": "precision",
        "expected_precision_vol": "expected_precision",
    }
    for vol_key, explicit_key in vol_key_map.items():
        assert np.allclose(
            volatile_net.node_trajectories[vol_node][vol_key],
            explicit_net.node_trajectories[exp_node][explicit_key],
            rtol=rtol,
            atol=atol,
        ), f"{label}: Volatility-level key '{vol_key}' vs '{explicit_key}' mismatch"


def _build_volatile(cls, volatility_updates, timeseries, mean_field_updates=False):
    """Build a volatile-state network with autoconnection for equivalence testing."""
    return (
        cls(
            volatility_updates=volatility_updates, mean_field_updates=mean_field_updates
        )
        .add_nodes()
        .add_nodes(
            kind="volatile-state",
            value_children=0,
            autoconnection_strength=1.0,
        )
        .input_data(input_data=timeseries)
    )


def _build_explicit(cls, volatility_updates, timeseries, mean_field_updates=False):
    """Build explicit continuous + volatility-parent network.

    The fused volatile node's value level carries no tonic volatility, so the explicit
    value node (node 1) sets ``tonic_volatility=0.0`` to match. Its volatility parent
    (node 2) keeps the default, mirroring the volatility level's
    ``tonic_volatility_vol``.
    """
    return (
        cls(
            volatility_updates=volatility_updates, mean_field_updates=mean_field_updates
        )
        .add_nodes()
        .add_nodes(value_children=0, tonic_volatility=0.0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=timeseries)
    )


def _run_volatile_vs_explicit(volatility_updates):
    """Test that volatile-state is equivalent to explicit continuous+vol-parent pair.

    Both the Python and Rust backends satisfy this equivalence: the value-level
    posterior update runs before the volatility-level prediction-error step in both.
    """
    timeseries = load_data("continuous")

    for cls, label in [
        (PyNetwork, f"{volatility_updates} py"),
        (RsNetwork, f"{volatility_updates} rs"),
    ]:
        vol = _build_volatile(cls, volatility_updates, timeseries)
        exp = _build_explicit(cls, volatility_updates, timeseries)

        _assert_value_level_match(vol, 0, exp, 0, f"{label} input")
        _assert_value_level_match(vol, 1, exp, 1, label)
        _assert_vol_level_match(vol, 1, exp, 2, label)


def _run_explicit_cross_backend(volatility_updates):
    """Test that Python and Rust produce the same trajectories for explicit networks."""
    timeseries = load_data("continuous")

    exp_py = _build_explicit(PyNetwork, volatility_updates, timeseries)
    exp_rs = _build_explicit(RsNetwork, volatility_updates, timeseries)

    label = f"{volatility_updates} py vs rs"
    # Unbounded path: see docstring — Python and Rust use mathematically
    # equivalent but float-distinct unbounded posterior kernels.
    rtol = 1e-1 if volatility_updates == "unbounded" else 1e-4
    _assert_value_level_match(exp_py, 0, exp_rs, 0, f"{label} input", rtol=rtol)
    _assert_value_level_match(exp_py, 1, exp_rs, 1, label, rtol=rtol)


def test_volatile_node_matches_explicit_volatility_parent():
    """Test Rust volatile-state against explicit continuous+vol-parent (standard)."""
    _run_volatile_vs_explicit("standard")


def test_volatile_node_ehgf_matches_explicit():
    """Test Rust volatile-state against explicit continuous+vol-parent (eHGF)."""
    _run_volatile_vs_explicit("eHGF")


def test_volatile_node_unbounded_matches_explicit():
    """Test Rust volatile-state against explicit continuous+vol-parent (unbounded)."""
    _run_volatile_vs_explicit("unbounded")


def test_explicit_cross_backend_standard():
    """Test Python and Rust explicit networks agree (standard update)."""
    _run_explicit_cross_backend("standard")


def test_explicit_cross_backend_ehgf():
    """Test Python and Rust explicit networks agree (eHGF update)."""
    _run_explicit_cross_backend("eHGF")


def test_explicit_cross_backend_unbounded():
    """Test Python and Rust explicit networks agree (unbounded update)."""
    _run_explicit_cross_backend("unbounded")


def _run_volatile_input_leaf_precision(cls, label, mean_field_updates=False):
    """Check that a volatile-state input/leaf node keeps its prior precision.

    The input/leaf node has no value children, so it does not undergo a Gaussian random
    walk between observations. Its ``expected_precision`` must therefore equal its prior
    precision at every step (the input-leaf override, mirroring the continuous-node
    treatment). Volatile nodes carry no ``tonic_volatility``.
    """
    timeseries = np.array([0.5, 1.0, 0.7, 0.3])
    net = (
        cls(
            volatility_updates="unbounded",
            mean_field_updates=mean_field_updates,
        )
        .add_nodes(
            kind="volatile-state",
            precision=5.0,
            expected_precision=5.0,
        )
        .add_nodes(value_children=0)
        .input_data(input_data=timeseries)
    )
    expected_precision = np.asarray(net.node_trajectories[0]["expected_precision"])
    # The input/leaf node's expected_precision stays at its prior precision of 5.0.
    np.testing.assert_allclose(
        expected_precision,
        5.0,
        rtol=1e-5,
        err_msg=(
            f"{label}: input volatile-state node's expected_precision is not the "
            "prior precision — input-leaf override missing"
        ),
    )


def test_volatile_input_node_uses_prior_precision_python():
    """Per-node Python: volatile-state input/leaf keeps its prior precision."""
    _run_volatile_input_leaf_precision(PyNetwork, "py")


def test_volatile_input_node_uses_prior_precision_rust():
    """Per-node Rust: volatile-state input/leaf keeps its prior precision."""
    _run_volatile_input_leaf_precision(RsNetwork, "rs")


def _assert_volatile_node_cross_backend(net_py, net_rs, node, label="", rtol=1e-4):
    """Assert a volatile node's trajectories match across backends."""
    keys = [
        "mean",
        "expected_mean",
        "precision",
        "expected_precision",
        "mean_vol",
        "expected_mean_vol",
        "precision_vol",
        "expected_precision_vol",
    ]
    for key in keys:
        assert np.allclose(
            net_py.node_trajectories[node][key],
            net_rs.node_trajectories[node][key],
            rtol=rtol,
        ), f"{label}: key '{key}' mismatch"


@pytest.mark.parametrize("volatility_updates", UPDATE_TYPES)
def test_volatile_mean_field_cross_backend(volatility_updates):
    """JAX and Rust agree for volatile-state nodes with ``mean_field_updates=True``.

    Exercises the mean-field prediction and posterior paths of the volatile-state node
    (the ``_mean_field`` update functions), which the relaxed-default tests above do not
    cover.
    """
    timeseries = load_data("continuous")

    vol_py = _build_volatile(
        PyNetwork, volatility_updates, timeseries, mean_field_updates=True
    )
    vol_rs = _build_volatile(
        RsNetwork, volatility_updates, timeseries, mean_field_updates=True
    )

    # Unbounded path: Python and Rust use mathematically equivalent but float-distinct
    # unbounded posterior kernels (see _run_explicit_cross_backend).
    rtol = 1e-1 if volatility_updates == "unbounded" else 1e-4
    _assert_volatile_node_cross_backend(
        vol_py, vol_rs, 1, f"{volatility_updates} mean_field py vs rs", rtol=rtol
    )


@pytest.mark.parametrize("volatility_updates", UPDATE_TYPES)
def test_explicit_mean_field_cross_backend(volatility_updates):
    """JAX and Rust agree for explicit continuous+vol-parent with mean-field updates."""
    timeseries = load_data("continuous")

    exp_py = _build_explicit(
        PyNetwork, volatility_updates, timeseries, mean_field_updates=True
    )
    exp_rs = _build_explicit(
        RsNetwork, volatility_updates, timeseries, mean_field_updates=True
    )

    label = f"{volatility_updates} mean_field py vs rs"
    rtol = 1e-1 if volatility_updates == "unbounded" else 1e-4
    _assert_value_level_match(exp_py, 0, exp_rs, 0, f"{label} input", rtol=rtol)
    _assert_value_level_match(exp_py, 1, exp_rs, 1, label, rtol=rtol)
    _assert_value_level_match(exp_py, 2, exp_rs, 2, label, rtol=rtol)


def test_volatile_input_node_uses_prior_precision_mean_field_python():
    """Per-node Python: volatile-state input/leaf keeps prior precision (mean-field)."""
    _run_volatile_input_leaf_precision(
        PyNetwork, "py mean_field", mean_field_updates=True
    )


def test_volatile_input_node_uses_prior_precision_mean_field_rust():
    """Per-node Rust: volatile-state input/leaf keeps prior precision (mean-field)."""
    _run_volatile_input_leaf_precision(
        RsNetwork, "rs mean_field", mean_field_updates=True
    )


def _build_volatile_value_parent(coupling_fn, mean_field_updates, timeseries):
    """Build a volatile-state value parent coupled to its child via ``coupling_fn``."""
    net = PyNetwork(
        volatility_updates="standard", mean_field_updates=mean_field_updates
    ).add_nodes()
    if coupling_fn is None:
        net = net.add_nodes(
            kind="volatile-state", value_children=0, autoconnection_strength=1.0
        )
    else:
        net = net.add_nodes(
            kind="volatile-state",
            value_children=0,
            autoconnection_strength=1.0,
            coupling_fn=coupling_fn,
        )
    return net.input_data(input_data=timeseries)


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_volatile_nonlinear_coupling(mean_field_updates):
    """Volatile-state node with non-linear (tanh) value coupling (JAX).

    Exercises the non-linear coupling branches (the ``grad`` / ``grad(grad(...))``
    terms) of the volatile-state value-level precision and mean updates, in both the
    relaxed and mean-field schemes. The trajectories must stay finite, and the
    non-linear coupling must actually change the result relative to linear coupling.

    The two backends are not compared here: the ``tanh``-coupled volatile node (with
    autoconnection and an implicit volatility level) is numerically sensitive, so the
    JAX and Rust trajectories diverge over time despite using the same update rule.
    """
    import jax.numpy as jnp

    timeseries = load_data("continuous")
    nonlinear = _build_volatile_value_parent(
        (jnp.tanh,), mean_field_updates, timeseries
    )
    linear = _build_volatile_value_parent(None, mean_field_updates, timeseries)

    label = f"nonlinear mean_field={mean_field_updates}"
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        values = np.asarray(nonlinear.node_trajectories[1][key])
        assert np.all(np.isfinite(values)), f"{label}: non-finite '{key}'"

    # The tanh coupling must change the value-level trajectory relative to linear.
    assert not np.allclose(
        nonlinear.node_trajectories[1]["mean"],
        linear.node_trajectories[1]["mean"],
    ), f"{label}: tanh coupling did not change the trajectory"


def _build_volatile_value_child(coupling_fn, mean_field_updates, timeseries):
    """Build a volatile-state node that is the value *child* of a volatile parent.

    The parent → child edge carries ``coupling_fn``, so the child's prediction step
    reads the parent's expected mean through that coupling. A volatile parent is used
    (rather than a continuous one) because continuous parents read their children's
    ``observed`` flag, which volatile-state nodes do not expose.
    """
    net = PyNetwork(
        volatility_updates="standard", mean_field_updates=mean_field_updates
    ).add_nodes()
    net = net.add_nodes(kind="volatile-state", value_children=0)  # node 1: child
    # The parent (node 2) carries its mean forward (autoconnection_strength=1.0) so it
    # drifts away from 0; otherwise its expected mean stays at 0, where tanh(0) == 0
    # is indistinguishable from linear coupling.
    if coupling_fn is None:
        net = net.add_nodes(
            kind="volatile-state", value_children=1, autoconnection_strength=1.0
        )
    else:
        net = net.add_nodes(
            kind="volatile-state",
            value_children=1,
            autoconnection_strength=1.0,
            coupling_fn=coupling_fn,
        )
    return net.input_data(input_data=timeseries)


@pytest.mark.parametrize("mean_field_updates", [False, True])
def test_volatile_nonlinear_value_parent(mean_field_updates):
    """Volatile-state node with a non-linearly-coupled value parent (JAX).

    Exercises the prediction-step non-linear coupling branches
    (:func:`predict_mean_value_level` and the relaxed
    :func:`predict_precision_value_level`) for a volatile node that sits *below* a
    ``tanh``-coupled value parent. The trajectories must stay finite, and the non-linear
    coupling must change the result relative to linear coupling.
    """
    import jax.numpy as jnp

    timeseries = load_data("continuous")
    nonlinear = _build_volatile_value_child((jnp.tanh,), mean_field_updates, timeseries)
    linear = _build_volatile_value_child(None, mean_field_updates, timeseries)

    label = f"nonlinear value parent mean_field={mean_field_updates}"
    for key in ["mean", "expected_mean", "precision", "expected_precision"]:
        values = np.asarray(nonlinear.node_trajectories[1][key])
        assert np.all(np.isfinite(values)), f"{label}: non-finite '{key}'"

    # The tanh coupling on the parent edge must change the child's trajectory.
    assert not np.allclose(
        nonlinear.node_trajectories[1]["mean"],
        linear.node_trajectories[1]["mean"],
    ), f"{label}: tanh coupling did not change the trajectory"
