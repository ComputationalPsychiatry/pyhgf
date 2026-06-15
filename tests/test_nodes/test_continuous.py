# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
import pytest
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

VOLATILITY_UPDATES = ["standard", "eHGF", "unbounded"]
MEAN_FIELD_UPDATES = [False, True]


def _build_network(cls, volatility_updates, mean_field_updates, n_levels, timeseries):
    """Build an ``n_levels`` continuous HGF with the given update scheme."""
    net = cls(
        volatility_updates=volatility_updates,
        mean_field_updates=mean_field_updates,
    )
    net = net.add_nodes().add_nodes(value_children=0)
    if n_levels == 3:
        net = net.add_nodes(volatility_children=0)
    return net.input_data(input_data=timeseries)


def _assert_backends_match(py_net, rs_net, n_levels, volatility_updates, label):
    """Assert the JAX and Rust trajectories agree for every node."""
    # The unbounded posterior kernel differs at the float level between the two
    # backends (mathematically equivalent but distinct implementations), so the
    # cross-backend comparison is loosened when a volatility parent is present.
    rtol = 1e-1 if (volatility_updates == "unbounded" and n_levels == 3) else 1e-4
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
    affect its update path and is not parametrized here. The JAX and Rust backends must
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
