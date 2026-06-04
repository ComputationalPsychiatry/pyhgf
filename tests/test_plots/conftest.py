# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Shared fixtures for the per-function plotting tests.

Each plotting function is tested in its own file under
:mod:`tests.test_plots`, but they share a common pool of fitted networks so
the cost of building/inferring them is paid once per test session.
"""

import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import pytest

from pyhgf import load_data
from pyhgf.model import HGF, DeepNetwork, Network

# Ensure the test suite runs on headless CI runners.
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Continuous HGF fixtures (USD-CHF time-series)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def continuous_data():
    """USD-CHF continuous input series."""
    return load_data("continuous")


@pytest.fixture(scope="session")
def two_level_continuous(continuous_data):
    """Build a 2-level continuous HGF fitted to the USD-CHF series."""
    return HGF(
        n_levels=2,
        model_type="continuous",
        initial_mean={"1": 1.04, "2": 1.0},
        initial_precision={"1": 1e4, "2": 1e1},
        tonic_volatility={"1": -13.0, "2": -2.0},
        tonic_drift={"1": 0.0, "2": 0.0},
        volatility_coupling={"1": 1.0},
    ).input_data(input_data=continuous_data)


@pytest.fixture(scope="session")
def three_level_continuous(continuous_data):
    """Build a 3-level continuous HGF fitted to the USD-CHF series."""
    return HGF(
        n_levels=3,
        model_type="continuous",
        initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
        initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
        tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
        tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": 1.0, "2": 1.0},
    ).input_data(input_data=continuous_data)


# ---------------------------------------------------------------------------
# Binary HGF fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def binary_data():
    """Binary HGF input series."""
    u, _ = load_data("binary")
    return u


@pytest.fixture(scope="session")
def two_level_binary(binary_data):
    """Build a 2-level binary HGF fitted to the binary input series."""
    return HGF(
        n_levels=2,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5},
        initial_precision={"1": 0.0, "2": 1e4},
        tonic_volatility={"1": None, "2": -6.0},
        tonic_drift={"1": None, "2": 0.0},
        volatility_coupling={"1": None},
        binary_precision=jnp.inf,
    ).input_data(binary_data)


@pytest.fixture(scope="session")
def three_level_binary(binary_data):
    """Build a 3-level binary HGF fitted to the binary input series."""
    return HGF(
        n_levels=3,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5, "3": 0.0},
        initial_precision={"1": 0.0, "2": 1e4, "3": 1e1},
        tonic_volatility={"1": None, "2": -6.0, "3": -2.0},
        tonic_drift={"1": None, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": None, "2": 1.0},
        binary_precision=jnp.inf,
    ).input_data(binary_data)


# ---------------------------------------------------------------------------
# Categorical fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def categorical_network():
    """Build a 3-category categorical HGF fitted to a small synthetic series."""
    rng = np.random.default_rng(0)
    input_data = np.array(
        [rng.multinomial(n=1, pvals=[0.1, 0.2, 0.7]) for _ in range(10)],
        dtype=float,
    )
    net = Network().add_nodes(
        kind="categorical-state",
        node_parameters={
            "n_categories": 3,
            "binary_parameters": {"tonic_volatility_2": -2.0},
        },
    )
    return net.input_data(input_data=input_data)


# ---------------------------------------------------------------------------
# Volatile-state fixture (for the NetworkX rendering path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def volatile_state_network():
    """Build a small network mixing continuous-state and volatile-state nodes."""
    rng = np.random.default_rng(0)
    net = (
        Network()
        .add_nodes(kind="continuous-state", n_nodes=2)
        .add_nodes(
            kind="volatile-state",
            n_nodes=2,
            value_children=[0, 1],
        )
    )
    return net.input_data(input_data=rng.standard_normal((5, 2)))


# ---------------------------------------------------------------------------
# DeepNetwork fixtures (for plot_layers / plot_deep_network)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trained_deep_network():
    """Build a small 3-layer DeepNetwork fitted with ``record_trajectories=True``."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3)).astype(np.float32)
    y = rng.standard_normal((20, 2)).astype(np.float32)

    from pyhgf.typing.vectorised import RECORD_ALL

    net = DeepNetwork().add_layer(size=2).add_layer(size=4).add_layer(size=3)
    net.fit(
        x=x,
        y=y,
        optimizer=optax.sgd(0.05),
        learning_kind="precision_weighted",
        record=RECORD_ALL,
    )
    return net


@pytest.fixture(scope="session")
def deep_network_for_graphviz():
    """Build an unfitted DeepNetwork to exercise the graphviz visualisation."""
    return (
        DeepNetwork()
        .add_layer(size=4)
        .add_layer(size=3, tonic_volatility=-1.0)
        .add_layer(size=2, tonic_volatility=-2.0)
    )
