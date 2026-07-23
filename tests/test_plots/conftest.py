# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Shared fixtures for the per-function plotting tests.

Each plotting function is tested in its own file under
:mod:`tests.test_plots`, but they share a common pool of fitted networks so
the cost of building/inferring them is paid once per test session.
"""

import matplotlib
import numpy as np
import optax
import pytest

from pyhgf import load_data
from pyhgf.model import DeepNetwork, Network

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
    return (
        Network()
        .add_nodes(precision=1e4)
        .add_nodes(
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 1.04,
                "precision": 1e4,
                "tonic_volatility": -13.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 1.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
        .input_data(input_data=continuous_data)
    )


@pytest.fixture(scope="session")
def three_level_continuous(continuous_data):
    """Build a 3-level continuous HGF fitted to the USD-CHF series."""
    return (
        Network()
        .add_nodes(precision=1e4)
        .add_nodes(
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 1.04,
                "precision": 1e4,
                "tonic_volatility": -13.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 1.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([2], [1.0]),
            node_parameters={
                "mean": 1.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
        .input_data(input_data=continuous_data)
    )


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
    return (
        Network()
        .add_nodes(
            kind="binary-state",
            node_parameters={"mean": 0.0, "precision": 0.0},
        )
        .add_nodes(
            kind="continuous-state",
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 0.5,
                "precision": 1e4,
                "tonic_volatility": -6.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
        .input_data(binary_data)
    )


@pytest.fixture(scope="session")
def three_level_binary(binary_data):
    """Build a 3-level binary HGF fitted to the binary input series."""
    return (
        Network()
        .add_nodes(
            kind="binary-state",
            node_parameters={"mean": 0.0, "precision": 0.0},
        )
        .add_nodes(
            kind="continuous-state",
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 0.5,
                "precision": 1e4,
                "tonic_volatility": -6.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 0.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
        .input_data(binary_data)
    )


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
        .add_layer(size=3, tonic_volatility_vol=-1.0)
        .add_layer(size=2, tonic_volatility_vol=-2.0)
    )
