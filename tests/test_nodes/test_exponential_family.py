# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np
from pyhgf.rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork

NETWORK_CLASSES = [PyNetwork, RsNetwork]


def test_gaussian():
    """Test the Gaussian node."""
    timeseries = load_data("continuous")

    results = {}
    for cls in NETWORK_CLASSES:
        results[cls.__name__] = (
            cls().add_nodes(kind="ef-state").input_data(input_data=timeseries)
        )

    # Ensure identical results across implementations
    ref = results[NETWORK_CLASSES[0].__name__]
    for name, net in results.items():
        if net is ref:
            continue
        for key in ["xis", "mean", "nus"]:
            assert np.isclose(
                ref.node_trajectories[0][key],
                net.node_trajectories[0][key],
            ).all(), f"{name} key '{key}' mismatch"


def test_multivariate_gaussian():
    """Test the multivariate Gaussian node."""
    # simulate an ordered spiral data set
    np.random.seed(123)
    N = 1000
    theta = np.sort(np.sqrt(np.random.rand(N)) * 5 * np.pi)
    r_a = -2 * theta - np.pi
    spiral_data = (
        np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        + np.random.randn(N, 2) * 2
    )

    # Python
    # ----------------------------------------------------------------------------------

    # generalised filtering
    bivariate_normal = (
        PyNetwork()
        .add_nodes(
            kind="ef-state",
            learning="generalised-filtering",
            distribution="multivariate-normal",
            dimension=2,
        )
        .input_data(input_data=spiral_data)
    )
    assert jnp.isclose(
        bivariate_normal.node_trajectories[0]["xis"][-1],
        jnp.array(
            [3.4652710e01, -1.0609777e00, 1.2103647e03, -3.6398651e01, 3.3951855e00],
            dtype="float32",
        ),
    ).all()

    # hgf updates
    bivariate_hgf = PyNetwork().add_nodes(
        kind="ef-state",
        learning="hgf-2",
        distribution="multivariate-normal",
        dimension=2,
    )

    # adapting prior parameter values to the sufficient statistics
    # covariances statistics will have greater variability and amplitudes
    for node_idx in [2, 5, 8, 11, 14]:
        bivariate_hgf.attributes[node_idx]["tonic_volatility"] = -2.0
    for node_idx in [1, 4, 7, 10, 13]:
        bivariate_hgf.attributes[node_idx]["precision"] = 0.01
    for node_idx in [9, 12, 15]:
        bivariate_hgf.attributes[node_idx]["mean"] = 10.0

    bivariate_hgf.input_data(input_data=spiral_data)

    assert jnp.isclose(
        bivariate_normal.node_trajectories[0]["xis"][-1],
        jnp.array(
            [3.4652710e01, -1.0609777e00, 1.2103647e03, -3.6398651e01, 3.3951855e00],
            dtype="float32",
        ),
    ).all()
