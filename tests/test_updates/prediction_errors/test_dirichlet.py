# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np

from pyhgf.model import Network
from pyhgf.updates.prediction_error.dirichlet import (
    dirichlet_node_prediction_error,
    get_candidate,
)


def test_get_candidate():
    mean, precision = get_candidate(
        value=5.0,
        sensory_precision=1.0,
        expected_mean=jnp.array([0.0, -5.0]),
        expected_sigma=jnp.array([1.0, 3.0]),
    )

    assert jnp.isclose(mean, 5.026636)
    assert jnp.isclose(precision, 1.2752448)


def test_dirichlet_node_prediction_error():

    network = (
        Network()
        .add_nodes(kind="dp-state", batch_size=2)
        .add_nodes(
            kind="ef-state",
            n_nodes=2,
            value_children=0,
            xis=jnp.array([0.0, 1 / 8]),
            nus=15.0,
        )
    )

    attributes, edges, _ = network.get_network()
    dirichlet_node_prediction_error(
        edges=edges,
        attributes=attributes,
        node_idx=0,
    )

    # test the plotting function
    network.plot_network()

    # add observations
    network.input_data(input_data=np.random.normal(0, 1, 5))

    # export to pandas
    network.to_pandas()
