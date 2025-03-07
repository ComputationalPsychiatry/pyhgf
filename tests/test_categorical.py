# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np

from pyhgf.model import Network


def test_categorical_state_node():
    # generate some categorical inputs data
    np.random.seed(123)
    input_data = np.array(
        [np.random.multinomial(n=1, pvals=[0.1, 0.2, 0.7]) for _ in range(10)],
        dtype=float,
    )

    # create the categorical HGF
    categorical_hgf = Network().add_nodes(
        kind="categorical-state",
        node_parameters={
            "n_categories": 3,
            "binary_parameters": {"tonic_volatility_2": -2.0},
        },
    )

    # fitting the model forwards
    categorical_hgf.input_data(input_data=input_data)

    # export to pandas data frame
    categorical_hgf.to_pandas()

    assert jnp.isclose(
        categorical_hgf.node_trajectories[0]["kl_divergence"].sum(), 1.2846234
    )
    assert jnp.isclose(
        categorical_hgf.node_trajectories[0]["surprise"].sum(), 12.514427
    )
