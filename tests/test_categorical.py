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

    # sanity check on the network structure
    # ensure that the number of parents and children match the number of coupling values
    for i in range(len(categorical_hgf.edges)):
        if categorical_hgf.edges[i].node_type == 2:
            # value parents ------------------------------------------------------------
            if categorical_hgf.edges[i].value_parents:
                assert len(categorical_hgf.edges[i].value_parents) == len(
                    categorical_hgf.attributes[i]["value_coupling_parents"]
                )
            else:
                assert (categorical_hgf.edges[i].value_parents is None) and (
                    categorical_hgf.attributes[i]["value_coupling_parents"] is None
                )

            # value children -----------------------------------------------------------
            if categorical_hgf.edges[i].value_children:
                assert len(categorical_hgf.edges[i].value_children) == len(
                    categorical_hgf.attributes[i]["value_coupling_children"]
                )
            else:
                assert (categorical_hgf.edges[i].value_children is None) and (
                    categorical_hgf.attributes[i]["value_coupling_children"] is None
                )

            # volatility parents -------------------------------------------------------
            if categorical_hgf.edges[i].volatility_parents:
                assert len(categorical_hgf.edges[i].volatility_parents) == len(
                    categorical_hgf.attributes[i]["volatility_coupling_parents"]
                )
            else:
                assert (categorical_hgf.edges[i].volatility_parents is None) and (
                    categorical_hgf.attributes[i]["volatility_coupling_parents"] is None
                )

            # volatility children ------------------------------------------------------
            if categorical_hgf.edges[i].volatility_children:
                assert len(categorical_hgf.edges[i].volatility_children) == len(
                    categorical_hgf.attributes[i]["volatility_coupling_children"]
                )
            else:
                assert (categorical_hgf.edges[i].volatility_children is None) and (
                    categorical_hgf.attributes[i]["volatility_coupling_children"]
                    is None
                )

    # fitting the model forwards
    categorical_hgf.input_data(input_data=input_data)

    # export to pandas data frame
    categorical_hgf.to_pandas()

    assert jnp.isclose(
        categorical_hgf.node_trajectories[0]["kl_divergence"].sum(), 1.2846234
    )
    assert jnp.isclose(
        categorical_hgf.node_trajectories[0]["surprise"].sum(), 12.514741
    )
