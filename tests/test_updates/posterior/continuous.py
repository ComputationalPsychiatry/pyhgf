# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp

from pyhgf.model import Network
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
)


def test_continuous_posterior_updates():

    network = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .add_nodes(volatility_children=2)
    )

    # Standard HGF updates -------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # value update
    attributes, edges, _ = network.get_network()
    attributes[0]["temp"]["value_prediction_error"] = 1.0357
    attributes[0]["mean"] = 1.0357

    new_attributes = continuous_node_posterior_update(
        attributes=attributes, node_idx=1, edges=edges
    )
    assert jnp.isclose(new_attributes[1]["mean"], 0.51785)

    # volatility update
    attributes, edges, _ = network.get_network()
    attributes[1]["temp"]["effective_precision"] = 0.01798621006309986
    attributes[1]["temp"]["value_prediction_error"] = 0.5225493907928467
    attributes[1]["temp"]["volatility_prediction_error"] = -0.23639076948165894
    attributes[1]["expected_precision"] = 0.9820137619972229
    attributes[1]["mean"] = 0.5225493907928467
    attributes[1]["precision"] = 1.9820137023925781

    new_attributes = continuous_node_posterior_update(
        attributes=attributes, node_idx=2, edges=edges
    )
    assert jnp.isclose(new_attributes[1]["mean"], -0.0021212)
    assert jnp.isclose(new_attributes[1]["precision"], 1.0022112)

    # eHGF updates ---------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # value update
    attributes, edges, _ = network.get_network()
    attributes[0]["temp"]["value_prediction_error"] = 1.0357
    attributes[0]["mean"] = 1.0357

    new_attributes = continuous_node_posterior_update_ehgf(
        attributes=attributes, node_idx=2, edges=edges
    )
    assert jnp.isclose(new_attributes[1]["mean"], 0.51785)

    # volatility update
    attributes, edges, _ = network.get_network()
    attributes[1]["temp"]["effective_precision"] = 0.01798621006309986
    attributes[1]["temp"]["value_prediction_error"] = 0.5225493907928467
    attributes[1]["temp"]["volatility_prediction_error"] = -0.23639076948165894
    attributes[1]["expected_precision"] = 0.9820137619972229
    attributes[1]["mean"] = 0.5225493907928467
    attributes[1]["precision"] = 1.9820137023925781

    new_attributes = continuous_node_posterior_update_ehgf(
        attributes=attributes, node_idx=2, edges=edges
    )
    assert jnp.isclose(new_attributes[1]["mean"], -0.00212589)
    assert jnp.isclose(new_attributes[1]["precision"], 1.0022112)
