# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import pytest

from pyhgf.model import Network
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
    continuous_node_posterior_update_unbounded,
)


def test_continuous_posterior_updates():
    """Test the continuous posterior updates."""
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
    assert jnp.isclose(new_attributes[2]["mean"], -0.0021212)
    assert jnp.isclose(new_attributes[2]["precision"], 1.0022112)

    # eHGF updates ---------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # value update
    attributes, edges, _ = network.get_network()
    attributes[0]["temp"]["value_prediction_error"] = 1.0357
    attributes[0]["mean"] = 1.0357

    new_attributes = continuous_node_posterior_update_ehgf(
        attributes=attributes, node_idx=2, edges=edges
    )
    assert jnp.isclose(new_attributes[1]["mean"], 0.5225494)

    # volatility update
    # The enhanced-HGF precision update recomputes the effective precision from the
    # node's posterior mean and the elapsed time, so ``time_step`` and the child's
    # previous variance must be set (as in the unbounded block below). Node 1 has no
    # value parent, so its conditional and marginal predicted precisions coincide.
    attributes, edges, _ = network.get_network()
    attributes[-1]["time_step"] = 1.0
    attributes[1]["temp"]["current_variance"] = 1.0
    attributes[1]["temp"]["conditional_expected_precision"] = 0.9820137619972229
    attributes[1]["temp"]["effective_precision"] = 0.01798621006309986
    attributes[1]["temp"]["value_prediction_error"] = 0.5225493907928467
    attributes[1]["temp"]["volatility_prediction_error"] = -0.23639076948165894
    attributes[1]["expected_precision"] = 0.9820137619972229
    attributes[1]["mean"] = 0.5225493907928467
    attributes[1]["precision"] = 1.9820137023925781

    new_attributes = continuous_node_posterior_update_ehgf(
        attributes=attributes, node_idx=2, edges=edges
    )
    assert jnp.isclose(new_attributes[2]["mean"], -0.00212589)
    assert jnp.isclose(new_attributes[2]["precision"], 1.0022061)

    # unbounded updates ----------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # volatility update
    # Node 1 has no value parent, so its conditional and marginal predicted
    # precisions coincide.
    attributes, edges, _ = network.get_network()
    attributes[-1]["time_step"] = 1.0
    attributes[1]["temp"]["current_variance"] = 1.0
    attributes[1]["expected_precision"] = 0.9820137619972229
    attributes[1]["temp"]["conditional_expected_precision"] = 0.9820137619972229
    attributes[1]["mean"] = 0.5225493907928467
    attributes[1]["precision"] = 1.9820137023925781

    new_attributes = continuous_node_posterior_update_unbounded(
        attributes=attributes, node_idx=2, edges=edges
    )
    assert jnp.isclose(new_attributes[2]["mean"], 0.01330645)
    assert jnp.isclose(new_attributes[2]["precision"], 1.0161277)


@pytest.mark.parametrize(
    "posterior_update",
    [continuous_node_posterior_update_ehgf, continuous_node_posterior_update_unbounded],
    ids=["eHGF", "unbounded"],
)
def test_volatility_update_counts_the_value_coupling_variance_as_known(
    posterior_update,
):
    r"""The value parents' variance of a volatility child is known variance.

    The eHGF and unbounded updates of node 2 model the predicted variance of its
    volatility child as known variance plus volatility. Moving the child's
    value-coupling variance :math:`T = 1/\hat{\pi}_1 - 1/\bar{\pi}_1` from its
    predicted precisions into its carried variance leaves the update unchanged.
    """
    network = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .add_nodes(volatility_children=2)
    )

    def update(current_variance, conditional_expected_precision, expected_precision):
        attributes, edges, _ = network.get_network()
        attributes[-1]["time_step"] = 1.0
        attributes[1]["temp"]["current_variance"] = current_variance
        attributes[1]["temp"]["conditional_expected_precision"] = (
            conditional_expected_precision
        )
        attributes[1]["expected_precision"] = expected_precision
        attributes[1]["temp"]["effective_precision"] = 0.3
        attributes[1]["temp"]["volatility_prediction_error"] = 0.4
        attributes[1]["mean"] = 0.52
        attributes[1]["precision"] = 1.98
        new_attributes = posterior_update(
            attributes=attributes, node_idx=2, edges=edges
        )
        return new_attributes[2]["mean"], new_attributes[2]["precision"]

    value_coupling_variance, expected_precision = 0.5, 0.6
    conditional = 1.0 / (1.0 / expected_precision - value_coupling_variance)
    in_predictions = update(1.0, conditional, expected_precision)
    in_carried_variance = update(
        1.0 + value_coupling_variance, expected_precision, expected_precision
    )
    for a, b in zip(in_predictions, in_carried_variance):
        assert jnp.isclose(a, b, rtol=1e-6)
    # Without that variance, the same child reads as more volatile.
    without = update(1.0, expected_precision, expected_precision)
    assert not jnp.isclose(in_predictions[1], without[1])
