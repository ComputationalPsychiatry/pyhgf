# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np
import pytest

from pyhgf.model import Network
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
    continuous_node_posterior_update_unbounded,
)

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64


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


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF", "unbounded"])
def test_noise_parent_of_two_inputs(volatility_updates):
    r"""A noise parent of several inputs adds one term of Eqs. 74 and 78 per input.

    Inputs 0 and 1 have value parents 2 and 3 and share the noise parent 4, with
    couplings :math:`\kappa_0` and :math:`\kappa_1`:

    .. math::

        \pi_4 = \hat{\pi}_4 + \sum_i \tfrac12 \kappa_i^2 (1 + \epsilon_i), \qquad
        \mu_4 = \hat{\mu}_4 + \frac{1}{2 \pi_4} \sum_i \kappa_i \epsilon_i,

    with :math:`\epsilon_i` the noise prediction error of input :math:`i`, computed
    here from the posteriors of the value parents (Eq. 72 of [1]_). A noise parent
    takes this update under every ``volatility_updates`` setting.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    kappas = (1.0, 0.6)
    observations = np.random.default_rng(1).normal(0.0, 2.0, (60, 2))
    with enable_x64(True):
        network = (
            Network(volatility_updates=volatility_updates)
            .add_nodes(n_nodes=2)
            .add_nodes(value_children=0)
            .add_nodes(value_children=1)
            .add_nodes(volatility_children=([0, 1], list(kappas)))
            .input_data(input_data=observations)
        )
        traj = network.node_trajectories

        precision = np.asarray(traj[4]["expected_precision"])
        mean_shift = np.zeros_like(precision)
        for input_idx, value_parent_idx, kappa in ((0, 2, kappas[0]), (1, 3, kappas[1])):  # fmt: skip
            noise_precision = np.asarray(traj[input_idx]["expected_precision"])
            residual = observations[:, input_idx] - np.asarray(traj[value_parent_idx]["mean"])  # fmt: skip
            epsilon = (
                noise_precision / np.asarray(traj[value_parent_idx]["precision"])
                + noise_precision * residual**2
                - 1
            )
            assert np.allclose(
                traj[input_idx]["temp"]["volatility_prediction_error"], epsilon
            )
            precision = precision + 0.5 * kappa**2 * (1 + epsilon)
            mean_shift = mean_shift + 0.5 * kappa * epsilon
        assert np.allclose(traj[4]["precision"], precision, rtol=1e-12)
        assert np.allclose(
            traj[4]["mean"],
            np.asarray(traj[4]["expected_mean"]) + mean_shift / precision,
            rtol=1e-12,
        )
