# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np

from pyhgf.model import Network

try:
    from jax import enable_x64
except ImportError:  # jax releases exporting it from jax.experimental only
    from jax.experimental import enable_x64


def test_noise_prediction_error_with_several_value_parents():
    r"""With several value parents, the noise prediction error reads all posteriors.

    The input has a linear value parent (node 1, coupling 0.5), a value parent with
    the coupling function :math:`\tanh` (node 2, coupling 1.5), and a noise parent
    (node 3). Its noise prediction error is the expected squared noise to first order
    under the posteriors of both value parents,

    .. math::

        \epsilon_u = \hat{\pi}_u \left( \sum_b
            \frac{(\alpha_b g_b'(\mu_b))^2}{\pi_b}
            + \left(u - \sum_b \alpha_b g_b(\mu_b)\right)^2 \right) - 1,

    which is Eq. 72 of [1]_ for one linear value parent with :math:`\alpha = 1`.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2026). The generalized hierarchical Gaussian filter.
       doi:10.7554/elife.110174.1
    """
    alphas = (0.5, 1.5)
    # a slowly varying series keeps the posterior precision of the tanh parent
    # positive (large jumps can make its curvature term negative)
    noise = np.random.default_rng(0).normal(size=40)
    observations = np.sin(np.arange(40) / 4.0) + 0.3 * noise
    with enable_x64(True):
        network = (
            Network()
            .add_nodes()
            .add_nodes(value_children=([0], [alphas[0]]))
            .add_nodes(value_children=([0], [alphas[1]]), coupling_fn=(jnp.tanh,))
            .add_nodes(volatility_children=0)
            .input_data(input_data=observations)
        )
        traj = network.node_trajectories
        for node_idx in range(4):
            assert np.all(np.isfinite(np.asarray(traj[node_idx]["precision"])))
        mu_1, pi_1 = np.asarray(traj[1]["mean"]), np.asarray(traj[1]["precision"])
        mu_2, pi_2 = np.asarray(traj[2]["mean"]), np.asarray(traj[2]["precision"])
        variance = alphas[0] ** 2 / pi_1 + (alphas[1] / np.cosh(mu_2) ** 2) ** 2 / pi_2
        mean = alphas[0] * mu_1 + alphas[1] * np.tanh(mu_2)
        noise_precision = np.asarray(traj[0]["expected_precision"])
        epsilon = noise_precision * (variance + (observations - mean) ** 2) - 1
        assert np.allclose(
            traj[0]["temp"]["volatility_prediction_error"], epsilon, rtol=1e-10
        )
