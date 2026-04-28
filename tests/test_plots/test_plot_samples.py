# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Smoke-tests for :func:`pyhgf.plots.matplotlib.plot_samples`."""

import matplotlib.pyplot as plt
import numpy as np
from jax.random import PRNGKey


def test_plot_samples_three_level_continuous(three_level_continuous):
    """Sample from a 3-level continuous HGF and render the resulting samples."""
    three_level_continuous.create_belief_propagation_fn(sampling_fn=True)
    three_level_continuous.sample(
        time_steps=np.ones(100),
        rng_key=PRNGKey(4),
        n_predictions=50,
    )
    three_level_continuous.plot_samples()
    plt.close("all")


def test_plot_samples_three_level_binary(three_level_binary):
    """Sample from a 3-level binary HGF and render the resulting samples."""
    three_level_binary.create_belief_propagation_fn(sampling_fn=True)
    three_level_binary.sample(
        time_steps=np.ones(100),
        rng_key=PRNGKey(4),
        n_predictions=50,
    )
    three_level_binary.plot_samples()
    plt.close("all")
