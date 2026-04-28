# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Smoke-tests for :func:`pyhgf.plots.matplotlib.plot_correlations`."""

import matplotlib.pyplot as plt


def test_plot_correlations_two_level_continuous(two_level_continuous):
    """Render the correlation heatmap for a 2-level continuous HGF."""
    two_level_continuous.plot_correlations()
    plt.close("all")


def test_plot_correlations_three_level_continuous(three_level_continuous):
    """Render the correlation heatmap for a 3-level continuous HGF."""
    three_level_continuous.plot_correlations()
    plt.close("all")


def test_plot_correlations_two_level_binary(two_level_binary):
    """Render the correlation heatmap for a 2-level binary HGF."""
    two_level_binary.plot_correlations()
    plt.close("all")


def test_plot_correlations_three_level_binary(three_level_binary):
    """Render the correlation heatmap for a 3-level binary HGF."""
    three_level_binary.plot_correlations()
    plt.close("all")
