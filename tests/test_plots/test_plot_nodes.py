# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Smoke-tests for :func:`pyhgf.plots.matplotlib.plot_nodes`."""

import matplotlib.pyplot as plt


def test_plot_nodes_two_level_continuous(two_level_continuous):
    """Render a node plot for a 2-level continuous HGF."""
    two_level_continuous.plot_nodes(node_idxs=2, show_posterior=True)
    plt.close("all")


def test_plot_nodes_three_level_continuous(three_level_continuous):
    """Render a node plot for a 3-level continuous HGF."""
    three_level_continuous.plot_nodes(node_idxs=2, show_posterior=True)
    plt.close("all")


def test_plot_nodes_two_level_binary(two_level_binary):
    """Render a node plot for a 2-level binary HGF."""
    two_level_binary.plot_nodes(node_idxs=1, show_posterior=True)
    plt.close("all")


def test_plot_nodes_three_level_binary(three_level_binary):
    """Render a node plot for a 3-level binary HGF."""
    three_level_binary.plot_nodes(node_idxs=2, show_posterior=True)
    plt.close("all")
