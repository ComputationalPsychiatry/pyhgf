# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Smoke-tests for :func:`pyhgf.plots.matplotlib.plot_trajectories`."""

import matplotlib.pyplot as plt


def test_plot_trajectories_two_level_continuous(two_level_continuous):
    """Render trajectories for a 2-level continuous HGF."""
    two_level_continuous.plot_trajectories(show_total_surprise=True)
    plt.close("all")


def test_plot_trajectories_three_level_continuous(three_level_continuous):
    """Render trajectories for a 3-level continuous HGF."""
    three_level_continuous.plot_trajectories(show_total_surprise=True)
    plt.close("all")


def test_plot_trajectories_two_level_binary(two_level_binary):
    """Render trajectories for a 2-level binary HGF."""
    two_level_binary.plot_trajectories(show_total_surprise=True)
    plt.close("all")


def test_plot_trajectories_three_level_binary(three_level_binary):
    """Render trajectories for a 3-level binary HGF."""
    three_level_binary.plot_trajectories(show_total_surprise=True)
    plt.close("all")
