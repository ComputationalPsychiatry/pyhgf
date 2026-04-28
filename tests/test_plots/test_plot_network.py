# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Smoke-tests for :meth:`Network.plot_network` (graphviz + networkx backends)."""

import matplotlib.pyplot as plt


def test_plot_network_two_level_continuous(two_level_continuous):
    """Render a 2-level continuous HGF via the default Graphviz backend."""
    two_level_continuous.plot_network()
    plt.close("all")


def test_plot_network_three_level_continuous(three_level_continuous):
    """Render a 3-level continuous HGF via both Graphviz and NetworkX backends."""
    three_level_continuous.plot_network()
    three_level_continuous.plot_network(backend="networkx")
    plt.close("all")


def test_plot_network_two_level_binary(two_level_binary):
    """Render a 2-level binary HGF via the default Graphviz backend."""
    two_level_binary.plot_network()
    plt.close("all")


def test_plot_network_three_level_binary(three_level_binary):
    """Render a 3-level binary HGF via the default Graphviz backend."""
    three_level_binary.plot_network()
    plt.close("all")


def test_plot_network_categorical(categorical_network):
    """Render a categorical HGF (the only plotting function it exposes)."""
    categorical_network.plot_network()
    plt.close("all")


def test_plot_network_networkx_volatile_state_nodes(volatile_state_network):
    """Exercise the NetworkX rendering path for volatile-state (type 6) nodes."""
    volatile_state_network.plot_network(backend="networkx")
    plt.close("all")
