# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Smoke-tests for :func:`pyhgf.plots.graphviz.plot_network.plot_deep_network`."""


def test_plot_deep_network(deep_network_for_graphviz):
    """Render the static structure of a small DeepNetwork via Graphviz."""
    from pyhgf.plots.graphviz.plot_network import plot_deep_network

    plot_deep_network(deep_network_for_graphviz, view=False)
