# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Louie Mølgaard Hessellund

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

if TYPE_CHECKING:

    from pyhgf.model import Network


def plot_network(
    network: "Network", figsize=(4, 4), node_size=700, ax=None, scale=1, arrow_size=35
):
    """Visualization of node network using NetworkX and pydot layout.

    Parameters
    ----------
    network : Network
        An instance of main Network class.
    figsize : tuple, optional
        Figure size in inches (width, height), by default (4, 4)
    node_size : int, optional
        Size of the nodes in the visualization, by default 700
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, creates a new figure, by default None
    scale : float, optional
        Scale factor for node positioning, by default 1
    arrow_size : int, optional
        Size of the arrows for volatility edges, by default 35

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the network visualization if ax is None,
        otherwise returns the NetworkX graph object

    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for idx in range(len(network.edges)):
        # Check if it's an input node
        is_input = idx in network.input_idxs
        # Check if it's a continuous state node
        if network.edges[idx].node_type == 2:
            G.add_node(f"x_{idx}", is_input=is_input, label=str(idx))

    # Add value parent edges
    for i, edge in enumerate(network.edges):
        value_parents = edge.value_parents
        if value_parents is not None:
            for value_parents_idx in value_parents:
                # Get the coupling function
                child_idx = network.edges[value_parents_idx].value_children.index(i)
                coupling_fn = network.edges[value_parents_idx].coupling_fn[child_idx]

                # Add edge with appropriate style
                G.add_edge(
                    f"x_{value_parents_idx}",
                    f"x_{i}",
                    edge_type="value",
                    coupling=coupling_fn is not None,
                )

    # Add volatility parent edges
    for i, edge in enumerate(network.edges):
        volatility_parents = edge.volatility_parents
        if volatility_parents is not None:
            for volatility_parents_idx in volatility_parents:
                G.add_edge(
                    f"x_{volatility_parents_idx}", f"x_{i}", edge_type="volatility"
                )

    # Create the plot if no axis is provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Use pydot layout for hierarchical arrangement
    pos = nx.nx_pydot.pydot_layout(G, prog="dot", root=None)

    # Scale the positions
    pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

    # Draw nodes
    node_colors = [
        "lightblue" if G.nodes[node]["is_input"] else "white" for node in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_size, edgecolors="black", ax=ax
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, labels={node: G.nodes[node]["label"] for node in G.nodes()}, ax=ax
    )

    # Draw value parent edges
    coupling_edges = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if d["edge_type"] == "value" and d["coupling"]
    ]
    normal_edges = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if d["edge_type"] == "value" and not d["coupling"]
    ]

    # Draw normal value edges
    nx.draw_networkx_edges(
        G, pos, edgelist=normal_edges, edge_color="black", arrowsize=arrow_size, ax=ax
    )

    # Draw coupling edges with a different style
    nx.draw_networkx_edges(
        G, pos, edgelist=coupling_edges, edge_color="black", style="dashed", ax=ax
    )

    # Draw volatility edges
    volatility_edges = [
        (u, v) for (u, v, d) in G.edges(data=True) if d["edge_type"] == "volatility"
    ]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=volatility_edges,
        edge_color="gray",
        style="dashed",
        arrowstyle="->",
        arrowsize=arrow_size,
        ax=ax,
    )

    ax.axis("off")
    return plt.gcf() if ax is None else G
