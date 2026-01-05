# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Louie Mølgaard Hessellund

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

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
    try:
        import networkx as nx
    except ImportError:
        print(
            (
                "NetworkX and pydot are required to plot networks. "
                "See https://networkx.org/documentation/stable/install.html"
            )
        )
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for idx in range(len(network.edges)):
        # Check if it's an input node
        is_input = idx in network.input_idxs
        # Check if it's a continuous state node
        if network.edges[idx].node_type == 2:
            G.add_node(f"x_{idx}", is_input=is_input, label=str(idx), node_type=2)
        # Check if it's a value-volatility node
        elif network.edges[idx].node_type == 6:
            G.add_node(f"x_{idx}", is_input=is_input, label=str(idx), node_type=6)

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

    # Separate regular and value-volatility nodes
    regular_nodes = [node for node in G.nodes() if G.nodes[node]["node_type"] == 2]
    vv_nodes = [node for node in G.nodes() if G.nodes[node]["node_type"] == 6]

    # Draw regular continuous state nodes
    if regular_nodes:
        regular_colors = [
            "lightblue" if G.nodes[node]["is_input"] else "white"
            for node in regular_nodes
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=regular_nodes,
            node_color=regular_colors,
            node_size=node_size,
            edgecolors="black",
            ax=ax,
        )

    # Draw value-volatility nodes with double circle effect
    if vv_nodes:
        import matplotlib.patches as mpatches

        # Calculate radius from node_size (node_size is in points^2)
        radius_inner = (node_size / 3.14159) ** 0.5 / 72  # Convert to data units
        radius_outer = radius_inner * 1.15  # Outer circle is 15% larger

        for node in vv_nodes:
            node_color = "lightblue" if G.nodes[node]["is_input"] else "white"
            x, y = pos[node]

            # Draw outer circle with dashed gray line (volatility)
            outer_circle = mpatches.Circle(
                (x, y),
                radius_outer,
                fill=False,
                edgecolor="gray",
                linewidth=2,
                linestyle=(0, (5, 5)),  # Explicit dash pattern: (offset, (dash, gap))
                zorder=2,
            )
            ax.add_patch(outer_circle)

            # Draw inner circle with solid black line (value)
            inner_circle = mpatches.Circle(
                (x, y),
                radius_inner,
                facecolor=node_color,
                edgecolor="black",
                linewidth=1.5,
                linestyle="-",
                zorder=3,
            )
            ax.add_patch(inner_circle)

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
