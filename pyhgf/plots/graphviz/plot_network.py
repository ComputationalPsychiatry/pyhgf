# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphviz.sources import Source

    from pyhgf.model import Network


def plot_network(network: "Network") -> "Source":
    """Visualization of node network using GraphViz.

    Parameters
    ----------
    network :
        An instance of main Network class.

    Notes
    -----
    This function requires [Graphviz](https://github.com/xflr6/graphviz) to be
    installed to work correctly.

    """
    try:
        import graphviz
    except ImportError:
        print(
            (
                "Graphviz is required to plot networks. "
                "See https://pypi.org/project/graphviz/"
            )
        )

    graphviz_structure = graphviz.Digraph("hgf-nodes", comment="Nodes structure")

    graphviz_structure.attr("node", shape="circle")

    # create the rest of nodes
    for idx in range(len(network.edges)):
        style = "filled" if idx in network.input_idxs else ""

        if network.edges[idx].node_type == 1:
            # binary state node
            graphviz_structure.node(
                f"x_{idx}", label=str(idx), shape="square", style=style
            )

        elif network.edges[idx].node_type == 2:
            # Continuous state nore
            graphviz_structure.node(
                f"x_{idx}", label=str(idx), shape="circle", style=style
            )

        elif network.edges[idx].node_type == 3:
            # Exponential family state nore
            graphviz_structure.node(
                f"x_{idx}",
                label=f"EF-{idx}",
                style="filled",
                shape="circle",
                fillcolor="#ced6e4",
            )

        elif network.edges[idx].node_type == 4:
            # Dirichlet Process state node
            graphviz_structure.node(
                f"x_{idx}",
                label=f"DP-{idx}",
                style="filled",
                shape="doublecircle",
                fillcolor="#e2d8c1",
            )

        elif network.edges[idx].node_type == 5:
            # Categorical state node
            graphviz_structure.node(
                f"x_{idx}",
                label=f"Ca-{idx}",
                style=style,
                shape="diamond",
                fillcolor="#e2d8c1",
            )

    # connect value parents
    for i, index in enumerate(network.edges):
        value_parents = index.value_parents

        if value_parents is not None:
            for value_parents_idx in value_parents:
                # get the coupling function from the value parent
                child_idx = network.edges[value_parents_idx].value_children.index(i)
                coupling_fn = network.edges[value_parents_idx].coupling_fn[child_idx]
                graphviz_structure.edge(
                    f"x_{value_parents_idx}",
                    f"x_{i}",
                    color="black" if coupling_fn is None else "black:invis:black",
                )

    # connect volatility parents
    for i, index in enumerate(network.edges):
        volatility_parents = index.volatility_parents

        if volatility_parents is not None:
            for volatility_parents_idx in volatility_parents:
                graphviz_structure.edge(
                    f"x_{volatility_parents_idx}",
                    f"x_{i}",
                    color="gray",
                    style="dashed",
                    arrowhead="dot",
                )

    # unflat the structure to better handle large/uneven networks
    graphviz_structure = graphviz_structure.unflatten(stagger=3)

    return graphviz_structure
