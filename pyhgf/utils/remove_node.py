# Author: Louie Mølgaard Hessellund <hessellundlouie@gmail.com>

from typing import Dict, List, Tuple, Union

from pyhgf.typing import AdjacencyLists, Edges


def _remove_edges(
    attributes: Dict,
    edges: Edges,
    kind: str = "value",
    parent_idxs=Union[int, List[int]],
    children_idxs=Union[int, List[int]],
) -> Tuple[Dict, Edges]:
    """Remove a value or volatility coupling link between a set of nodes.

    Parameters
    ----------
    attributes :
        Attributes of the neural network.
    edges :
        Edges of the neural network.
    kind :
        The kind of coupling to remove, can be `"value"` or `"volatility"`.
    parent_idxs :
        The index(es) of the parent node(s) to disconnect.
    children_idxs :
        The index(es) of the children node(s) to disconnect.

    Returns
    -------
    Tuple[Dict, Edges]
        Updated attributes and edges with removed connections.

    """
    if kind not in ["value", "volatility"]:
        raise ValueError(
            f"The kind of coupling should be value or volatility, got {kind}"
        )

    if isinstance(children_idxs, int):
        children_idxs = [children_idxs]
    if isinstance(parent_idxs, int):
        parent_idxs = [parent_idxs]

    edges_as_list = list(edges)

    # Update parent nodes
    for parent_idx in parent_idxs:
        if parent_idx >= len(edges_as_list):
            continue

        node = edges_as_list[parent_idx]
        children = node.value_children if kind == "value" else node.volatility_children
        coupling_key = f"{kind}_coupling_children"

        if children is not None and children:
            # Get indices of children to keep
            keep_indices = [
                i for i, child in enumerate(children) if child not in children_idxs
            ]
            new_children = tuple(children[i] for i in keep_indices)

            # Update coupling strengths if they exist
            if (
                coupling_key in attributes[parent_idx]
                and attributes[parent_idx][coupling_key]
            ):
                new_strengths = tuple(
                    attributes[parent_idx][coupling_key][i] for i in keep_indices
                )
                attributes[parent_idx][coupling_key] = (
                    new_strengths if new_strengths else None
                )

            # Update node edges
            if kind == "value":
                edges_as_list[parent_idx] = AdjacencyLists(
                    node.node_type,
                    node.value_parents,
                    node.volatility_parents,
                    new_children if new_children else None,
                    node.volatility_children,
                    node.coupling_fn,
                )
            else:
                edges_as_list[parent_idx] = AdjacencyLists(
                    node.node_type,
                    node.value_parents,
                    node.volatility_parents,
                    node.value_children,
                    new_children if new_children else None,
                    node.coupling_fn,
                )

    # Update children nodes
    for child_idx in children_idxs:
        if child_idx >= len(edges_as_list):
            continue

        node = edges_as_list[child_idx]
        parents = node.value_parents if kind == "value" else node.volatility_parents
        coupling_key = f"{kind}_coupling_parents"

        if parents is not None and parents:
            # Get indices of parents to keep
            keep_indices = [
                i for i, parent in enumerate(parents) if parent not in parent_idxs
            ]
            new_parents = tuple(parents[i] for i in keep_indices)

            # Update coupling strengths if they exist
            if (
                coupling_key in attributes[child_idx]
                and attributes[child_idx][coupling_key]
            ):
                new_strengths = tuple(
                    attributes[child_idx][coupling_key][i] for i in keep_indices
                )
                attributes[child_idx][coupling_key] = (
                    new_strengths if new_strengths else None
                )

            # Update node edges
            if kind == "value":
                edges_as_list[child_idx] = AdjacencyLists(
                    node.node_type,
                    new_parents if new_parents else None,
                    node.volatility_parents,
                    node.value_children,
                    node.volatility_children,
                    node.coupling_fn,
                )
            else:
                edges_as_list[child_idx] = AdjacencyLists(
                    node.node_type,
                    node.value_parents,
                    new_parents if new_parents else None,
                    node.value_children,
                    node.volatility_children,
                    node.coupling_fn,
                )

    return attributes, tuple(edges_as_list)


def remove_node(attributes: Dict, edges: Edges, index: int) -> Tuple[Dict, Edges]:
    """Remove a given node from the network.

    This function removes a node from the network by deleting its parameters in the
    attributes and edges variables, and adjusts the indices of the remaining nodes.

    Parameters
    ----------
    attributes :
        The attributes of the network.
    edges :
        The edges of the network.
    index :
        The index of the node to remove.

    Returns
    -------
    Tuple[Dict, Edges]
        Updated attributes and edges with the node removed and indices adjusted.

    """
    # ensure that the node exists in the network
    if index not in attributes or index >= len(edges):
        raise ValueError(f"Node with index {index} does not exist in the network")

    edges_as_list = list(edges)
    node = edges_as_list[index]

    # First remove all connections to/from this node using the _remove_edges function
    if node.value_parents:
        attributes, edges = _remove_edges(
            attributes,
            edges,
            "value",
            parent_idxs=node.value_parents,
            children_idxs=index,
        )
        edges_as_list = list(edges)

    if node.volatility_parents:
        attributes, edges = _remove_edges(
            attributes,
            edges,
            "volatility",
            parent_idxs=node.volatility_parents,
            children_idxs=index,
        )
        edges_as_list = list(edges)

    if node.value_children:
        attributes, edges = _remove_edges(
            attributes,
            edges,
            "value",
            parent_idxs=index,
            children_idxs=node.value_children,
        )
        edges_as_list = list(edges)

    if node.volatility_children:
        attributes, edges = _remove_edges(
            attributes,
            edges,
            "volatility",
            parent_idxs=index,
            children_idxs=node.volatility_children,
        )
        edges_as_list = list(edges)

    # Now remove the node
    edges_as_list.pop(index)
    attributes.pop(index)

    # Create new edges list with adjusted indices
    new_edges = []
    for node in edges_as_list:
        new_value_parents = None
        new_volatility_parents = None
        new_value_children = None
        new_volatility_children = None

        if node.value_parents:
            new_value_parents = tuple(
                p if p < index else p - 1 for p in node.value_parents
            )

        if node.volatility_parents:
            new_volatility_parents = tuple(
                p if p < index else p - 1 for p in node.volatility_parents
            )

        if node.value_children:
            new_value_children = tuple(
                c if c < index else c - 1 for c in node.value_children
            )

        if node.volatility_children:
            new_volatility_children = tuple(
                c if c < index else c - 1 for c in node.volatility_children
            )

        new_edges.append(
            AdjacencyLists(
                node.node_type,
                new_value_parents,
                new_volatility_parents,
                new_value_children,
                new_volatility_children,
                node.coupling_fn,
            )
        )

    # Adjust attributes indices
    new_attributes = {-1: attributes[-1]}  # Preserve the time_step
    for old_idx, attr in attributes.items():
        if old_idx == -1 or old_idx == index:
            continue
        new_idx = old_idx if old_idx < index else old_idx - 1
        new_attributes[new_idx] = attr

    return new_attributes, tuple(new_edges)
