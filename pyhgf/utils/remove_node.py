# Author: Louie Mølgaard Hessellund <hessellundlouie@gmail.com>

from typing import Union

import jax.numpy as jnp

from pyhgf.typing import AdjacencyLists, Edges


def _keep_strengths(node_attributes: dict, key: str, keep: list[int]) -> dict:
    """Keep the coupling strengths of the edges at the positions in ``keep``.

    The strengths are stored as an array (or ``None``), so they are tested against
    ``None``: the truth value of an array is ambiguous for two entries or more, and
    ``False`` for a single zero strength. The attributes of the node are copied, not
    changed in place.
    """
    strengths = node_attributes.get(key)
    if strengths is None:
        return node_attributes
    return {
        **node_attributes,
        key: jnp.array([strengths[i] for i in keep]) if keep else None,
    }


def _remove_edges(
    attributes: dict,
    edges: Edges,
    kind: str = "value",
    parent_idxs=Union[int, list[int]],
    children_idxs=Union[int, list[int]],
) -> tuple[dict, Edges]:
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
    tuple[dict, Edges]
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

    # the attributes passed in (e.g. those of a Network) are left unchanged
    attributes = dict(attributes)
    edges_as_list = list(edges)

    # Update parent nodes
    for parent_idx in parent_idxs:
        if parent_idx >= len(edges_as_list):
            continue
        node = edges_as_list[parent_idx]
        children = getattr(node, f"{kind}_children")
        if not children:
            continue
        keep = [i for i, child in enumerate(children) if child not in children_idxs]
        attributes[parent_idx] = _keep_strengths(
            attributes[parent_idx], f"{kind}_coupling_children", keep
        )
        new_edges = {f"{kind}_children": tuple(children[i] for i in keep) or None}
        if kind == "value":
            # coupling_fn[i] is the coupling function of the edge to value_children[i]
            # (linear when missing), so it is pruned with the children
            fns = node.coupling_fn
            new_edges["coupling_fn"] = tuple(
                fns[i] if i < len(fns) else None for i in keep
            ) or (None,)
        edges_as_list[parent_idx] = node._replace(**new_edges)

    # Update children nodes
    for child_idx in children_idxs:
        if child_idx >= len(edges_as_list):
            continue
        node = edges_as_list[child_idx]
        parents = getattr(node, f"{kind}_parents")
        if not parents:
            continue
        keep = [i for i, parent in enumerate(parents) if parent not in parent_idxs]
        attributes[child_idx] = _keep_strengths(
            attributes[child_idx], f"{kind}_coupling_parents", keep
        )
        edges_as_list[child_idx] = node._replace(**{
            f"{kind}_parents": tuple(parents[i] for i in keep) or None
        })

    return attributes, tuple(edges_as_list)


def remove_node(attributes: dict, edges: Edges, index: int) -> tuple[dict, Edges]:
    """Remove a given node from the network.

    This function removes a node from the network by deleting its parameters in the
    attributes and edges variables, and adjusts the indices of the remaining nodes.
    The attributes passed in are not changed.

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
    tuple[dict, Edges]
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

    # Now remove the node (its attributes are dropped when the indices are adjusted)
    edges_as_list.pop(index)

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
