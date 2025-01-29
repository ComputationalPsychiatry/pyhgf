# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, List, Optional, Tuple, Union

from pyhgf.typing import AdjacencyLists, Edges


def add_edges(
    attributes: Dict,
    edges: Edges,
    kind="value",
    parent_idxs=Union[int, List[int]],
    children_idxs=Union[int, List[int]],
    coupling_strengths: Union[float, List[float], Tuple[float]] = 1.0,
    coupling_fn: Tuple[Optional[Callable], ...] = (None,),
) -> Tuple:
    """Add a value or volatility coupling link between a set of nodes.

    Parameters
    ----------
    attributes :
        Attributes of the neural network.
    edges :
        Edges of the neural network.
    kind :
        The kind of coupling can be `"value"` or `"volatility"`.
    parent_idxs :
        The index(es) of the parent node(s).
    children_idxs :
        The index(es) of the children node(s).
    coupling_strengths :
        The coupling strength between the parents and children.
    coupling_fn :
        Coupling function(s) between the current node and its value children.
        It has to be provided as a tuple. If multiple value children are specified,
        the coupling functions must be stated in the same order of the children.
        Note: if a node has multiple parents nodes with different coupling
        functions, a coupling function should be indicated for all the parent nodes.
        If no coupling function is stated, the relationship between nodes is assumed
        linear.

    """
    if kind not in ["value", "volatility"]:
        raise ValueError(
            f"The kind of coupling should be value or volatility, got {kind}"
        )
    if isinstance(children_idxs, int):
        children_idxs = [children_idxs]
    assert isinstance(children_idxs, (list, tuple))

    if isinstance(parent_idxs, int):
        parent_idxs = [parent_idxs]
    assert isinstance(parent_idxs, (list, tuple))

    if isinstance(coupling_strengths, int):
        coupling_strengths = [float(coupling_strengths)]
    if isinstance(coupling_strengths, float):
        coupling_strengths = [coupling_strengths]

    assert isinstance(coupling_strengths, (list, tuple))

    edges_as_list = list(edges)
    # update the parent nodes
    # -----------------------
    for parent_idx in parent_idxs:
        # unpack the parent's edges
        (
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            this_coupling_fn,
        ) = edges_as_list[parent_idx]

        if kind == "value":
            if value_children is None:
                value_children = tuple(children_idxs)
                attributes[parent_idx]["value_coupling_children"] = tuple(
                    coupling_strengths
                )
            else:
                value_children = value_children + tuple(children_idxs)
                attributes[parent_idx]["value_coupling_children"] += tuple(
                    coupling_strengths
                )
                this_coupling_fn = this_coupling_fn + coupling_fn
        elif kind == "volatility":
            if volatility_children is None:
                volatility_children = tuple(children_idxs)
                attributes[parent_idx]["volatility_coupling_children"] = tuple(
                    coupling_strengths
                )
            else:
                volatility_children = volatility_children + tuple(children_idxs)
                attributes[parent_idx]["volatility_coupling_children"] += tuple(
                    coupling_strengths
                )

        # save the updated edges back
        edges_as_list[parent_idx] = AdjacencyLists(
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            this_coupling_fn,
        )

    # update the children nodes
    # -------------------------
    for children_idx in children_idxs:
        # unpack this node's edges
        (
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            coupling_fn,
        ) = edges_as_list[children_idx]

        if kind == "value":
            if value_parents is None:
                value_parents = tuple(parent_idxs)
                attributes[children_idx]["value_coupling_parents"] = tuple(
                    coupling_strengths
                )
            else:
                value_parents = value_parents + tuple(parent_idxs)
                attributes[children_idx]["value_coupling_parents"] += tuple(
                    coupling_strengths
                )
        elif kind == "volatility":
            if volatility_parents is None:
                volatility_parents = tuple(parent_idxs)
                attributes[children_idx]["volatility_coupling_parents"] = tuple(
                    coupling_strengths
                )
            else:
                volatility_parents = volatility_parents + tuple(parent_idxs)
                attributes[children_idx]["volatility_coupling_parents"] += tuple(
                    coupling_strengths
                )

        # save the updated edges back
        edges_as_list[children_idx] = AdjacencyLists(
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            coupling_fn,
        )

    # convert the list back to a tuple
    edges = tuple(edges_as_list)

    return attributes, edges
