# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pyhgf.typing import Edges, Attributes


def set_coupling(
    attributes: Attributes,
    edges: Edges,
    parent_idx: int,
    child_idx: int,
    coupling: float,
) -> Attributes:
    """Set a new coupling strength between a parent and a child node.

    This function ensure that the couplings are updated in both the parent and child
    nodes, while preserving the other coupling values.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    parent_idx :
        Pointer to the parent node.
    child_idx :
        Pointer to the child node.
    coupling :
        The new coupling strength between the parent and child node.

    """
    # 1. update the coupling strength in the attributes dictionary for the child node
    # -------------------------------------------------------------------------------
    parent_couplings = attributes[child_idx]["value_coupling_parents"]

    # find the index of the parent in the child's parent list
    idx = edges[child_idx].value_parents.index(parent_idx)  # type: ignore
    attributes[child_idx]["value_coupling_parents"] = parent_couplings.at[idx].set(
        coupling
    )

    # 2. update the coupling strength in the attributes dictionary for the parent node
    # --------------------------------------------------------------------------------
    children_couplings = attributes[parent_idx]["value_coupling_children"]

    # find the index of the parent in the child's parent list
    idx = edges[parent_idx].value_children.index(child_idx)  # type: ignore
    attributes[parent_idx]["value_coupling_children"] = children_couplings.at[idx].set(
        coupling
    )

    return attributes
