# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Optional, Union

import jax.numpy as jnp

from pyhgf.typing import AdjacencyLists, Edges


def add_edges(
    attributes: dict,
    edges: Edges,
    kind="value",
    parent_idxs=Union[int, list[int]],
    children_idxs=Union[int, list[int]],
    coupling_strengths: Union[float, list[float], tuple[float]] = 1.0,
    coupling_fn: Optional[tuple[Optional[Callable], ...]] = None,
) -> tuple:
    """Add a value or volatility coupling link between a set of nodes.

    Every parent in `parent_idxs` is linked to every child in `children_idxs`. The
    coupling strength of each new edge is stored at both ends (in the
    ``*_coupling_children`` attribute of the parent and the ``*_coupling_parents``
    attribute of the child), and the parent's ``coupling_fn`` is kept aligned with its
    value children, so that ``coupling_fn[i]`` always describes the edge to
    ``value_children[i]``.

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
        The coupling strength(s) of the new edges. A single value applies to every
        edge. With a single parent, a sequence gives one strength per child, in the
        order of `children_idxs`. With a single child, a sequence gives one strength
        per parent, in the order of `parent_idxs`.
    coupling_fn :
        Coupling function(s) between the parent(s) and the new value children, as a
        tuple. A single function applies to every new child, otherwise one function
        per child must be given, in the order of `children_idxs`. `None` stands for a
        linear coupling. If `coupling_fn` is not provided, the function already
        declared on the parent at that position (e.g. through
        ``add_nodes(coupling_fn=...)``) is kept, and the coupling is linear
        otherwise. This argument is ignored for volatility coupling.
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

    # Constant-state nodes can only be parents, never children.
    for children_idx in children_idxs:
        if edges[children_idx].node_type == 0:
            raise ValueError(
                "Constant-state nodes cannot have parents. "
                f"Cannot add {kind} coupling to constant-state node {children_idx}."
            )

    # One coupling strength per (parent, child) edge: strengths[i][j] is the strength
    # of the edge from parent_idxs[i] to children_idxs[j]. The same value is written
    # at both ends of the edge, because the prediction of the child reads it from
    # the child's attributes and the update of the parent from the parent's.
    n_parents, n_children = len(parent_idxs), len(children_idxs)
    if len(coupling_strengths) == 1:
        strengths = [[coupling_strengths[0]] * n_children for _ in range(n_parents)]
    elif n_parents == 1 and len(coupling_strengths) == n_children:
        strengths = [list(coupling_strengths)]
    elif n_children == 1 and len(coupling_strengths) == n_parents:
        strengths = [[strength] for strength in coupling_strengths]
    else:
        raise ValueError(
            f"Got {len(coupling_strengths)} coupling strengths for {n_parents} "
            f"parent(s) and {n_children} child(ren). Provide a single value, one "
            "value per child (single parent) or one value per parent (single child)."
        )

    # one coupling function per new value child, with the same rule as add_nodes
    if (kind == "value") and (coupling_fn is not None):
        coupling_fn = tuple(coupling_fn)
        if len(coupling_fn) == 1:
            coupling_fn = len(children_idxs) * coupling_fn
        elif len(coupling_fn) != len(children_idxs):
            raise ValueError(
                "The number of coupling fn and value children do not match"
            )

    edges_as_list = list(edges)
    # update the parent nodes
    # -----------------------
    for i, parent_idx in enumerate(parent_idxs):
        # unpack the parent's edges
        (
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            this_coupling_fn,
        ) = edges_as_list[parent_idx]

        new_strengths = jnp.array(strengths[i])

        if kind == "value":
            n_existing = 0 if value_children is None else len(value_children)
            if value_children is None:
                value_children = tuple(children_idxs)
                attributes[parent_idx]["value_coupling_children"] = new_strengths
            else:
                value_children = value_children + tuple(children_idxs)
                attributes[parent_idx]["value_coupling_children"] = jnp.concatenate([
                    attributes[parent_idx]["value_coupling_children"],
                    new_strengths,
                ])

            # The posterior updates of the parent iterate over
            # zip(value_children, value_coupling_children, coupling_fn), and the
            # prediction of a child reads coupling_fn[child_position]: coupling_fn
            # needs one entry per value child, otherwise the last children are
            # silently dropped from the parent's update. Missing entries default to
            # a linear coupling, and explicit functions replace the entries of the
            # new children.
            n_total = len(value_children)
            fns = list(this_coupling_fn)
            fns += [None] * max(0, n_total - len(fns))
            if coupling_fn is not None:
                fns[n_existing:n_total] = coupling_fn
            this_coupling_fn = tuple(fns)

        elif kind == "volatility":
            if volatility_children is None:
                volatility_children = tuple(children_idxs)
                attributes[parent_idx]["volatility_coupling_children"] = new_strengths
            else:
                volatility_children = volatility_children + tuple(children_idxs)
                attributes[parent_idx]["volatility_coupling_children"] = (
                    jnp.concatenate([
                        attributes[parent_idx]["volatility_coupling_children"],
                        new_strengths,
                    ])
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
    for j, children_idx in enumerate(children_idxs):
        # unpack this node's edges
        (
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            child_coupling_fn,
        ) = edges_as_list[children_idx]

        new_strengths = jnp.array([strengths[i][j] for i in range(n_parents)])

        if kind == "value":
            if value_parents is None:
                value_parents = tuple(parent_idxs)
                attributes[children_idx]["value_coupling_parents"] = new_strengths
            else:
                value_parents = value_parents + tuple(parent_idxs)
                attributes[children_idx]["value_coupling_parents"] = jnp.concatenate([
                    attributes[children_idx]["value_coupling_parents"],
                    new_strengths,
                ])
        elif kind == "volatility":
            if volatility_parents is None:
                volatility_parents = tuple(parent_idxs)
                attributes[children_idx]["volatility_coupling_parents"] = new_strengths
            else:
                volatility_parents = volatility_parents + tuple(parent_idxs)
                attributes[children_idx]["volatility_coupling_parents"] = (
                    jnp.concatenate([
                        attributes[children_idx]["volatility_coupling_parents"],
                        new_strengths,
                    ])
                )

        # save the updated edges back
        edges_as_list[children_idx] = AdjacencyLists(
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            child_coupling_fn,
        )

    # convert the list back to a tuple
    edges = tuple(edges_as_list)

    return attributes, edges
