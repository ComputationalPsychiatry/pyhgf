# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Optional, Union

import jax.numpy as jnp

from pyhgf.typing import Edges


def add_edges(
    attributes: dict,
    edges: Edges,
    kind="value",
    parent_idxs=Union[int, list[int]],
    children_idxs=Union[int, list[int]],
    coupling_strengths: Union[float, list[float], tuple[float, ...]] = 1.0,
    coupling_fn: Optional[tuple[Optional[Callable], ...]] = None,
) -> tuple:
    """Add a value or volatility coupling link between a set of nodes.

    Every parent in `parent_idxs` is linked to every child in `children_idxs`. The
    coupling strength of each new edge is stored at both ends (in the
    ``*_coupling_children`` attribute of the parent and the ``*_coupling_parents``
    attribute of the child). The parent's ``coupling_fn`` holds at least one entry per
    value child, so that ``coupling_fn[i]`` describes the edge to
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
        otherwise. This argument is ignored for volatility coupling and for
        constant-state parents, which are always linearly coupled.
    """
    if kind not in ["value", "volatility"]:
        raise ValueError(
            f"The kind of coupling should be value or volatility, got {kind}"
        )
    parent_idxs = (parent_idxs,) if isinstance(parent_idxs, int) else tuple(parent_idxs)
    children_idxs = (
        (children_idxs,) if isinstance(children_idxs, int) else tuple(children_idxs)
    )
    n_parents, n_children = len(parent_idxs), len(children_idxs)

    # Constant-state nodes can only be parents, never children.
    for children_idx in children_idxs:
        if edges[children_idx].node_type == 0:
            raise ValueError(
                "Constant-state nodes cannot have parents. "
                f"Cannot add {kind} coupling to constant-state node {children_idx}."
            )

    # strengths[i, j] is the strength of the edge from parent_idxs[i] to
    # children_idxs[j]. It is written at both ends of the edge, because the prediction
    # of the child reads it from the child's attributes and the update of the parent
    # from the parent's.
    strengths = jnp.ravel(jnp.asarray(coupling_strengths, dtype=float))
    if strengths.size == 1:
        strengths = jnp.full((n_parents, n_children), strengths[0])
    elif n_parents == 1 and strengths.size == n_children:
        strengths = strengths.reshape(1, n_children)
    elif n_children == 1 and strengths.size == n_parents:
        strengths = strengths.reshape(n_parents, 1)
    else:
        raise ValueError(
            f"Got {strengths.size} coupling strengths for {n_parents} parent(s) and "
            f"{n_children} child(ren). Provide a single value, one value per child "
            "(single parent) or one value per parent (single child)."
        )

    # one coupling function per new value child, with the same rule as add_nodes
    if (kind == "value") and (coupling_fn is not None):
        coupling_fn = tuple(coupling_fn)
        if len(coupling_fn) == 1:
            coupling_fn = n_children * coupling_fn
        elif len(coupling_fn) != n_children:
            raise ValueError(
                "The number of coupling fn and value children do not match"
            )

    edges_as_list = list(edges)

    # update the parent nodes
    # -----------------------
    for i, parent_idx in enumerate(parent_idxs):
        node = edges_as_list[parent_idx]
        children = getattr(node, f"{kind}_children") or ()
        key = f"{kind}_coupling_children"
        attributes[parent_idx][key] = (
            jnp.concatenate([attributes[parent_idx][key], strengths[i]])
            if children
            else strengths[i]
        )
        new_edges = {f"{kind}_children": children + children_idxs}

        if kind == "value":
            # The posterior updates of the parent iterate over
            # zip(value_children, value_coupling_children, coupling_fn), so
            # coupling_fn needs one entry per value child, otherwise the last children
            # are dropped from the update. Missing entries are linear, explicit
            # functions replace the entries of the new children, and constant-state
            # parents stay linear (see add_constant_state).
            n_total = len(children) + n_children
            fns = list(node.coupling_fn) + [None] * (n_total - len(node.coupling_fn))
            if (coupling_fn is not None) and (node.node_type != 0):
                fns[len(children) : n_total] = coupling_fn
            new_edges["coupling_fn"] = tuple(fns)

        edges_as_list[parent_idx] = node._replace(**new_edges)

    # update the children nodes
    # -------------------------
    for j, children_idx in enumerate(children_idxs):
        node = edges_as_list[children_idx]
        parents = getattr(node, f"{kind}_parents") or ()
        key = f"{kind}_coupling_parents"
        attributes[children_idx][key] = (
            jnp.concatenate([attributes[children_idx][key], strengths[:, j]])
            if parents
            else strengths[:, j]
        )
        edges_as_list[children_idx] = node._replace(**{
            f"{kind}_parents": parents + parent_idxs
        })

    return attributes, tuple(edges_as_list)
