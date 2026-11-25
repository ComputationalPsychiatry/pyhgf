# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Uniform views of the elements of a vectorised network.

A vectorised network is a chain of elements, each a ``Layer`` or a ``LayerStack``. These
helpers read what the belief propagation sweeps and the weight learning need from either
kind, so that neither has to branch on the element type.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from pyhgf.typing.vectorised import LayerStack, VectorisedNetwork

#: Floor applied to precisions before any division.
_EPS = 1e-30


def _stack_slice(stack: LayerStack, index: int):
    """Return ``(state, params, weights_mean)`` of the stack slice at ``index``.

    Index ``0`` is the bottommost slice, ``-1`` the topmost.
    """
    state = jax.tree_util.tree_map(lambda x: x[index], stack.state)
    params = jax.tree_util.tree_map(lambda x: x[index], stack.params)
    return state, params, stack.weights_mean[index]


def _child_view(elem):
    """Treat a ``Layer`` or ``LayerStack`` uniformly when acting as a child.

    Returns ``(state, kind, is_input_layer)``. What's needed when something above is
    doing a posterior update or computing prediction-error-driven weight gradients using
    this element's state as the child.

    For a ``LayerStack``, the child role is filled by the *topmost* slice (the slice
    closest to the parent above the stack).
    """
    if isinstance(elem, LayerStack):
        state, _, _ = _stack_slice(elem, -1)
        return state, elem.kind, False  # interior; never the clamped leaf
    return elem.state, elem.kind, elem.is_input_layer


def _sweeps_reach_top(network: VectorisedNetwork) -> bool:
    """Whether both sweeps should treat the top element as a member of the hierarchy.

    Gating the two sweeps on one predicate keeps them in step: a top element that
    receives a posterior update must also receive the precision prediction that
    carries it into the next step, or its precision would accumulate with nothing
    advancing it.

    Requires ``update_input_layer``, something below the top to send it a message,
    and a volatile top element. A binary layer's predicted precision is a function
    of its predicted mean, which is clamped to the predictors here, so there is no
    precision of its own for the sweeps to move.
    """
    return (
        network.update_input_layer
        and len(network.layers) > 1
        and network.layers[-1].kind == "volatile"
    )


def _edge_mask(elem) -> Optional[jnp.ndarray]:
    """Boolean mask of the weights that are edges, or ``None`` when all of them are.

    A one-to-one element (``fully_connected=False``) connects node :math:`i` of the
    child layer to node :math:`i` of this layer only, so only the diagonal of its
    ``weights_mean`` is an edge. A ``LayerStack`` mask broadcasts over the slice axis.
    """
    if elem.fully_connected or elem.weights_mean is None:
        return None
    n_children, n_parents = elem.weights_mean.shape[-2:]
    return jnp.eye(n_children, n_parents, dtype=bool)
