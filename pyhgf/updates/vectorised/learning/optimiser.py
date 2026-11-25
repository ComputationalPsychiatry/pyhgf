# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Optimiser step of the gradient learning kinds.

The kinds ``"standard"`` and ``"precision_weighted"`` hold no state of their own: each
returns a descent gradient, and one optax optimiser step moves every weight matrix along
it (:func:`_apply_weight_updates`).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import optax

from pyhgf.typing.vectorised import VectorisedNetwork
from pyhgf.updates.vectorised.elements import _edge_mask


def _apply_weight_updates(
    network: VectorisedNetwork,
    grads: tuple,
    opt_state: optax.OptState,
    optimiser: optax.GradientTransformation,
) -> tuple[VectorisedNetwork, optax.OptState]:
    """One optimiser step on every ``weights_mean``, from precomputed gradients.

    In a one-to-one element, only the diagonal weights move. The gradient is masked
    before the optimiser, so its state and any transform that reads the whole gradient
    (such as ``optax.clip_by_global_norm``) see only the edges. The update is masked
    too, since a transform that does not scale with the gradient (such as
    ``optax.add_noise``) would still move the other weights.
    """
    elements = list(network.layers)
    weights = tuple(elem.weights_mean for elem in elements)
    masks = tuple(_edge_mask(elem) for elem in elements)
    grads = tuple(
        g if m is None else jnp.where(m, g, 0.0) for g, m in zip(grads, masks)
    )

    updates, new_opt_state = optimiser.update(grads, opt_state, weights)
    new_weights = optax.apply_updates(weights, updates)
    for i, new_w in enumerate(new_weights):
        if new_w is not None:
            if masks[i] is not None:
                new_w = jnp.where(masks[i], new_w, weights[i])
            elements[i] = dataclasses.replace(elements[i], weights_mean=new_w)

    return dataclasses.replace(network, layers=tuple(elements)), new_opt_state
