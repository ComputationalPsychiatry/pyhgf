# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Carry trained Equinox weights into equivalent PyHGF networks.

These converters build a :class:`~pyhgf.model.DeepNetwork` that computes exactly the
same function as a given Equinox module, with the module's weights placed in PyHGF's
layout.

The layout differences the converters absorb:

- **Layer order.** PyHGF stacks layers input-on-top, output-at-bottom; the
  weight matrix connecting two layers is stored on the upper (parent) layer.
  The matrix orientation itself matches Equinox: shape
  ``(out_features, in_features)``.
- **Biases.** Equinox keeps a bias vector next to each matrix; PyHGF folds
  it into the matrix as an extra last column, wired to a constant always-on
  node.
- **Bias placement across a nonlinearity.** PyHGF applies a coupling
  function when a layer predicts the one *below* it, so the bias that
  Equinox adds *before* an activation (``fc1.bias`` in
  ``fc2(gelu(fc1(x)))``) belongs to the input→hidden matrix, one connection
  above where the activation acts.
- **Embeddings.** A table lookup equals a one-hot vector times the table,
  so an embedding becomes a linear network fed one-hot inputs, with the
  table transposed to PyHGF's ``(out, in)`` orientation.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from pyhgf.model.deep_network import DeepNetwork

__all__ = [
    "from_linear",
    "from_feedforward",
    "from_embedding",
]


def _weights_with_bias(weight: jnp.ndarray, bias: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Fold an optional bias vector into the matrix as its last column."""
    if bias is None:
        return jnp.asarray(weight)
    return jnp.concatenate([jnp.asarray(weight), jnp.asarray(bias)[:, None]], axis=1)


def _set_weights(net: DeepNetwork, weights: dict) -> DeepNetwork:
    """Replace ``weights_in`` on the given layer indices."""
    if net.state is None:
        raise ValueError("Add at least one layer before setting weights.")
    elements = list(net.state.layers)
    for i, w in weights.items():
        elements[i] = dataclasses.replace(elements[i], weights_in=jnp.asarray(w))
    net.state = dataclasses.replace(net.state, layers=tuple(elements))
    return net


def from_linear(
    linear: eqx.nn.Linear,
    leaf_kwargs: Optional[dict] = None,
    layer_kwargs: Optional[dict] = None,
) -> DeepNetwork:
    """Build a two-layer network computing exactly ``linear(x)``.

    The bottom (output) layer has ``out_features`` nodes, the top (input) layer
    ``in_features`` nodes, and the connecting matrix carries the Linear's weight with
    the bias folded in as the last column (if the Linear has one).

    Parameters
    ----------
    linear :
        The Equinox layer whose weights are transplanted.
    leaf_kwargs :
        Extra ``add_layer`` keyword arguments for the bottom (observed)
        layer — e.g. the backprop-parity configuration.
    layer_kwargs :
        Extra ``add_layer`` keyword arguments for the top (input) layer.

    Returns
    -------
    DeepNetwork
        A network whose ``predict`` reproduces the Linear's forward pass.
    """
    out_features, in_features = linear.weight.shape
    net = (
        DeepNetwork()
        .add_layer(size=out_features, add_constant_input=False, **(leaf_kwargs or {}))
        .add_layer(
            size=in_features,
            add_constant_input=linear.bias is not None,
            **(layer_kwargs or {}),
        )
    )
    return _set_weights(net, {1: _weights_with_bias(linear.weight, linear.bias)})


def from_feedforward(
    fc1: eqx.nn.Linear,
    fc2: eqx.nn.Linear,
    activation: Callable = jax.nn.gelu,
    leaf_kwargs: Optional[dict] = None,
    layer_kwargs: Optional[dict] = None,
) -> DeepNetwork:
    """Build a three-layer network computing exactly ``fc2(activation(fc1(x)))``.

    The hidden layer holds the pre-activation values and applies ``activation`` as its
    coupling when predicting the output layer, which is why ``fc1``'s bias lands on the
    input→hidden matrix (the hidden node's value is ``fc1.weight @ x + fc1.bias``, and
    the activation acts on the whole of it one connection below), while ``fc2``'s bias
    lands on the hidden → output matrix, outside the activation.

    Parameters
    ----------
    fc1 :
        The first Equinox layer (input → hidden, inside the activation).
    fc2 :
        The second Equinox layer (hidden → output).
    activation :
        The nonlinearity between them (default GELU, matching the Transformer
        feed-forward block).
    leaf_kwargs :
        Extra ``add_layer`` keyword arguments for the bottom (observed) layer.
    layer_kwargs :
        Extra ``add_layer`` keyword arguments for the hidden and input layers — e.g. the
        backprop-parity configuration.

    Returns
    -------
    DeepNetwork
        A network whose ``predict`` reproduces the feed-forward block.
    """
    hidden, in_features = fc1.weight.shape
    out_features = fc2.weight.shape[0]
    net = (
        DeepNetwork()
        .add_layer(size=out_features, add_constant_input=False, **(leaf_kwargs or {}))
        .add_layer(
            size=hidden,
            add_constant_input=fc2.bias is not None,
            coupling_fn=activation,
            **(layer_kwargs or {}),
        )
        .add_layer(
            size=in_features,
            add_constant_input=fc1.bias is not None,
            **(layer_kwargs or {}),
        )
    )
    return _set_weights(
        net,
        {
            1: _weights_with_bias(fc2.weight, fc2.bias),
            2: _weights_with_bias(fc1.weight, fc1.bias),
        },
    )


def from_embedding(
    embedding: eqx.nn.Embedding,
    leaf_kwargs: Optional[dict] = None,
    layer_kwargs: Optional[dict] = None,
) -> DeepNetwork:
    """Build a two-layer network reproducing a table lookup from one-hot inputs.

    A lookup is a matrix product with a one-hot vector: ``table[i]`` equals
    ``one_hot(i) @ table``. The top (input) layer therefore has one node per
    table row, the bottom (output) layer one node per embedding dimension, and the
    connecting matrix is the table transposed into PyHGF's ``(out, in)`` orientation.
    Feed it one-hot rows: ``net.predict(jax.nn.one_hot(ids, num_embeddings))``.

    Parameters
    ----------
    embedding :
        The Equinox embedding whose table is transplanted.
    leaf_kwargs :
        Extra ``add_layer`` keyword arguments for the bottom (observed)
        layer.
    layer_kwargs :
        Extra ``add_layer`` keyword arguments for the top (input) layer.

    Returns
    -------
    DeepNetwork
        A network whose ``predict`` on one-hot inputs reproduces the lookup.
    """
    num_embeddings, embedding_size = embedding.weight.shape
    net = (
        DeepNetwork()
        .add_layer(size=embedding_size, add_constant_input=False, **(leaf_kwargs or {}))
        .add_layer(
            size=num_embeddings, add_constant_input=False, **(layer_kwargs or {})
        )
    )
    return _set_weights(net, {1: jnp.asarray(embedding.weight).T})
