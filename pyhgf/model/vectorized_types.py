# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Data structures for vectorized deep predictive coding networks.

This module defines JAX-compatible NamedTuple types for layer-wise
vectorized operations in deep HGF networks.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class LayerState(NamedTuple):
    """State for all nodes in a layer. All arrays have shape (n_nodes,).

    This represents the state of a volatile node layer with both value level (external)
    and volatility level (internal) variables.
    """

    # Value level (external)
    mean: Array
    precision: Array
    expected_mean: Array
    expected_precision: Array
    effective_precision: Array
    value_prediction_error: Array

    # Volatility level (internal) - for volatile nodes
    mean_vol: Array
    precision_vol: Array
    expected_mean_vol: Array
    expected_precision_vol: Array
    effective_precision_vol: Array
    volatility_prediction_error: Array

    # Observation mask
    observed: Array  # int array

    @classmethod
    def create(cls, n_nodes: int) -> "LayerState":
        """Create a LayerState with default initialization.

        Parameters
        ----------
        n_nodes :
            Number of nodes in the layer.

        Returns
        -------
        LayerState
            Initialized layer state with zeros for means/errors and ones for precisions.
        """
        return cls(
            # Value level
            mean=jnp.zeros(n_nodes),
            precision=jnp.ones(n_nodes),
            expected_mean=jnp.zeros(n_nodes),
            expected_precision=jnp.ones(n_nodes),
            effective_precision=jnp.zeros(n_nodes),
            value_prediction_error=jnp.zeros(n_nodes),
            # Volatility level
            mean_vol=jnp.zeros(n_nodes),
            precision_vol=jnp.ones(n_nodes),
            expected_mean_vol=jnp.zeros(n_nodes),
            expected_precision_vol=jnp.ones(n_nodes),
            effective_precision_vol=jnp.zeros(n_nodes),
            volatility_prediction_error=jnp.zeros(n_nodes),
            # Observation mask
            observed=jnp.ones(n_nodes, dtype=jnp.int32),
        )


class LayerParams(NamedTuple):
    """Static parameters for a layer. All arrays have shape (n_nodes,).

    These parameters control the volatility dynamics of the layer.
    """

    tonic_volatility: Array  # Value level tonic volatility
    tonic_volatility_vol: Array  # Volatility level tonic volatility
    volatility_coupling: Array  # Internal volatility coupling strength

    @classmethod
    def create(
        cls,
        n_nodes: int,
        tonic_volatility: float = -4.0,
        tonic_volatility_vol: float = -2.0,
        volatility_coupling: float = 1.0,
    ) -> "LayerParams":
        """Create LayerParams with specified values.

        Parameters
        ----------
        n_nodes :
            Number of nodes in the layer.
        tonic_volatility :
            Value level tonic volatility (log scale).
        tonic_volatility_vol :
            Volatility level tonic volatility (log scale).
        volatility_coupling :
            Internal volatility coupling strength.

        Returns
        -------
        LayerParams
            Initialized layer parameters.
        """
        return cls(
            tonic_volatility=jnp.full(n_nodes, tonic_volatility),
            tonic_volatility_vol=jnp.full(n_nodes, tonic_volatility_vol),
            volatility_coupling=jnp.full(n_nodes, volatility_coupling),
        )


class NetworkState(NamedTuple):
    """Complete network state.

    This represents the full state of a vectorized deep network, including all layer
    states, inter-layer weights, and parameters.
    """

    layers: tuple  # tuple[LayerState, ...] - Layer 0 = output, Layer N = input
    weights: tuple  # tuple[Array, ...] - weights[i] connects layer[i] to layer[i+1]
    params: tuple  # tuple[LayerParams, ...] - params[i] for layer[i]
    time_step: float

    @property
    def n_layers(self) -> int:
        """Number of layers in the network."""
        return len(self.layers)

    def get_layer_sizes(self) -> list:
        """Get the size of each layer."""
        return [layer.mean.shape[0] for layer in self.layers]
