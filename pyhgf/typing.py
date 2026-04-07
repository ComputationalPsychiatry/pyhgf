# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

from typing import Callable, NamedTuple, Optional, Union

import jax.numpy as jnp
from jax import Array
from jaxlib.xla_extension import PjitFunction


class AdjacencyLists(NamedTuple):
    """Indexes to a node's value and volatility parents.

    The variable `node_type` encode the type of state node:
    * 0: input node.
    * 1: binary state node.
    * 2: continuous state node.
    * 3: exponential family state node - univariate Gaussian distribution with unknown
        mean and unknown variance.
    * 4: Dirichlet Process state node.

    The variable `coupling_fn` list the coupling functions between this nodes and the
    children nodes. If `None` is provided, a linear coupling is assumed.

    """

    node_type: int
    value_parents: Optional[tuple]
    volatility_parents: Optional[tuple]
    value_children: Optional[tuple]
    volatility_children: Optional[tuple]
    coupling_fn: tuple[Optional[Callable], ...]


# the nodes' attributes
Attributes = dict[Union[int, str], dict]

# the network edges
Edges = tuple[AdjacencyLists, ...]

# the update sequence
Sequence = tuple[tuple[int, PjitFunction], ...]


class UpdateSequence(NamedTuple):
    """Set of update functions to apply to the network."""

    prediction_steps: Sequence
    update_steps: Sequence
    pre_prediction_steps: Optional[Sequence] = None
    post_update_steps: Optional[Sequence] = None
    action_steps: Optional[Sequence] = None


class LearningSequence(NamedTuple):
    """Set of update functions to update the weights of a deep network."""

    prediction_steps: Sequence
    update_steps: Sequence
    learning_steps: Sequence


# a fully defined network
NetworkParameters = tuple[Attributes, Edges, UpdateSequence]


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
            Initialized layer state with zeros for means/errors
            and ones for precisions.
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
        tonic_volatility_vol: float = -4.0,
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
