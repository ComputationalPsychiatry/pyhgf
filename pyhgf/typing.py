# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, NamedTuple, Optional, Union

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
    """Set of update functions to apply to the network."""

    prediction_steps: Sequence
    update_steps: Sequence
    learning_steps: Sequence


# a fully defined network
NetworkParameters = tuple[Attributes, Edges, UpdateSequence]
