# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, NamedTuple, Optional, Union


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
Sequence = tuple[tuple[int, Callable[..., Attributes]], ...]
UpdateSequence = tuple[Sequence, Sequence]

# a fully defined network
NetworkParameters = tuple[Attributes, Edges, UpdateSequence]
