# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union


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
    value_parents: Optional[Tuple]
    volatility_parents: Optional[Tuple]
    value_children: Optional[Tuple]
    volatility_children: Optional[Tuple]
    coupling_fn: Tuple[Optional[Callable], ...]


# the nodes' attributes
Attributes = Dict[Union[int, str], Dict]

# the network edges
Edges = Tuple[AdjacencyLists, ...]

# the update sequence
Sequence = Tuple[Tuple[int, Callable[..., Attributes]], ...]
UpdateSequence = Tuple[Sequence, Sequence]

# a fully defined network
NetworkParameters = Tuple[Attributes, Edges, UpdateSequence]
