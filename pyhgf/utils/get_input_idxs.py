# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Tuple

from pyhgf.typing import Edges


def get_input_idxs(edges: Edges) -> Tuple[int, ...]:
    """List all possible default inputs nodes.

    An input node is a state node without any child.

    Parameters
    ----------
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.

    """
    return tuple(
        [
            i
            for i in range(len(edges))
            if (
                (edges[i].value_children is None)
                & (edges[i].volatility_children is None)
            )
        ]
    )
