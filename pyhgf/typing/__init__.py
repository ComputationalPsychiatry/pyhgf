# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Type definitions for pyhgf.

The package exposes two parallel groups of public types:

* **Nodalised types** powering :class:`pyhgf.model.Network` and the per-node
  belief-propagation kernels. Defined in :mod:`pyhgf.typing.typing`.
* **Vectorised Equinox types** powering :class:`pyhgf.model.DeepNetwork`.
  Defined in :mod:`pyhgf.typing.vectorised`.
"""

from pyhgf.typing.typing import (
    AdjacencyLists,
    Attributes,
    Edges,
    LearningSequence,
    NetworkParameters,
    Sequence,
    UpdateSequence,
)
from pyhgf.typing.vectorised import (
    RECORD_ALL,
    Layer,
    LayerParams,
    LayerStack,
    LayerState,
    Network,
    stack_layers,
)

__all__ = [
    # Nodalised types (pyhgf.model.Network).
    "AdjacencyLists",
    "Attributes",
    "Edges",
    "LearningSequence",
    "NetworkParameters",
    "Sequence",
    "UpdateSequence",
    # Vectorised Equinox PyTree types (pyhgf.model.DeepNetwork).
    "Layer",
    "LayerParams",
    "LayerStack",
    "LayerState",
    "Network",
    "RECORD_ALL",
    "stack_layers",
]
