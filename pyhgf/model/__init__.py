from pyhgf.typing import LayerParams, LayerState
from pyhgf.utils.vectorized_belief_propagation import prediction_pass as predict

from .add_nodes import (
    add_binary_state,
    add_categorical_state,
    add_constant_state,
    add_continuous_state,
    add_dp_state,
    add_ef_state,
    add_volatile_state,
    get_couplings,
    insert_nodes,
    update_parameters,
)
from .deep_network import DeepNetwork
from .network import Network
from .transplant import from_embedding, from_feedforward, from_linear

__all__ = [
    "Network",
    "DeepNetwork",
    "from_linear",
    "from_feedforward",
    "from_embedding",
    "LayerState",
    "LayerParams",
    "predict",
    "add_nodes",
    "add_constant_state",
    "add_continuous_state",
    "add_volatile_state",
    "add_binary_state",
    "add_ef_state",
    "add_categorical_state",
    "add_dp_state",
    "get_couplings",
    "update_parameters",
    "insert_nodes",
]


def __getattr__(name):
    """Raise an informative error when the deprecated `HGF` class is imported."""
    if name == "HGF":
        raise ImportError(
            "The `HGF` class is deprecated and has been removed. Build the network "
            "directly using the `Network` class together with `add_nodes()` instead. "
            "Please refer to the main documentation for examples."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
