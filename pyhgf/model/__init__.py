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
from .builder import LayerConfig, resolve_coupling_fn
from .deep_network import DeepNetwork
from .fused import FusedPipeline, step_report
from .hybrid import (
    DeepNetworkAdapter,
    EquinoxAdapter,
    PCModule,
    PCSequential,
    Residual,
    gelu_adapter,
    layer_norm_adapter,
    linear_adapter,
)
from .network import Network
from .transformer import HybridGPT, MultiHeadAttention, hybrid_from_gpt
from .transplant import from_embedding, from_feedforward, from_linear

__all__ = [
    "Network",
    "DeepNetwork",
    "LayerConfig",
    "resolve_coupling_fn",
    "from_linear",
    "from_feedforward",
    "from_embedding",
    "PCModule",
    "EquinoxAdapter",
    "DeepNetworkAdapter",
    "PCSequential",
    "Residual",
    "gelu_adapter",
    "layer_norm_adapter",
    "linear_adapter",
    "step_report",
    "FusedPipeline",
    "MultiHeadAttention",
    "HybridGPT",
    "hybrid_from_gpt",
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
