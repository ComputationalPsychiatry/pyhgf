# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Configuration and factory functions for building DeepNetworks declaratively.

Provides dataclasses and utilities for constructing networks from configuration
dictionaries (e.g., JSON, YAML) or lists of layer configs, enabling reproducible
hyperparameter sweeps, model zoos, and configuration-driven experiments.

Example:
    >>> from pyhgf.model import LayerConfig, DeepNetwork
    >>> config = {
    ...     "layers": [
    ...         {"size": 10},
    ...         {"size": 20, "coupling_fn": "gelu"},
    ...         {"size": 15}
    ...     ],
    ...     "volatility_updates": "unbounded"
    ... }
    >>> net = DeepNetwork.from_dict(config)
    >>> net.n_layers
    3
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Optional

import jax.nn


@dataclass
class LayerConfig:
    """Configuration for one layer in a DeepNetwork.

    Encapsulates all per-layer settings so they can be serialized to/from JSON, YAML,
    or other formats. Each field corresponds to a parameter of
    :meth:`DeepNetwork.add_layer`.

    Attributes
    ----------
    size : int
        Number of nodes in the layer.
    kind : str, default "volatile"
        Type of nodes: "volatile", "binary", or "categorical".
    add_constant_input : bool, default True
        Whether to add a bias term to the layer's predictions.
    fully_connected : bool, default True
        Whether the layer is fully connected (dense) or one-to-one.
    coupling_fn : Optional[str], default None
        Name of coupling function ("identity", "gelu", "relu", etc.) or None to use the
        network-level default. Must be a name that can be resolved via
        :func:`resolve_coupling_fn`.
    volatility_parent : bool, default True
        Whether the layer has an internal volatility parent.
    tonic_volatility_vol : Optional[float], default None
        Per-layer override for tonic_volatility_vol parameter.

    Notes
    -----
    String coupling function names (e.g., "gelu", "relu") are resolved at network build
    time. Use None to inherit the network-level coupling function.
    """

    size: int
    kind: str = "volatile"
    add_constant_input: bool = True
    fully_connected: bool = True
    coupling_fn: Optional[str] = None
    volatility_parent: bool = True
    tonic_volatility_vol: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        Excludes None values for cleaner configs. Use this when serializing to JSON or
        YAML to avoid cluttering the output.

        Returns
        -------
        dict[str, Any]
            Dictionary with only non-None fields.
        """
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerConfig:
        """Construct from a dictionary.

        Provides flexibility: missing keys get their default values, and extra keys are
        ignored (allowing extension without breaking).

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with layer configuration.

        Returns
        -------
        LayerConfig
            Configured layer.
        """
        # Keep only recognised fields; extra keys are ignored so configs can
        # carry annotations without breaking construction.
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def resolve_coupling_fn(name_or_fn: Optional[str | Callable]) -> Optional[Callable]:
    """Resolve a coupling function name to a callable.

    Parameters
    ----------
    name_or_fn : Optional[str | Callable]
        Either a string name ("gelu", "relu", "identity", etc.), a callable, or None
        (returns None).

    Returns
    -------
    Optional[Callable]
        The resolved function, or None if input is None.

    Raises
    ------
    ValueError
        If the name is not recognized.

    Examples
    --------
    >>> fn = resolve_coupling_fn("gelu")
    >>> fn is jax.nn.gelu
    True
    >>> resolve_coupling_fn(None) is None
    True
    >>> resolve_coupling_fn(lambda x: x)  # Already callable
    <function <lambda> at ...>
    """
    if name_or_fn is None:
        return None

    if callable(name_or_fn):
        return name_or_fn

    if isinstance(name_or_fn, str):
        mapping = {
            "identity": lambda x: x,
            "gelu": jax.nn.gelu,
            "relu": jax.nn.relu,
            "tanh": jax.nn.tanh,
            "sigmoid": jax.nn.sigmoid,
            "elu": jax.nn.elu,
            "silu": jax.nn.silu,
            "swish": jax.nn.swish,
        }
        if name_or_fn not in mapping:
            raise ValueError(
                f"Unknown coupling function: '{name_or_fn}'. "
                f"Valid names: {sorted(mapping.keys())}. "
                f"Alternatively, pass a callable directly."
            )
        return mapping[name_or_fn]

    raise TypeError(
        f"coupling_fn must be None, a string name, or a callable; "
        f"got {type(name_or_fn).__name__}"
    )
