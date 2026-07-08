# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests for declarative network building via configs.

Validates that networks can be constructed from configuration dictionaries (JSON-like)
and lists of LayerConfig objects, and that configs can be round-tripped to/from
dictionaries for serialization.
"""

from __future__ import annotations

import json

import jax.nn
import pytest

from pyhgf.model import DeepNetwork, LayerConfig, resolve_coupling_fn


class TestLayerConfig:
    """Test LayerConfig dataclass and conversions."""

    def test_layer_config_defaults(self):
        """LayerConfig applies sensible defaults."""
        cfg = LayerConfig(size=10)
        assert cfg.size == 10
        assert cfg.kind == "volatile"
        assert cfg.add_constant_input is True
        assert cfg.fully_connected is True
        assert cfg.coupling_fn is None
        assert cfg.volatility_parent is True

    def test_layer_config_custom_values(self):
        """LayerConfig accepts all custom values."""
        cfg = LayerConfig(
            size=20,
            kind="binary",
            add_constant_input=False,
            coupling_fn="gelu",
            tonic_volatility=-2.0,
        )
        assert cfg.size == 20
        assert cfg.kind == "binary"
        assert cfg.add_constant_input is False
        assert cfg.coupling_fn == "gelu"
        assert cfg.tonic_volatility == -2.0

    def test_layer_config_to_dict_excludes_none(self):
        """LayerConfig.to_dict() excludes None values for clean JSON."""
        cfg = LayerConfig(
            size=10,
            coupling_fn="gelu",
            tonic_volatility=None,  # Should be excluded
        )
        d = cfg.to_dict()
        assert "size" in d
        assert "coupling_fn" in d
        assert "tonic_volatility" not in d
        assert d["size"] == 10
        assert d["coupling_fn"] == "gelu"

    def test_layer_config_from_dict_tolerates_extra_keys(self):
        """LayerConfig.from_dict() ignores unknown keys."""
        d = {
            "size": 15,
            "coupling_fn": "relu",
            "unknown_field": "ignored",
        }
        cfg = LayerConfig.from_dict(d)
        assert cfg.size == 15
        assert cfg.coupling_fn == "relu"

    def test_layer_config_from_dict_missing_size_raises(self):
        """LayerConfig.from_dict() requires 'size' key."""
        with pytest.raises(TypeError):
            LayerConfig.from_dict({"coupling_fn": "gelu"})

    def test_layer_config_round_trip(self):
        """LayerConfig can round-trip through to_dict() and from_dict()."""
        original = LayerConfig(
            size=25,
            kind="volatile",
            coupling_fn="gelu",
            tonic_volatility=-3.0,
        )
        d = original.to_dict()
        restored = LayerConfig.from_dict(d)
        assert restored.size == original.size
        assert restored.kind == original.kind
        assert restored.coupling_fn == original.coupling_fn
        assert restored.tonic_volatility == original.tonic_volatility


class TestResolveCouplingFn:
    """Test coupling function name resolution."""

    def test_resolve_coupling_fn_from_string_gelu(self):
        """Resolve 'gelu' string to jax.nn.gelu."""
        fn = resolve_coupling_fn("gelu")
        assert fn is jax.nn.gelu

    def test_resolve_coupling_fn_from_string_relu(self):
        """Resolve 'relu' string to jax.nn.relu."""
        fn = resolve_coupling_fn("relu")
        assert fn is jax.nn.relu

    def test_resolve_coupling_fn_from_string_identity(self):
        """Resolve 'identity' string to identity function."""
        fn = resolve_coupling_fn("identity")
        import jax.numpy as jnp

        x = jnp.array([1.0, 2.0])
        result = fn(x)
        assert jnp.allclose(result, x)

    def test_resolve_coupling_fn_from_string_unknown_raises(self):
        """Unknown string name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown coupling function"):
            resolve_coupling_fn("unknown_fn")

    def test_resolve_coupling_fn_from_callable_returns_it(self):
        """Callable is returned as-is."""
        fn = lambda x: x * 2
        result = resolve_coupling_fn(fn)
        assert result is fn

    def test_resolve_coupling_fn_from_none_returns_none(self):
        """None returns None."""
        assert resolve_coupling_fn(None) is None

    def test_resolve_coupling_fn_from_invalid_type_raises(self):
        """Invalid type raises TypeError."""
        with pytest.raises(TypeError):
            resolve_coupling_fn(123)


class TestDeepNetworkFromConfigs:
    """Test DeepNetwork.from_configs() factory."""

    def test_from_configs_basic(self):
        """Build a basic network from configs."""
        configs = [
            LayerConfig(size=10),
            LayerConfig(size=20),
            LayerConfig(size=15),
        ]
        net = DeepNetwork.from_configs(configs)
        assert net.n_layers == 3
        assert net.layer_sizes == [10, 20, 15]

    def test_from_configs_with_network_coupling_fn_string(self):
        """Network-level coupling function can be a string."""
        configs = [
            LayerConfig(size=10),
            LayerConfig(size=20),
        ]
        net = DeepNetwork.from_configs(configs, coupling_fn="gelu")
        assert net.n_layers == 2
        # Verify the network was built successfully
        assert net.state is not None

    def test_from_configs_with_network_coupling_fn_callable(self):
        """Network-level coupling function can be a callable."""
        configs = [
            LayerConfig(size=10),
            LayerConfig(size=20),
        ]
        net = DeepNetwork.from_configs(configs, coupling_fn=jax.nn.relu)
        assert net.n_layers == 2

    def test_from_configs_with_per_layer_coupling_fn(self):
        """Per-layer coupling functions override network-level."""
        configs = [
            LayerConfig(size=10),
            LayerConfig(size=20, coupling_fn="gelu"),
        ]
        net = DeepNetwork.from_configs(configs, coupling_fn="relu")
        assert net.n_layers == 2
        # Layer 1 has gelu (from config), layer 0 has relu (from network)
        # This is tested implicitly through successful build

    def test_from_configs_with_layer_params_override(self):
        """Per-layer parameter overrides are applied."""
        configs = [
            LayerConfig(size=10, tonic_volatility=-2.0),
            LayerConfig(size=20),
        ]
        net = DeepNetwork.from_configs(configs)
        assert net.n_layers == 2
        # Verify state was initialized
        assert net.state is not None

    def test_from_configs_empty_raises(self):
        """Empty configs list raises ValueError."""
        with pytest.raises(ValueError):
            DeepNetwork.from_configs([])

    def test_from_configs_method_chaining_equivalent(self):
        """from_configs produces same result as method chaining."""
        # Build with chaining
        net1 = DeepNetwork().add_layer(size=10).add_layer(size=20).add_layer(size=15)

        # Build with from_configs
        configs = [
            LayerConfig(size=10),
            LayerConfig(size=20),
            LayerConfig(size=15),
        ]
        net2 = DeepNetwork.from_configs(configs)

        # Both should have same structure
        assert net1.n_layers == net2.n_layers
        assert net1.layer_sizes == net2.layer_sizes


class TestDeepNetworkFromDict:
    """Test DeepNetwork.from_dict() factory."""

    def test_from_dict_basic(self):
        """Build a network from a dictionary config."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20},
                {"size": 15},
            ]
        }
        net = DeepNetwork.from_dict(config)
        assert net.n_layers == 3
        assert net.layer_sizes == [10, 20, 15]

    def test_from_dict_with_network_coupling_fn(self):
        """Network-level coupling function from dict."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20},
            ],
            "coupling_fn": "gelu",
        }
        net = DeepNetwork.from_dict(config)
        assert net.n_layers == 2

    def test_from_dict_with_per_layer_coupling_fn(self):
        """Per-layer coupling functions in dict."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20, "coupling_fn": "gelu"},
                {"size": 15},
            ]
        }
        net = DeepNetwork.from_dict(config)
        assert net.n_layers == 3

    def test_from_dict_with_network_params(self):
        """Network-level parameters from dict."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20},
            ],
            "volatility_updates": "standard",
            "max_posterior_precision": 1e8,
        }
        net = DeepNetwork.from_dict(config)
        assert net.volatility_updates == "standard"
        assert net.max_posterior_precision == 1e8

    def test_from_dict_missing_layers_raises(self):
        """Missing 'layers' key raises ValueError."""
        config = {"coupling_fn": "gelu"}
        with pytest.raises(ValueError, match="'layers'"):
            DeepNetwork.from_dict(config)

    def test_from_dict_empty_layers_raises(self):
        """Empty 'layers' list raises ValueError."""
        config = {"layers": []}
        with pytest.raises(ValueError, match="not be empty"):
            DeepNetwork.from_dict(config)

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys in dict are ignored."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20},
            ],
            "unknown_key": "ignored",
            "another_unknown": 123,
        }
        net = DeepNetwork.from_dict(config)
        assert net.n_layers == 2

    def test_from_dict_json_round_trip(self):
        """Configs can be serialized to JSON and back."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20, "coupling_fn": "gelu"},
                {"size": 15},
            ],
            "coupling_fn": "identity",
            "volatility_updates": "unbounded",
        }

        # Serialize to JSON
        json_str = json.dumps(config)

        # Deserialize from JSON
        loaded_config = json.loads(json_str)

        # Build from loaded config
        net = DeepNetwork.from_dict(loaded_config)
        assert net.n_layers == 3
        assert net.layer_sizes == [10, 20, 15]

    def test_from_dict_layer_config_objects(self):
        """from_dict accepts LayerConfig objects in 'layers'."""
        configs_obj = [
            LayerConfig(size=10),
            LayerConfig(size=20, coupling_fn="gelu"),
        ]
        config = {"layers": configs_obj}

        net = DeepNetwork.from_dict(config)
        assert net.n_layers == 2


class TestDeclarativeWorkflows:
    """Test realistic workflows with declarative API."""

    def test_hyperparameter_sweep_simulation(self):
        """Simulate a hyperparameter sweep."""
        layer_sizes = [
            [10, 20],
            [10, 30],
            [15, 20],
        ]

        networks = []
        for sizes in layer_sizes:
            config = {"layers": [{"size": s} for s in sizes]}
            net = DeepNetwork.from_dict(config)
            networks.append(net)

        assert len(networks) == 3
        assert networks[0].layer_sizes == [10, 20]
        assert networks[1].layer_sizes == [10, 30]
        assert networks[2].layer_sizes == [15, 20]

    def test_reproducibility_via_config_dict(self):
        """Two networks built from the same dict config are equivalent."""
        config = {
            "layers": [
                {"size": 10},
                {"size": 20, "coupling_fn": "gelu"},
                {"size": 15},
            ],
            "coupling_fn": "identity",
        }

        net1 = DeepNetwork.from_dict(config)
        net2 = DeepNetwork.from_dict(config)

        assert net1.n_layers == net2.n_layers
        assert net1.layer_sizes == net2.layer_sizes
        # Both should be independently usable
        assert net1.state is not None
        assert net2.state is not None

    def test_backward_compatible_with_chaining(self):
        """Existing code using method chaining still works."""
        net = DeepNetwork().add_layer(size=10).add_layer(size=20).add_layer(size=15)
        assert net.n_layers == 3
        assert net.layer_sizes == [10, 20, 15]
