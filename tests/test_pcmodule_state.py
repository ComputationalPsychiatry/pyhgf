# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests for the PCModule protocol and the ``init_state()`` pytree contract.

Validates that:
1. PCModule protocol enforcement catches incomplete implementations
2. Each part's ``init_state()`` returns the pytree the executor expects
3. Composite parts (Sequential, Residual, attention, GPT) nest child states
   in the right order
"""

from __future__ import annotations

import jax.numpy as jnp
import optax
import pytest

from pyhgf.model import (
    DeepNetwork,
    DeepNetworkAdapter,
    EquinoxAdapter,
    PCModule,
    PCSequential,
    Residual,
    gelu_adapter,
)
from pyhgf.model.transformer import HybridGPT, MultiHeadAttention


class TestPCModuleProtocol:
    """Test PCModule protocol enforcement via __init_subclass__."""

    def test_incomplete_subclass_missing_init(self):
        """Subclass without __init__ raises TypeError at class definition."""
        with pytest.raises(TypeError, match="must define __init__"):

            class BadPart(PCModule):
                def init_state(self):
                    return ()

    def test_incomplete_subclass_missing_init_state(self):
        """Subclass without init_state raises TypeError at class definition."""
        with pytest.raises(TypeError, match="must define init_state"):

            class BadPart(PCModule):
                def __init__(self):
                    pass

    def test_valid_subclass_passes(self):
        """Complete subclass with __init__ and init_state passes validation."""

        # This should not raise
        class GoodPart(PCModule):
            def __init__(self):
                pass

            def init_state(self):
                return ()

        assert issubclass(GoodPart, PCModule)


class TestEquinoxAdapterState:
    """Frozen parts hold no state."""

    def test_init_state_is_empty_tuple(self):
        """EquinoxAdapter.init_state() returns the empty pytree ()."""
        assert gelu_adapter().init_state() == ()
        adapter = EquinoxAdapter(
            forward_fn=lambda x: (x, ()),
            backward_fn=lambda cache, error: error,
        )
        assert adapter.init_state() == ()


class TestDeepNetworkAdapterState:
    """Learning parts hold (network, opt_state)."""

    @pytest.fixture
    def simple_network(self):
        """Create a simple 2-layer network for testing."""
        return (
            DeepNetwork()
            .add_layer(size=4)  # output
            .add_layer(size=8)  # input
        )

    def test_init_state_without_layers_raises(self):
        """init_state() on empty network raises ValueError."""
        adapter = DeepNetworkAdapter(DeepNetwork())
        with pytest.raises(ValueError, match="no layers yet"):
            adapter.init_state()

    def test_init_state_with_frozen_weights(self, simple_network):
        """init_state() with optimizer=None has opt_state=None."""
        network, opt_state = DeepNetworkAdapter(
            simple_network, optimizer=None
        ).init_state()
        assert network is not None
        assert opt_state is None

    def test_init_state_with_optimizer(self, simple_network):
        """init_state() with optimizer initializes opt_state."""
        network, opt_state = DeepNetworkAdapter(
            simple_network, optimizer=optax.sgd(0.1)
        ).init_state()
        assert network is not None
        assert opt_state is not None

    def test_network_state_structure(self, simple_network):
        """The threaded network carries its layers."""
        network, _ = DeepNetworkAdapter(simple_network).init_state()
        assert hasattr(network, "layers")
        assert len(network.layers) == 2


class TestPCSequentialState:
    """PCSequential returns a tuple of child states, in order."""

    def test_init_state_single_part(self):
        """Sequential with one part returns a 1-tuple."""
        part = EquinoxAdapter(forward_fn=lambda x: (x, ()), backward_fn=lambda c, e: e)
        state = PCSequential([part]).init_state()
        assert state == ((),)

    def test_init_state_multiple_parts(self):
        """Sequential returns child states in part order."""
        net = DeepNetwork().add_layer(4).add_layer(8)
        parts = [
            EquinoxAdapter(forward_fn=lambda x: (x, ()), backward_fn=lambda c, e: e),
            DeepNetworkAdapter(net),
            gelu_adapter(),
        ]
        state = PCSequential(parts).init_state()
        assert len(state) == 3
        assert state[0] == ()  # frozen
        assert state[1][0] is net.state  # learning: (network, opt_state)
        assert state[2] == ()  # frozen

    def test_child_states_in_order(self):
        """Child states are in the same order as parts."""
        net1 = DeepNetwork().add_layer(4).add_layer(8)
        net2 = DeepNetwork().add_layer(6).add_layer(10)
        state = PCSequential([
            DeepNetworkAdapter(net1),
            DeepNetworkAdapter(net2),
        ]).init_state()
        assert state[0][0] is net1.state
        assert state[1][0] is net2.state


class TestResidualState:
    """Residual passes its branch's state through unchanged."""

    def test_init_state_wraps_branch_state(self):
        """Residual returns the branch's state directly."""
        net = DeepNetwork().add_layer(4).add_layer(8)
        state = Residual(DeepNetworkAdapter(net)).init_state()
        assert state[0] is net.state  # (network, opt_state)

    def test_residual_with_sequential_branch(self):
        """Residual wrapping a Sequential delegates to the chain's tuple."""
        net = DeepNetwork().add_layer(4).add_layer(8)
        seq = PCSequential([gelu_adapter(), DeepNetworkAdapter(net)])
        state = Residual(seq).init_state()
        assert len(state) == 2
        assert state[0] == ()


class TestMultiHeadAttentionState:
    """Attention returns the (q, k, v, o) tuple of its weight tables."""

    def test_init_state_collects_four_projections(self):
        """Attention state aggregates Q, K, V, O parts."""
        attn = MultiHeadAttention(
            gelu_adapter(), gelu_adapter(), gelu_adapter(), gelu_adapter(), n_heads=8
        )
        assert attn.init_state() == ((), (), (), ())

    def test_attention_with_mixed_frozen_learning(self):
        """Attention can mix frozen and learning parts."""
        net = DeepNetwork().add_layer(64).add_layer(64)
        attn = MultiHeadAttention(
            DeepNetworkAdapter(net),
            gelu_adapter(),
            gelu_adapter(),
            gelu_adapter(),
            n_heads=8,
        )
        q_state, k_state, v_state, o_state = attn.init_state()
        assert q_state[0] is net.state  # learning: (network, opt_state)
        assert (k_state, v_state, o_state) == ((), (), ())


class TestHybridGPTState:
    """HybridGPT returns (pipeline_state, token_state, position_state)."""

    def test_init_state_frozen_embeddings(self):
        """Frozen embeddings contribute empty states."""
        gpt = HybridGPT(
            tok_table=jnp.ones((256, 64)),
            pos_table=jnp.ones((512, 64)),
            pipeline=PCSequential([gelu_adapter()]),
            token_part=None,
            position_part=None,
        )
        pipeline_state, token_state, position_state = gpt.init_state()
        assert pipeline_state == ((),)
        assert token_state == ()
        assert position_state == ()

    def test_init_state_learnable_embeddings(self):
        """Learnable embeddings contribute (network, opt_state) states."""
        tok_net = DeepNetwork().add_layer(64).add_layer(256)
        pos_net = DeepNetwork().add_layer(64).add_layer(512)
        gpt = HybridGPT(
            tok_table=jnp.ones((256, 64)),
            pos_table=jnp.ones((512, 64)),
            pipeline=PCSequential([gelu_adapter()]),
            token_part=DeepNetworkAdapter(tok_net),
            position_part=DeepNetworkAdapter(pos_net),
        )
        pipeline_state, token_state, position_state = gpt.init_state()
        assert pipeline_state == ((),)
        assert token_state[0] is tok_net.state
        assert position_state[0] is pos_net.state


class TestBackwardCompatibility:
    """Existing construction patterns still work."""

    def test_existing_deep_network_adapter_construction(self):
        """DeepNetworkAdapter construction works unchanged."""
        net = DeepNetwork().add_layer(4).add_layer(8)
        adapter = DeepNetworkAdapter(net, optimizer=optax.adam(1e-3))
        assert adapter.net is net
        assert adapter.optimizer is not None

    def test_existing_sequential_construction(self):
        """PCSequential construction works unchanged."""
        parts = [gelu_adapter(), gelu_adapter()]
        assert PCSequential(parts).parts == parts

    def test_existing_residual_construction(self):
        """Residual construction works unchanged."""
        branch = gelu_adapter()
        assert Residual(branch).branch is branch
