# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Tests for the VectorizedDeepNetwork class."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyhgf.model import VectorizedDeepNetwork
from pyhgf.typing import LayerParams, LayerState, NetworkState


class TestLayerState:
    """Tests for LayerState."""

    def test_create(self):
        """Test LayerState creation."""
        state = LayerState.create(n_nodes=10)

        assert state.mean.shape == (10,)
        assert state.precision.shape == (10,)
        assert state.mean_vol.shape == (10,)

        # Check default values
        assert jnp.allclose(state.mean, 0.0)
        assert jnp.allclose(state.precision, 1.0)


class TestLayerParams:
    """Tests for LayerParams."""

    def test_create_default(self):
        """Test LayerParams creation with defaults."""
        params = LayerParams.create(n_nodes=10)

        assert params.tonic_volatility.shape == (10,)
        assert jnp.allclose(params.tonic_volatility, -4.0)
        assert jnp.allclose(params.tonic_volatility_vol, -4.0)
        assert jnp.allclose(params.volatility_coupling, 1.0)

    def test_create_custom(self):
        """Test LayerParams creation with custom values."""
        params = LayerParams.create(
            n_nodes=5,
            tonic_volatility=-3.0,
            tonic_volatility_vol=-1.0,
            volatility_coupling=0.5,
        )

        assert jnp.allclose(params.tonic_volatility, -3.0)
        assert jnp.allclose(params.tonic_volatility_vol, -1.0)
        assert jnp.allclose(params.volatility_coupling, 0.5)


class TestVectorizedDeepNetwork:
    """Tests for VectorizedDeepNetwork."""

    def test_init(self):
        """Test network initialization."""
        net = VectorizedDeepNetwork()

        assert net.coupling_fn == jnp.tanh
        assert net.layer_sizes == []
        assert net.state is None

    def test_add_nodes(self):
        """Test adding output layer nodes."""
        net = VectorizedDeepNetwork().add_layer(size=10)

        assert net.layer_sizes == [10]
        assert len(net.tonic_volatilities) == 1

    def test_add_layer(self):
        """Test adding hidden layers."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=10)
            .add_layer(size=8)
            .add_layer(size=4)
        )

        assert net.layer_sizes == [10, 8, 4]
        assert net.n_layers == 3
        assert net.n_nodes == 22

    def test_add_layer_stack(self):
        """Test adding multiple layers at once."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=10)
            .add_layer_stack(layer_sizes=[8, 6, 4])
        )

        assert net.layer_sizes == [10, 8, 6, 4]
        assert net.n_layers == 4

    def test_repr(self):
        """Test string representation."""
        net = VectorizedDeepNetwork().add_layer(size=10).add_layer(size=5)

        assert "VectorizedDeepNetwork" in repr(net)
        assert "nodes=15" in repr(net)

    def test_init_state(self):
        """Test state initialization."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=10)
            .add_layer(size=8)
            .add_layer(size=4)
        )

        state = net._init_state()

        assert isinstance(state, NetworkState)
        assert len(state.layers) == 3
        assert len(state.weights) == 2
        assert len(state.params) == 3

        # Check layer sizes
        assert state.layers[0].mean.shape == (10,)
        assert state.layers[1].mean.shape == (8,)
        assert state.layers[2].mean.shape == (4,)

        # Check weight shapes
        assert state.weights[0].shape == (10, 9)
        assert state.weights[1].shape == (8, 5)

    def test_fit_simple(self):
        """Test fitting on simple synthetic data."""
        # Create a simple network
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=3)  # Output
            .add_layer(size=5)  # Hidden
            .add_layer(size=4)  # Input
        )

        # Create simple data
        n_samples = 10
        x = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples, 3)

        # Fit
        net.fit(x, y, lr=0.1)

        assert net.state is not None
        assert net.trajectories is not None
        assert net.predictions is not None

        # Check predictions shape
        assert net.predictions.shape == (n_samples, 3)

    def test_predict(self):
        """Test prediction after fitting."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2)
            .add_layer(size=4)
            .add_layer(size=3)
        )

        # Train
        x_train = np.random.randn(20, 3)
        y_train = np.random.randn(20, 2)
        net.fit(x_train, y_train, lr=0.1)

        # Predict
        x_test = np.random.randn(5, 3)
        preds = net.predict(x_test)

        assert preds.shape == (5, 2)

    def test_predict_single_sample(self):
        """Test prediction on a single sample."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2)
            .add_layer(size=4)
            .add_layer(size=3)
        )

        x_train = np.random.randn(20, 3)
        y_train = np.random.randn(20, 2)
        net.fit(x_train, y_train, lr=0.1)

        # Single sample prediction
        x_single = np.random.randn(3)
        pred = net.predict(x_single)

        assert pred.shape == (2,)

    def test_predict_before_fit_raises(self):
        """Test that predicting before fit raises an error."""
        net = VectorizedDeepNetwork().add_layer(size=2).add_layer(size=3)

        with pytest.raises(ValueError, match="must be fit"):
            net.predict(np.random.randn(5, 3))

    def test_get_weights(self):
        """Test getting weights after fit."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2)
            .add_layer(size=4)
            .add_layer(size=3)
        )

        x = np.random.randn(10, 3)
        y = np.random.randn(10, 2)
        net.fit(x, y, lr=0.1)

        weights = net.state.weights

        assert len(weights) == 2
        assert weights[0].shape == (2, 5)
        assert weights[1].shape == (4, 4)

    def test_reset(self):
        """Test resetting the network."""
        net = VectorizedDeepNetwork().add_layer(size=2).add_layer(size=3)

        x = np.random.randn(10, 3)
        y = np.random.randn(10, 2)
        net.fit(x, y, lr=0.1)

        # Store old weights
        old_weights = net.state.weights

        # Reset
        net.reset()

        # Weights should be all ones after reset
        new_weights = net.state.weights
        assert jnp.allclose(new_weights[0], jnp.ones_like(new_weights[0]))

    def test_custom_coupling_fn(self):
        """Test network with custom coupling function."""
        net = VectorizedDeepNetwork(coupling_fn=jax.nn.sigmoid)

        net.add_layer(size=2).add_layer(size=3)

        x = np.random.randn(10, 3)
        y = np.random.randn(10, 2)
        net.fit(x, y, lr=0.1)

        # Should work without errors
        preds = net.predict(x[:5])
        assert preds.shape == (5, 2)


class TestVectorizedUpdates:
    """Tests for vectorized update functions."""

    def test_prediction_shapes(self):
        """Test that prediction preserves shapes."""
        from pyhgf.updates.vectorized.volatile import vectorized_layer_prediction

        child_state = LayerState.create(10)
        parent_state = LayerState.create(8)
        weights = jnp.ones((10, 8)) * 0.1
        params = LayerParams.create(10)

        result = vectorized_layer_prediction(
            child_state, parent_state, weights, params, time_step=1.0
        )

        assert result.mean.shape == (10,)
        assert result.expected_mean.shape == (10,)
        assert result.expected_precision.shape == (10,)

    def test_prediction_error_shapes(self):
        """Test that prediction error preserves shapes."""
        from pyhgf.updates.vectorized.volatile import vectorized_layer_prediction_error

        state = LayerState.create(10)
        # Set some values so we get non-trivial errors
        state = state._replace(
            mean=jnp.ones(10) * 0.5,
            expected_mean=jnp.ones(10) * 0.3,
        )

        result = vectorized_layer_prediction_error(state, n_parents=5)

        assert result.value_prediction_error.shape == (10,)
        assert result.volatility_prediction_error.shape == (10,)

    def test_posterior_update_shapes(self):
        """Test that posterior update preserves shapes."""
        from pyhgf.updates.vectorized.volatile import vectorized_layer_posterior_update

        parent_state = LayerState.create(8)
        child_state = LayerState.create(10)

        # Set some values
        child_state = child_state._replace(
            value_prediction_error=jnp.ones(10) * 0.1,
            expected_precision=jnp.ones(10),
        )
        parent_state = parent_state._replace(
            expected_precision=jnp.ones(8),
            effective_precision=jnp.ones(8) * 0.5,
        )

        weights = jnp.ones((10, 8)) * 0.1
        params = LayerParams.create(8)
        coupling_fn_grad = jax.grad(lambda x: jnp.tanh(x))

        result = vectorized_layer_posterior_update(
            parent_state,
            child_state,
            weights,
            params,
            coupling_fn_grad,
            n_value_parents=8,
        )

        assert result.mean.shape == (8,)
        assert result.precision.shape == (8,)
        assert result.mean_vol.shape == (8,)
        assert result.precision_vol.shape == (8,)

    def test_weight_update_shapes(self):
        """Test that weight update preserves shapes."""
        from pyhgf.updates.vectorized.learning import vectorized_weight_update

        parent_state = LayerState.create(8)
        child_state = LayerState.create(10)

        # Set some values
        child_state = child_state._replace(mean=jnp.ones(10) * 0.5)
        parent_state = parent_state._replace(mean=jnp.ones(8) * 0.3)

        weights = jnp.ones((10, 8)) * 0.1

        result = vectorized_weight_update(
            parent_state, child_state, weights, jnp.tanh, lr=0.1
        )

        assert result[0].shape == (10, 8)
        assert result[1] is None  # no Adam
        assert result[2] is None


class TestAddLayerValidation:
    """Tests for add_layer input validation."""

    def test_invalid_kind_raises(self):
        """Invalid layer kind raises ValueError."""
        net = VectorizedDeepNetwork()
        with pytest.raises(ValueError, match="Invalid layer kind"):
            net.add_layer(size=5, kind="unknown")

    def test_one_to_one_with_bias_raises(self):
        """fully_connected=False with add_constant_input=True raises."""
        net = VectorizedDeepNetwork().add_layer(size=5, add_constant_input=False)
        with pytest.raises(ValueError, match="One-to-one layers"):
            net.add_layer(size=5, fully_connected=False, add_constant_input=True)

    def test_one_to_one_size_mismatch_raises(self):
        """fully_connected=False with different child size raises."""
        net = VectorizedDeepNetwork().add_layer(size=5, add_constant_input=False)
        with pytest.raises(ValueError, match="same size"):
            net.add_layer(size=8, fully_connected=False, add_constant_input=False)

    def test_one_to_one_valid(self):
        """fully_connected=False with matching sizes succeeds."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=4, add_constant_input=False)
            .add_layer(size=4, fully_connected=False, add_constant_input=False)
        )
        state = net._init_state()
        # One-to-one weight should be an identity matrix
        assert state.weights[0].shape == (4, 4)
        np.testing.assert_allclose(state.weights[0], jnp.eye(4))


class TestAddConstantInput:
    """Tests for bias (add_constant_input) behaviour."""

    def test_bias_adds_extra_column(self):
        """add_constant_input=True adds an extra weight column for bias."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=3, add_constant_input=False)
            .add_layer(size=5, add_constant_input=True)
        )
        state = net._init_state()
        # 5 parent nodes + 1 bias = 6 columns
        assert state.weights[0].shape == (3, 6)

    def test_no_bias_no_extra_column(self):
        """add_constant_input=False gives exact parent-count columns."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=3, add_constant_input=False)
            .add_layer(size=5, add_constant_input=False)
        )
        state = net._init_state()
        assert state.weights[0].shape == (3, 5)

    def test_fit_with_no_bias(self):
        """Network with add_constant_input=False fits and predicts."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=4, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)
        net.fit(x, y, lr=0.1)
        preds = net.predict(x[:3])
        assert preds.shape == (3, 2)


class TestBinaryLayer:
    """Tests for binary layer support."""

    def test_add_binary_layer(self):
        """Can add a binary output layer."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=1, kind="binary")
            .add_layer(size=4)
            .add_layer(size=3, add_constant_input=False)
        )
        assert net.layer_kinds[0] == "binary"
        assert net.n_layers == 3

    def test_binary_fit_and_predict(self):
        """Binary output layer fits and produces sigmoid-range predictions."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=1, kind="binary")
            .add_layer(size=4)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(15, 3).astype(np.float32)
        y = np.random.choice([0.0, 1.0], size=(15, 1)).astype(np.float32)
        net.fit(x, y, lr=0.1)
        preds = net.predict(x[:5])
        assert preds.shape == (5, 1)
        # Binary output should be in (0, 1)
        assert np.all(np.array(preds) > 0.0)
        assert np.all(np.array(preds) < 1.0)


class TestWeightInitialisation:
    """Tests for weight_initialisation method."""

    def _make_network(self):
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=3, add_constant_input=False)
            .add_layer(size=5, add_constant_input=False)
            .add_layer(size=4, add_constant_input=False)
        )
        net.state = net._init_state()
        return net

    def test_none_strategy_is_noop(self):
        """strategy=None leaves weights unchanged."""
        net = self._make_network()
        old_w0 = net.state.weights[0].copy()
        net.weight_initialisation(strategy=None)
        np.testing.assert_array_equal(net.state.weights[0], old_w0)

    def test_xavier(self):
        """Xavier initialisation changes weights and is reproducible."""
        net1 = self._make_network()
        net2 = self._make_network()
        net1.weight_initialisation("xavier", seed=0)
        net2.weight_initialisation("xavier", seed=0)
        np.testing.assert_array_equal(net1.state.weights[0], net2.state.weights[0])
        # Weights should no longer be all ones
        assert not jnp.allclose(net1.state.weights[0], 1.0)

    def test_he(self):
        """He initialisation produces non-trivial weights."""
        net = self._make_network()
        net.weight_initialisation("he", seed=42)
        assert not jnp.allclose(net.state.weights[0], 1.0)

    def test_orthogonal(self):
        """Orthogonal initialisation produces non-trivial weights."""
        net = self._make_network()
        net.weight_initialisation("orthogonal", seed=42)
        assert not jnp.allclose(net.state.weights[0], 1.0)

    def test_sparse(self):
        """Sparse initialisation produces non-trivial weights."""
        net = self._make_network()
        net.weight_initialisation("sparse", seed=42)
        assert not jnp.allclose(net.state.weights[0], 1.0)

    def test_invalid_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        net = self._make_network()
        with pytest.raises(ValueError, match="Invalid weight initialisation"):
            net.weight_initialisation("unknown")

    def test_no_state_raises(self):
        """Calling weight_initialisation before state init raises."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=3, add_constant_input=False)
            .add_layer(size=4, add_constant_input=False)
        )
        with pytest.raises(ValueError, match="State must be initialised"):
            net.weight_initialisation("xavier")


class TestPerLayerCouplingFn:
    """Tests for per-layer coupling function overrides."""

    def test_per_layer_fn_stored(self):
        """Per-layer coupling_fn is stored correctly."""
        net = (
            VectorizedDeepNetwork(coupling_fn=jnp.tanh)
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=4, coupling_fn=jax.nn.relu, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        # Layer 0 and 2 should use network default (tanh)
        assert net.coupling_fns[0] is jnp.tanh
        assert net.coupling_fns[2] is jnp.tanh
        # Layer 1 should use the override
        assert net.coupling_fns[1] is jax.nn.relu

    def test_per_layer_fn_fit_predict(self):
        """Network with mixed coupling functions fits and predicts."""
        net = (
            VectorizedDeepNetwork(coupling_fn=jnp.tanh)
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=4, coupling_fn=jax.nn.relu, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)
        net.fit(x, y, lr=0.1)
        preds = net.predict(x[:3])
        assert preds.shape == (3, 2)


class TestDynamicLearningRate:
    """Tests for lr='dynamic' (Kalman-gain) mode."""

    def test_dynamic_lr_fit(self):
        """fit(lr='dynamic') runs without error."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=4, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)
        net.fit(x, y, lr="dynamic")
        assert net.predictions.shape == (10, 2)


class TestPropagationFnCaching:
    """Tests for JIT-compiled propagation function caching."""

    def test_same_lr_reuses_fn(self):
        """Calling fit twice with same lr reuses the compiled function."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randn(5, 2).astype(np.float32)
        net.fit(x, y, lr=0.1)
        fn_after_first = net._propagation_fn
        net.fit(x, y, lr=0.1)
        assert net._propagation_fn is fn_after_first

    def test_different_lr_recreates_fn(self):
        """Changing lr triggers recompilation."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randn(5, 2).astype(np.float32)
        net.fit(x, y, lr=0.1)
        fn_first = net._propagation_fn
        net.fit(x, y, lr=0.2)
        assert net._propagation_fn is not fn_first

    def test_reset_clears_cache(self):
        """reset() clears the cached propagation function."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=2, add_constant_input=False)
            .add_layer(size=3, add_constant_input=False)
        )
        x = np.random.randn(5, 3).astype(np.float32)
        y = np.random.randn(5, 2).astype(np.float32)
        net.fit(x, y, lr=0.1)
        assert net._propagation_fn is not None
        net.reset()
        assert net._propagation_fn is None
        assert net._propagation_lr is None
        assert net._prediction_fn is None


class TestScaling:
    """Tests for network scaling behavior."""

    def test_medium_network(self):
        """Test a medium-sized network compiles and runs."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=10)  # Output
            .add_layer(size=32)  # Hidden 1
            .add_layer(size=32)  # Hidden 2
            .add_layer(size=20)  # Input
        )

        # Total: 10 + 32 + 32 + 20 = 94 nodes
        assert net.n_nodes == 94

        x = np.random.randn(50, 20)
        y = np.random.randn(50, 10)

        # This should compile and run
        net.fit(x, y, lr=0.1)

        preds = net.predict(x[:10])
        assert preds.shape == (10, 10)

    @pytest.mark.slow
    def test_large_network(self):
        """Test a larger network (like FashionMNIST scale)."""
        net = (
            VectorizedDeepNetwork()
            .add_layer(size=10)  # Output (labels)
            .add_layer(size=32)  # Hidden
            .add_layer(size=32)  # Hidden
            .add_layer(size=32)  # Hidden
            .add_layer(size=784)  # Input (image)
        )

        # Total: 10 + 32*3 + 784 = 890 nodes
        assert net.n_nodes == 890

        x = np.random.randn(10, 784)
        y = np.random.randn(10, 10)

        # This should compile (the main test is that it doesn't OOM)
        net.fit(x, y, lr=0.1)

        preds = net.predict(x[:2])
        assert preds.shape == (2, 10)
