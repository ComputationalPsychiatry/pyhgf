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

        assert result.shape == (10, 8)


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
