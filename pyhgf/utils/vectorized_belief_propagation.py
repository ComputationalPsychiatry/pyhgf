# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized belief propagation step for deep predictive coding networks."""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax.numpy as jnp

from pyhgf.typing import NetworkState
from pyhgf.updates.vectorized.learning import vectorized_weight_update
from pyhgf.updates.vectorized.volatile import (
    vectorized_layer_posterior_update,
    vectorized_layer_prediction,
    vectorized_layer_prediction_error,
)


def propagation_step(
    state: NetworkState,
    inputs: tuple,
    coupling_fns: list[Callable],
    coupling_fn_grads: list[Callable],
    add_constant_inputs: list[bool],
    lr: Union[float, str],
) -> tuple[NetworkState, tuple[NetworkState, jnp.ndarray]]:
    """Single propagation step through the network.

    This performs a full inference-then-learning cycle for one data sample:
    prediction, prediction-error computation, posterior update, and weight update.

    Parameters
    ----------
    state :
        Current network state.
    inputs :
        Tuple ``(x, y)`` of input (predictor) and output (observation) vectors.
    coupling_fns :
        Per-layer coupling functions.  ``coupling_fns[i]`` is applied to
        ``layer[i].expected_mean`` when layer *i* acts as a parent.
    coupling_fn_grads :
        Gradients of the coupling functions (same indexing as *coupling_fns*).
    add_constant_inputs :
        Per-layer flags indicating whether a bias term is added.
    lr :
        Learning rate (float) or ``"dynamic"`` for Kalman-gain updates.

    Returns
    -------
    new_state :
        Updated network state.
    (new_state, output_pred) :
        Carry-along tuple for ``jax.lax.scan`` compatibility.
    """
    x, y = inputs
    layers = list(state.layers)
    weights = list(state.weights)
    params = list(state.params)

    n_layers = len(layers)

    # 1. Set predictors (top layer = input)
    layers[-1] = layers[-1]._replace(expected_mean=x, mean=x)

    # 2. Set observations (bottom layer = output)
    layers[0] = layers[0]._replace(mean=y)

    # 3. Prediction: top-down (using current parent means)
    for i in range(n_layers - 1, 0, -1):
        layers[i - 1] = vectorized_layer_prediction(
            child_state=layers[i - 1],
            parent_state=layers[i],
            weights=weights[i - 1],
            params=params[i - 1],
            time_step=state.time_step,
            coupling_fn=coupling_fns[i],  # parent i's coupling fn
            parent_has_constant=add_constant_inputs[i],
        )

    # Step 4a: PE for output layer (mean = y, observation-pinned)
    # Exclude the bias column (if any) from the parent count.
    n_parents_0 = weights[0].shape[1] - (1 if add_constant_inputs[1] else 0)
    layers[0] = vectorized_layer_prediction_error(
        layer=layers[0],
        n_parents=n_parents_0,
    )

    # Step 4b: per hidden layer — posterior then PE (interleaved)
    for i in range(1, n_layers - 1):
        # Real (non-constant) parent count for this layer
        n_vp = weights[i].shape[1] - (1 if add_constant_inputs[i + 1] else 0)
        layers[i] = vectorized_layer_posterior_update(
            layer=layers[i],
            child=layers[i - 1],
            weights=weights[i - 1],
            params=params[i],
            coupling_fn_grad=coupling_fn_grads[i],  # parent i's grad
            n_value_parents=n_vp,
            parent_has_constant=add_constant_inputs[i],
        )
        # Recompute PE using updated posterior mean so the layer
        # above receives the correct (post-posterior) error signal.
        layers[i] = vectorized_layer_prediction_error(
            layer=layers[i],
            n_parents=n_vp,
        )

    # ========== LEARNING PHASE (after inference converges) ==========
    # Update weights once using converged activities
    lr_value: Optional[float] = None if lr == "dynamic" else float(lr)
    for i in range(1, n_layers):
        weights[i - 1] = vectorized_weight_update(
            parent_state=layers[i],
            child_state=layers[i - 1],
            weights=weights[i - 1],
            coupling_fn=coupling_fns[i],
            lr=lr_value,
            parent_has_constant=add_constant_inputs[i],
        )

    new_state = NetworkState(
        layers=tuple(layers),
        weights=tuple(weights),
        params=tuple(params),
        time_step=state.time_step,
    )

    # Return output prediction for monitoring
    output_pred = layers[0].expected_mean

    return new_state, (new_state, output_pred)
