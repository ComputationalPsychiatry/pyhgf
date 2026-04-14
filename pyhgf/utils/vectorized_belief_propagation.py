# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized belief propagation step for deep predictive coding networks."""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax.numpy as jnp

from pyhgf.typing import NetworkState
from pyhgf.updates.vectorized.binary import (
    vectorized_binary_prediction,
    vectorized_binary_prediction_error,
)
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
    layer_kinds: Optional[list[str]] = None,
    adam_params: Optional[tuple[float, float, float, Optional[float]]] = None,
    update_type: str = "eHGF",
) -> tuple[NetworkState, jnp.ndarray]:
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
    layer_kinds :
        Per-layer node type (``"volatile"`` or ``"binary"``).  Defaults to
        all ``"volatile"`` when *None*.
    adam_params :
        Tuple ``(beta1, beta2, epsilon, lr_override)`` for Adam optimiser.
        ``lr_override`` is an optional Adam-specific learning rate that overrides the
        main *lr* argument.  When *None*, Adam is not used.

    Returns
    -------
    new_state :
        Updated network state.
    output_pred :
        Output layer prediction (expected mean).
    """
    x, y = inputs
    layers = list(state.layers)
    weights = list(state.weights)
    params = list(state.params)

    n_layers = len(layers)

    # Default: all volatile layers
    if layer_kinds is None:
        layer_kinds = ["volatile"] * n_layers

    # 1. Set predictors (top layer = input)
    layers[-1] = layers[-1]._replace(expected_mean=x, mean=x)

    # 2. Set observations (bottom layer = output)
    layers[0] = layers[0]._replace(mean=y)

    # 3. Prediction: top-down (using current parent means)
    for i in range(n_layers - 1, 0, -1):
        if layer_kinds[i - 1] == "binary":
            layers[i - 1] = vectorized_binary_prediction(
                child_state=layers[i - 1],
                parent_state=layers[i],
                weights=weights[i - 1],
                coupling_fn=coupling_fns[i],
                parent_has_constant=add_constant_inputs[i],
            )
        else:
            layers[i - 1] = vectorized_layer_prediction(
                child_state=layers[i - 1],
                parent_state=layers[i],
                weights=weights[i - 1],
                params=params[i - 1],
                time_step=state.time_step,
                coupling_fn=coupling_fns[i],
                parent_has_constant=add_constant_inputs[i],
            )

    # Step 4a: PE for output layer (mean = y, observation-pinned)
    # Exclude the bias column (if any) from the parent count.
    n_parents_0 = weights[0].shape[1] - (1 if add_constant_inputs[1] else 0)
    if layer_kinds[0] == "binary":
        layers[0] = vectorized_binary_prediction_error(layer=layers[0])
    else:
        layers[0] = vectorized_layer_prediction_error(
            layer=layers[0],
            n_parents=n_parents_0,
            params=params[0],
            update_type=update_type,
        )

    # Step 4b: per hidden layer — posterior then PE (interleaved)
    for i in range(1, n_layers - 1):
        # Real (non-constant) parent count for this layer
        n_vp = weights[i].shape[1] - (1 if add_constant_inputs[i + 1] else 0)
        layers[i] = vectorized_layer_posterior_update(
            layer=layers[i],
            child=layers[i - 1],
            weights=weights[i - 1],
            coupling_fn_grad=coupling_fn_grads[i],  # parent i's grad
            parent_has_constant=add_constant_inputs[i],
        )
        # Recompute PE and update volatility level so the layer
        # above receives the correct (post-posterior) error signal.
        if layer_kinds[i] == "binary":
            layers[i] = vectorized_binary_prediction_error(layer=layers[i])
        else:
            layers[i] = vectorized_layer_prediction_error(
                layer=layers[i],
                n_parents=n_vp,
                params=params[i],
                update_type=update_type,
            )

    # ========== LEARNING PHASE (after inference converges) ==========
    # Update weights once using converged activities
    lr_value: Optional[float] = None if lr == "dynamic" else float(lr)

    adam_m_list = list(state.adam_m)
    adam_v_list = list(state.adam_v)
    use_adam = adam_params is not None
    adam_t = state.adam_t + 1 if use_adam else state.adam_t
    if adam_params is not None:
        beta1, beta2, epsilon, adam_lr_override = adam_params
        # Adam lr override takes precedence (matches Rust behaviour)
        if adam_lr_override is not None:
            lr_value = adam_lr_override
    else:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for i in range(1, n_layers):
        # Binary nodes don't learn coupling weights (matches Network/Rust).
        if layer_kinds[i - 1] == "binary":
            continue
        weights[i - 1], new_m, new_v = vectorized_weight_update(
            parent_state=layers[i],
            child_state=layers[i - 1],
            weights=weights[i - 1],
            coupling_fn=coupling_fns[i],
            lr=lr_value,
            parent_has_constant=add_constant_inputs[i],
            adam_m=adam_m_list[i - 1] if use_adam else None,
            adam_v=adam_v_list[i - 1] if use_adam else None,
            adam_t=adam_t,
            adam_beta1=beta1,
            adam_beta2=beta2,
            adam_epsilon=epsilon,
        )
        if use_adam and new_m is not None:
            adam_m_list[i - 1] = new_m
            adam_v_list[i - 1] = new_v

    new_state = NetworkState(
        layers=tuple(layers),
        weights=tuple(weights),
        params=tuple(params),
        time_step=state.time_step,
        adam_m=tuple(adam_m_list),
        adam_v=tuple(adam_v_list),
        adam_t=adam_t,
    )

    # Return output prediction for monitoring
    output_pred = layers[0].expected_mean

    return new_state, output_pred
