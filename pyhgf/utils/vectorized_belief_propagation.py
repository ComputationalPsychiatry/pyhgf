# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorized belief propagation step for deep predictive coding networks."""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax.numpy as jnp
from jax.nn import sigmoid

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
    volatility_parents: Optional[list[bool]] = None,
    learning_kind: str = "precision_weighted",
    weight_update: bool = True,
    max_posterior_precision: float = 1e10,
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
        Learning rate: a non-negative float for direct scaling, or ``"adam"``
        for the Adam optimiser.
    layer_kinds :
        Per-layer node type (``"volatile"`` or ``"binary"``).  Defaults to
        all ``"volatile"`` when *None*.
    adam_params :
        Tuple ``(beta1, beta2, epsilon, lr_override)`` for Adam optimiser.
        ``lr_override`` is an optional Adam-specific learning rate that overrides the
        main *lr* argument.  When *None*, Adam is not used.
    volatility_parents :
        Per-layer flag controlling whether the implied internal volatility parent
        is active.  When ``True`` (default), mean_vol and precision_vol are
        predicted and updated for that layer.  When ``False``, the volatility
        level is frozen and only tonic_volatility determines the expected
        precision.  Defaults to all ``True`` when *None*.
    learning_kind :
        Gradient computation mode passed to :func:`vectorized_weight_update`:
        ``"standard"``, ``"precision_weighted"`` (default), or ``"precision_ratio"``.
    weight_update :
        If ``True`` (default), run the weight-learning phase at the end of the
        step. If ``False``, weights and Adam moments are passed through
        unchanged — useful for inference-only forward passes through a fixed
        network.
    max_posterior_precision :
        Upper bound applied to every posterior precision write (value level via
        :func:`vectorized_layer_posterior_update` and volatility level via
        :func:`vectorized_layer_prediction_error`). Default ``1e10``.

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

    # Default: all layers have an implied volatility parent
    if volatility_parents is None:
        volatility_parents = [True] * n_layers

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
                has_volatility_parent=volatility_parents[i - 1],
                # Layer 0 is the observation layer of a DeepNetwork — it has no
                # value children below, so it does not undergo a random walk.
                is_input_layer=(i - 1 == 0),
            )

    # Step 4a: PE for output layer (mean = y, observation-pinned)
    if layer_kinds[0] == "binary":
        layers[0] = vectorized_binary_prediction_error(layer=layers[0])
    else:
        layers[0] = vectorized_layer_prediction_error(
            layer=layers[0],
            params=params[0],
            update_type=update_type,
            has_volatility_parent=volatility_parents[0],
            max_posterior_precision=max_posterior_precision,
        )

    # Step 4b: per hidden layer — posterior then PE (interleaved)
    for i in range(1, n_layers - 1):
        layers[i] = vectorized_layer_posterior_update(
            layer=layers[i],
            child=layers[i - 1],
            weights=weights[i - 1],
            coupling_fn_grad=coupling_fn_grads[i],  # parent i's grad
            parent_has_constant=add_constant_inputs[i],
            max_posterior_precision=max_posterior_precision,
        )
        # Recompute PE and update volatility level so the layer
        # above receives the correct (post-posterior) error signal.
        if layer_kinds[i] == "binary":
            layers[i] = vectorized_binary_prediction_error(layer=layers[i])
        else:
            layers[i] = vectorized_layer_prediction_error(
                layer=layers[i],
                params=params[i],
                update_type=update_type,
                has_volatility_parent=volatility_parents[i],
                max_posterior_precision=max_posterior_precision,
            )

    # ========== LEARNING PHASE (after inference converges) ==========
    # Update weights once using converged activities — skipped when
    # ``weight_update=False`` so the same network can be used for pure
    # inference (forward + posterior + PE only).
    use_adam = lr == "adam"
    adam_t = state.adam_t + 1 if (use_adam and weight_update) else state.adam_t

    adam_m_list = list(state.adam_m)
    adam_v_list = list(state.adam_v)

    if weight_update:
        if adam_params is not None:
            beta1, beta2, epsilon, _adam_lr_override = adam_params
            _adam_lr = _adam_lr_override if _adam_lr_override is not None else 1e-3
        else:
            beta1, beta2, epsilon, _adam_lr = 0.9, 0.999, 1e-8, 1e-3

        for i in range(1, n_layers):
            weights[i - 1], new_m, new_v = vectorized_weight_update(
                parent_state=layers[i],
                child_state=layers[i - 1],
                weights=weights[i - 1],
                coupling_fn=coupling_fns[i],
                kind=learning_kind,
                lr=lr,
                parent_has_constant=add_constant_inputs[i],
                child_is_binary=(layer_kinds[i - 1] == "binary"),
                adam_m=adam_m_list[i - 1] if use_adam else None,
                adam_v=adam_v_list[i - 1] if use_adam else None,
                adam_t=adam_t,
                adam_lr=_adam_lr,
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
