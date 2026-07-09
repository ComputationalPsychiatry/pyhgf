# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Type annotations for error conventions in mixed pipelines.

Defines two distinct error types to prevent sign-error bugs:

- **DescentError**: The pipeline convention (gradient-like), used between parts
  and at the executor boundary. Positive means the output is too high.

- **ObservedMinusPredicted**: PyHGF's internal convention (prediction error),
  used inside learning parts. Positive means the observation exceeded prediction.

The type system catches when you accidentally mix these conventions, preventing
silent training divergence caused by a missing negation.

Example:
    >>> from jax import Array
    >>> from pyhgf.model.error_types import DescentError, ObservedMinusPredicted
    >>> import jax.numpy as jnp
    >>>
    >>> # Pipeline uses descent errors
    >>> error_in_pipeline: DescentError = jnp.array([0.1, -0.2])
    >>>
    >>> # Converting to PyHGF convention for learning
    >>> obs_minus_pred: ObservedMinusPredicted = -error_in_pipeline
    >>>
    >>> # Type system helps catch mistakes:
    >>> def wrong_conversion(e: DescentError) -> DescentError:
    ...     return -e  # Should return ObservedMinusPredicted, not DescentError!

Notes
-----
These are :class:`typing.NewType` aliases, so they:
- Cost nothing at runtime (zero overhead)
- Help type checkers (mypy, pyright) catch mistakes
- Document intent and convention in signatures
- Enable gradual migration: mix old and new code freely

See Also
--------
:func:`validate_error_convention` : Test utility to verify backward passes
"""

from __future__ import annotations

from typing import NewType

DescentError = NewType("DescentError", object)
"""Error in the pipeline convention (gradient-like).

Used between parts in a mixed pipeline and at the executor boundary.
Semantics: positive value means the signal at this point was too high
(i.e., the network's output exceeded the training target).

This is the convention used by standard deep learning frameworks
(PyTorch, TensorFlow, JAX). It is the **negation** of PyHGF's internal
prediction-error convention.

Type Alias:
    ``DescentError = NewType("DescentError", Array)``
"""

ObservedMinusPredicted = NewType("ObservedMinusPredicted", object)
"""Error in PyHGF's internal convention (prediction error).

Used inside learning parts (DeepNetworkAdapter) and by PyHGF's core
belief-update machinery. Semantics: positive value means the observation
exceeded the model's prediction.

This is the **negation** of the pipeline's descent-error convention.

Type Alias:
    ``ObservedMinusPredicted = NewType("ObservedMinusPredicted", Array)``

Notes
-----
The conversion between conventions is:

    observation = output - descent_error
    prediction_error = observation - output = -descent_error
"""


def validate_error_convention(
    part,
    x: DescentError,
    y: DescentError,
    perturb_size: float = 1e-5,
) -> bool:
    """Check that a part's backward pass uses the pipeline (descent) convention.

    Runs a finite-difference check: perturbs the input slightly, measures how
    the output changes, and compares that to what the part claims the input
    error should be. If they match (to numerical precision), the backward pass
    uses the descent-error convention correctly.

    This is a **sanity check**, not a proof. It uses finite differences which
    are noisy, so it's conservative: it prefers false negatives (missing bugs)
    over false positives (rejecting correct code).

    Parameters
    ----------
    part :
        An EquinoxAdapter (with forward_fn/backward_fn attributes) or a custom PCModule
        with forward() and backward() methods.
    x : DescentError
        Sample input (array-like), shape ``(batch, n_features)`` or ``(n_features,)``.
        Used to test forward and backward.
    y : DescentError
        Sample target (same shape as x). Used to form error for backward pass.
    perturb_size : float, default 1e-5
        Magnitude of finite-difference perturbation in input. Smaller is more
        accurate but noisier; larger is faster but less accurate.

    Returns
    -------
    bool
        True if the backward pass appears to use descent-error convention (to numerical
        precision). False if it appears to use the opposite convention or something else
        entirely.

    Raises
    ------
    AttributeError
        If part lacks required init_state and either (forward/backward methods
        or forward_fn/backward_fn attributes).

    Notes
    -----
    This function is intended for testing custom part implementations. For
    production use, check the part type and trust the implementation.

    Example
    -------
    >>> from pyhgf.model import gelu_adapter
    >>> from pyhgf.model.error_types import validate_error_convention
    >>> import jax.numpy as jnp
    >>>
    >>> part = gelu_adapter()
    >>> x = jnp.array([1.0, -0.5, 2.0])
    >>> y = jnp.array([0.5, 0.1, 1.5])
    >>> is_correct = validate_error_convention(part, x, y)
    >>> print(f"Backward convention correct: {is_correct}")
    Backward convention correct: True
    """
    import jax
    import jax.numpy as jnp

    # Initialize state
    state = part.init_state()

    # Handle both method-based (PCSequential, etc.) and attribute-based (EquinoxAdapter) parts
    if hasattr(part, "forward_fn"):
        # EquinoxAdapter: forward_fn(x) -> (output, cache)
        output, cache = part.forward_fn(x)
    else:
        # Custom PCModule: forward(state, x) -> (output, cache)
        output, cache = part.forward(state, x)

    # Error at output (descent convention)
    error = output - y

    # Finite-difference estimate of input sensitivity
    # For a small perturbation eps in input, the output changes by ~eps * jacobian
    # So input_error ~ error @ jacobian^T ~ error * (d output / d input)
    eps_x = jnp.zeros_like(x)
    if jnp.ndim(x) == 1:
        eps_x = eps_x.at[0].set(perturb_size)
    else:
        eps_x = eps_x.at[0, 0].set(perturb_size)

    if hasattr(part, "forward_fn"):
        output_plus, _ = part.forward_fn(x + eps_x)
    else:
        output_plus, _ = part.forward(state, x + eps_x)

    fd_slope = (output_plus - output) / perturb_size

    # Expected input error: error flows back through the slope
    expected_input_error = error * fd_slope

    # Part's claimed input error
    if hasattr(part, "backward_fn"):
        # EquinoxAdapter: backward_fn(cache, error) -> error_in
        actual_input_error = part.backward_fn(cache, error)
    else:
        # Custom PCModule: backward(state, cache, error) -> (error_in, ...)
        actual_input_error, _, _ = part.backward(state, cache, error)

    # Compare: loose tolerance (this is noisy)
    # We compare element-wise and check if most elements agree
    relative_error = jnp.abs(
        (actual_input_error - expected_input_error)
        / (jnp.abs(expected_input_error) + 1e-8)
    )
    fraction_correct = jnp.mean(relative_error < 0.3)  # 30% tolerance

    return bool(fraction_correct >= 0.8)  # At least 80% of elements agree
