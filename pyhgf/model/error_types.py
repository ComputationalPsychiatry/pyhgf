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
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Check that a part's backward pass uses the pipeline (descent) convention.

    A descent-convention backward pass must map the error at the output onto the
    error at the input by the vector-Jacobian product ``Jᵀ @ error``, where
    ``J = ∂output/∂input``. This is exactly what reverse-mode autodiff computes,
    so the check compares the part's hand-derived ``backward`` against the VJP
    that :func:`jax.vjp` produces for its ``forward``.

    The comparison is exact (up to floating-point tolerance) and works for any
    input shape, so it catches a flipped sign or a wrong Jacobian anywhere in the
    array. If the backward instead used PyHGF's observed-minus-predicted
    convention, every element would be negated and the check would fail.

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
    rtol : float, default 1e-4
        Relative tolerance for the element-wise comparison against the autodiff VJP.
    atol : float, default 1e-5
        Absolute tolerance for the element-wise comparison against the autodiff VJP.

    Returns
    -------
    bool
        True if the backward pass matches the autodiff VJP (descent-error
        convention). False if it uses the opposite convention or a wrong Jacobian.

    Raises
    ------
    AttributeError
        If part lacks required init_state and either (forward/backward methods
        or forward_fn/backward_fn attributes).

    Notes
    -----
    This function is intended for testing custom part implementations. For
    production use, check the part type and trust the implementation. The
    ``forward`` pass must be differentiable by JAX for the VJP to be available.

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

    # Handle both attribute-based (EquinoxAdapter, forward_fn/backward_fn) and
    # method-based (custom PCModule, forward/backward) parts. In both cases
    # ``forward`` returns only the output, so it can be differentiated on its own.
    if hasattr(part, "forward_fn"):
        forward = lambda inp: part.forward_fn(inp)[0]  # noqa: E731
        _, cache = part.forward_fn(x)
    else:
        forward = lambda inp: part.forward(state, inp)[0]  # noqa: E731
        _, cache = part.forward(state, x)

    # Output and the reverse-mode VJP of the forward pass.
    output, vjp_fn = jax.vjp(forward, x)

    # Error at the output in the descent convention (positive = output too high).
    error = output - y

    # Autodiff ground truth: the input error a descent-convention backward pass
    # must produce is the vector-Jacobian product Jᵀ @ error.
    (expected_input_error,) = vjp_fn(error)

    # The part's hand-derived input error.
    if hasattr(part, "backward_fn"):
        # EquinoxAdapter: backward_fn(cache, error) -> error_in
        actual_input_error = part.backward_fn(cache, error)
    else:
        # Custom PCModule: backward(state, cache, error) -> (error_in, ...)
        actual_input_error, _, _ = part.backward(state, cache, error)

    return bool(
        jnp.allclose(actual_input_error, expected_input_error, rtol=rtol, atol=atol)
    )
