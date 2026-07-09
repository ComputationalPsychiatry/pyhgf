# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests for error convention type annotations and validation.

Validates that:
1. Error type markers are defined and importable
2. validate_error_convention detects correct implementations
3. The validation catches sign errors (or at least detects unusual patterns)
4. Type annotations document the convention correctly
"""

from __future__ import annotations

import jax.nn
import jax.numpy as jnp
import pytest

from pyhgf.model import (
    DescentError,
    ObservedMinusPredicted,
    gelu_adapter,
    layer_norm_adapter,
    validate_error_convention,
)
from pyhgf.model.error_types import validate_error_convention as direct_validate


class _IdentityModule:
    """Minimal method-based part (forward/backward) using the descent convention.

    Exercises the ``part.forward(state, x)`` / ``part.backward(state, cache, error)``
    branches of :func:`validate_error_convention`, i.e. the path for custom
    :class:`~pyhgf.model.hybrid.PCModule` implementations rather than the attribute-
    based :class:`~pyhgf.model.hybrid.EquinoxAdapter`.

    The forward pass is the identity (slope 1), so the descent-convention backward pass
    returns the output error unchanged.
    """

    def init_state(self):
        return ()

    def forward(self, state, x):
        return x, ()

    def backward(self, state, cache, error):
        return error, state, ()


class _SignFlippedModule(_IdentityModule):
    """Same as :class:`_IdentityModule` but the backward pass negates the error.

    This is the classic convention bug: returning an observed-minus-predicted
    error where a descent error is expected. :func:`validate_error_convention`
    should reject it.
    """

    def backward(self, state, cache, error):
        return -error, state, ()


class TestErrorTypeMarkers:
    """Test that error type markers exist and are properly defined."""

    def test_descent_error_exists(self):
        """DescentError is defined and importable."""
        assert DescentError is not None

    def test_observed_minus_predicted_exists(self):
        """ObservedMinusPredicted is defined and importable."""
        assert ObservedMinusPredicted is not None

    def test_error_types_are_distinct(self):
        """Error types are distinct (different NewType instances)."""
        # NewTypes are callable, and calling them returns the value unchanged
        x = jnp.array([1.0, 2.0])
        descent = DescentError(x)
        obs_minus_pred = ObservedMinusPredicted(x)

        # Both are the same array at runtime, but types are different
        assert jnp.array_equal(descent, obs_minus_pred)
        # Type names are distinct
        assert DescentError.__name__ == "DescentError"
        assert ObservedMinusPredicted.__name__ == "ObservedMinusPredicted"


class TestValidateErrorConvention:
    """Test error convention validation utility."""

    def test_validate_gelu_adapter_correct(self):
        """GELU adapter's backward formula uses descent-error convention."""
        part = gelu_adapter()

        # Simple test case
        x = DescentError(jnp.array([0.5, -0.2, 1.5]))
        y = DescentError(jnp.array([0.3, -0.1, 1.0]))

        # GELU's hand-derived backward matches the autodiff VJP of its forward.
        assert validate_error_convention(part, x, y) is True

    def test_validate_returns_bool(self):
        """Validation function returns a plain Python bool."""
        part = gelu_adapter()

        x = DescentError(jnp.array([1.0, -2.0, 3.0]))
        y = DescentError(jnp.array([0.5, -1.0, 1.5]))

        result = validate_error_convention(part, x, y)
        assert isinstance(result, bool)

    def test_validate_method_based_part_correct(self):
        """Method-based parts (forward/backward) validate under descent convention."""
        part = _IdentityModule()

        x = DescentError(jnp.array([1.0, -0.5, 2.0]))
        y = DescentError(jnp.array([0.5, 0.1, 1.5]))

        assert validate_error_convention(part, x, y) is True

    def test_validate_batched_input(self):
        """Validation handles multi-dimensional (batched) input."""
        part = _IdentityModule()

        x = DescentError(jnp.array([[1.0, -0.5, 2.0], [0.2, 1.3, -1.0]]))
        y = DescentError(jnp.array([[0.5, 0.1, 1.5], [0.0, 1.0, -0.5]]))

        assert validate_error_convention(part, x, y) is True

    def test_validate_detects_sign_error(self):
        """A backward pass with the wrong (negated) sign fails validation.

        The error is caught even though only the sign is wrong and every other aspect of
        the backward pass is correct.
        """
        part = _SignFlippedModule()

        x = DescentError(jnp.array([1.0, -0.5, 2.0]))
        y = DescentError(jnp.array([0.5, 0.1, 1.5]))

        assert validate_error_convention(part, x, y) is False

    def test_validate_layer_norm_non_diagonal_jacobian(self):
        """LayerNorm validates even though its Jacobian is not diagonal.

        LayerNorm couples every output to every input, so the backward pass is a full
        vector-Jacobian product rather than an element-wise scaling. The autodiff
        comparison handles this exactly.
        """
        import equinox as eqx

        part = layer_norm_adapter(eqx.nn.LayerNorm(shape=(4,)))

        x = DescentError(jnp.array([1.0, -0.5, 2.0, 0.3]))
        y = DescentError(jnp.array([0.5, 0.1, 1.5, 0.0]))

        assert validate_error_convention(part, x, y) is True

    def test_validate_missing_method_raises(self):
        """Validation raises AttributeError if part is incomplete."""
        from pyhgf.model.error_types import validate_error_convention as val_fn

        # Object with no init_state method
        class IncompleteAdapter:
            def forward(self, state, x):
                return x, ()

            def backward(self, state, cache, error):
                return error, state, ()

        adapter = IncompleteAdapter()
        x = DescentError(jnp.array([1.0]))
        y = DescentError(jnp.array([0.5]))

        with pytest.raises(AttributeError):
            val_fn(adapter, x, y)


class TestErrorConventionDocumentation:
    """Test that error conventions are properly documented."""

    def test_equinox_adapter_has_error_convention_docs(self):
        """EquinoxAdapter docstring mentions error convention."""
        from pyhgf.model import EquinoxAdapter

        doc = EquinoxAdapter.__doc__
        assert "Error Convention" in doc or "descent" in doc.lower()

    def test_deep_network_adapter_has_boundary_docs(self):
        """DeepNetworkAdapter docstring explains the convention boundary."""
        from pyhgf.model import DeepNetworkAdapter

        doc = DeepNetworkAdapter.__doc__
        assert "Error Convention" in doc or "boundary" in doc.lower()

    def test_error_types_module_has_explanation(self):
        """error_types module docstring explains both conventions."""
        from pyhgf.model import error_types

        doc = error_types.__doc__
        assert "DescentError" in doc
        assert "ObservedMinusPredicted" in doc


class TestConventionSemanticsDocumentation:
    """Test that the semantics are clearly documented in the module."""

    def test_error_types_module_docstring(self):
        """error_types module has comprehensive semantics documentation."""
        from pyhgf.model import error_types

        doc = error_types.__doc__
        assert "DescentError" in doc
        assert "descent" in doc.lower()
        assert "convention" in doc.lower()


class TestTypeConversionConcept:
    """Test that type conversion between conventions is conceptually clear."""

    def test_descent_to_obs_minus_pred_formula_documented(self):
        """The conversion formula is documented."""
        from pyhgf.model.error_types import ObservedMinusPredicted as OMP

        doc = OMP.__doc__
        # Should mention the conversion: -descent = obs - pred
        assert "negation" in doc.lower() or "-" in doc

    def test_conversion_is_negation(self):
        """Converting between conventions is just negation."""
        # This is conceptual; at runtime NewTypes are transparent
        from pyhgf.model.error_types import DescentError as DE
        from pyhgf.model.error_types import ObservedMinusPredicted as OMP

        x = jnp.array([1.0, -2.0])

        # Conceptually: descent_error = x, then obs_minus_pred = -x
        # At runtime they're the same type, but types document intent
        descent = DE(x)
        obs_minus = OMP(-x)

        # They're negations of each other
        assert jnp.allclose(descent, -obs_minus)


class TestBackwardCompatibility:
    """Test that existing code without type hints still works."""

    def test_gelu_adapter_still_works_untyped(self):
        """GELU adapter works regardless of type annotation."""
        part = gelu_adapter()

        # No type annotation
        x = jnp.array([1.0, -0.5])
        y = jnp.array([0.5, -0.2])

        state = part.init_state()
        output, cache = part.forward_fn(x)
        error = output - y
        error_in = part.backward_fn(cache, error)

        # Should just work
        assert error_in.shape == x.shape

    def test_sequential_composition_untyped(self):
        """Sequential parts work regardless of type annotation."""
        from pyhgf.model import PCSequential

        part1 = gelu_adapter()
        part2 = gelu_adapter()
        seq = PCSequential([part1, part2])

        x = jnp.array([1.0, -0.5])
        state = seq.init_state()

        # Should work without explicit types
        assert state is not None
