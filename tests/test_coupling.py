# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Tests of the coupling functions, their derivatives and their Gaussian moments."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from pyhgf.coupling import (
    COUPLINGS,
    Coupling,
    gauss_hermite_moments,
    leaky_relu_moments,
    resolve_coupling_fn,
)


def _numerical_moments(coupling, mean, sd):
    """Moments of a coupling function under a Gaussian, by adaptive integration.

    Returns the mean, the variance and the regression slope Cov[x, g(x)] / Var[x], which
    is the mean slope E[g'(x)] by Stein's lemma.
    """
    density = stats.norm(mean, sd).pdf
    bounds = (mean - 12.0 * sd, mean + 12.0 * sd)
    # the rectified linear units have a kink at 0
    points = [0.0] if bounds[0] < 0.0 < bounds[1] else None

    def expectation(fn):
        return integrate.quad(
            lambda x: fn(x) * density(x), *bounds, points=points, limit=200
        )[0]

    expected_value = expectation(lambda x: float(coupling(x)))
    variance = expectation(lambda x: (float(coupling(x)) - expected_value) ** 2)
    slope = expectation(lambda x: (x - mean) * float(coupling(x))) / sd**2
    return expected_value, variance, slope


@pytest.mark.parametrize("name", ["identity", "relu", "leaky_relu"])
@pytest.mark.parametrize(
    ("mean", "sd"),
    [(-0.75, 1.0), (0.0, 1.0), (0.75, 1.0), (2.0, 0.3), (-1.0, 2.0), (-3.0, 0.5)],
)
def test_closed_form_moments(name, mean, sd):
    """The closed-form moments are the moments of the function under the Gaussian."""
    coupling = COUPLINGS[name]
    assert coupling.closed_form_moments is not None
    np.testing.assert_allclose(
        [float(moment) for moment in coupling.moments(mean, sd**2)],
        _numerical_moments(coupling, mean, sd),
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("name", "atol"),
    [("tanh", 1e-3), ("sigmoid", 1e-3), ("gelu", 1e-3), ("silu", 1e-3), ("elu", 1e-2)],
)
@pytest.mark.parametrize(("mean", "sd"), [(-0.75, 1.0), (0.0, 0.5), (2.0, 0.3)])
def test_quadrature_moments(name, atol, mean, sd):
    """The quadrature moments approximate the moments of the function.

    With 20 nodes and a standard deviation of at most 1, the error is below 1e-3 for the
    smooth functions. The second derivative of the exponential linear unit jumps at 0,
    which gives an error of up to 1e-2 on its mean slope.
    """
    coupling = COUPLINGS[name]
    assert coupling.closed_form_moments is None
    np.testing.assert_allclose(
        [float(moment) for moment in coupling.moments(mean, sd**2)],
        _numerical_moments(coupling, mean, sd),
        atol=atol,
    )


def test_quadrature_is_exact_for_polynomials():
    """The Gauss-Hermite rule gives the exact moments of a cubic function."""
    mean, variance = 1.0, 4.0
    expected_value, expected_variance, slope = gauss_hermite_moments(
        lambda x: x**3, lambda x: 3.0 * x**2, mean, variance
    )
    # E[x^3] = mu^3 + 3 mu s^2 and E[3 x^2] = 3 (mu^2 + s^2)
    np.testing.assert_allclose(expected_value, 13.0, rtol=1e-6)
    np.testing.assert_allclose(slope, 15.0, rtol=1e-6)
    # Var[x^3] = E[x^6] - E[x^3]^2, with E[x^6] of a Gaussian
    sixth_moment = (
        mean**6
        + 15 * mean**4 * variance
        + 45 * mean**2 * variance**2
        + 15 * variance**3
    )
    np.testing.assert_allclose(expected_variance, sixth_moment - 13.0**2, rtol=1e-5)


@pytest.mark.parametrize("name", sorted(set(COUPLINGS) - {"linear", "swish"}))
def test_point_belief_limit(name):
    """With a variance of 0, the moments are g(mu), 0 and g'(mu)."""
    coupling = COUPLINGS[name]
    means = jnp.array([-1.5, -0.2, 0.4, 2.0])
    expected_value, variance, slope = coupling.moments(means, jnp.zeros(4))
    np.testing.assert_allclose(expected_value, coupling(means), atol=1e-6)
    np.testing.assert_allclose(variance, 0.0, atol=1e-6)
    np.testing.assert_allclose(slope, coupling.df(means), atol=1e-6)


@pytest.mark.parametrize("name", sorted(COUPLINGS))
def test_derivatives_match_automatic_differentiation(name):
    """The stored derivatives are those of the function, away from the kinks."""
    coupling = COUPLINGS[name]
    x = jnp.array([-2.5, -0.7, 0.3, 1.9])
    np.testing.assert_allclose(
        coupling.df(x), jax.vmap(jax.grad(coupling.f))(x), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        coupling.d2f(x),
        jax.vmap(jax.grad(jax.grad(coupling.f)))(x),
        rtol=1e-5,
        atol=1e-6,
    )
    # the derivatives keep the shape of their argument, scalars included
    assert coupling.df(0.5).shape == ()
    assert coupling.d2f(jnp.zeros((2, 3))).shape == (2, 3)


def test_derivatives_at_the_kink():
    """The derivative at 0 is 0 for the rectified linear unit, 1 for the leaky one."""
    assert float(COUPLINGS["relu"].df(0.0)) == 0.0
    assert float(COUPLINGS["leaky_relu"].df(0.0)) == 1.0


def test_leaky_relu_moments_limits():
    """A slope of 1 gives the identity, and a far mean the linear branches."""
    np.testing.assert_allclose(
        [float(moment) for moment in leaky_relu_moments(0.3, 2.0, 1.0)],
        [0.3, 2.0, 1.0],
        rtol=1e-6,
    )
    # far above 0, the function is the identity; far below, it is s * x
    np.testing.assert_allclose(
        [float(moment) for moment in leaky_relu_moments(50.0, 1.0, 0.01)],
        [50.0, 1.0, 1.0],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        [float(moment) for moment in leaky_relu_moments(-50.0, 1.0, 0.01)],
        [-0.5, 1e-4, 0.01],
        rtol=1e-5,
    )


@pytest.mark.parametrize("name", ["relu", "tanh"])
def test_moments_are_differentiable_at_zero_variance(name):
    """The moments have finite gradients for a point belief."""
    coupling = COUPLINGS[name]
    for index in range(3):
        gradients = jax.grad(
            lambda mean, variance: coupling.moments(mean, variance)[index],
            argnums=(0, 1),
        )(0.3, 0.0)
        assert all(bool(jnp.isfinite(gradient)) for gradient in gradients)


def test_resolve_coupling_fn():
    """Names, functions and couplings resolve to couplings created once."""
    assert resolve_coupling_fn(None) is None
    assert resolve_coupling_fn("identity") is resolve_coupling_fn("linear")
    assert resolve_coupling_fn("tanh") is COUPLINGS["tanh"]
    assert resolve_coupling_fn(COUPLINGS["gelu"]) is COUPLINGS["gelu"]

    # the functions of jax with an analytic entry resolve to this entry
    assert resolve_coupling_fn(jnp.tanh) is COUPLINGS["tanh"]
    assert resolve_coupling_fn(jax.nn.relu) is COUPLINGS["relu"]
    assert resolve_coupling_fn(jax.nn.leaky_relu) is COUPLINGS["leaky_relu"]
    assert resolve_coupling_fn(jax.nn.sigmoid) is COUPLINGS["sigmoid"]
    assert resolve_coupling_fn(jax.nn.gelu) is COUPLINGS["gelu"]

    with pytest.raises(ValueError, match="Unknown coupling function"):
        resolve_coupling_fn("sigmiod")
    with pytest.raises(TypeError):
        resolve_coupling_fn(123)


def test_from_callable_returns_one_object_per_function():
    """Wrapping the same function twice returns the same coupling."""

    def cubic(x):
        return x**3

    coupling = Coupling.from_callable(cubic)
    assert coupling is Coupling.from_callable(cubic)
    assert coupling is resolve_coupling_fn(cubic)
    assert coupling.name == "cubic"
    np.testing.assert_allclose(coupling(2.0), 8.0)
    np.testing.assert_allclose(coupling.df(2.0), 12.0)
    np.testing.assert_allclose(coupling.d2f(2.0), 12.0)

    # another number of quadrature nodes is another coupling
    finer = Coupling.from_callable(cubic, n_points=40)
    assert finer is not coupling
    assert finer.n_points == 40
    assert finer is Coupling.from_callable(cubic, n_points=40)


def test_coupling_copies_are_the_same_object():
    """A copy of a coupling is the coupling, so that it compares equal to it."""
    coupling = COUPLINGS["tanh"]
    assert copy.copy(coupling) is coupling
    assert copy.deepcopy(coupling) is coupling
    assert copy.deepcopy((coupling, None))[0] is coupling
    assert repr(coupling) == "Coupling(tanh)"
