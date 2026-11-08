# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from pyhgf.math import (
    MultivariateNormal,
    Normal,
    binary_surprise,
    binary_surprise_finite_precision,
    dirichlet_kullback_leibler,
    gaussian_density,
    gaussian_predictive_distribution,
    gaussian_surprise,
    lambert_w0,
    lambert_w0_log,
    sigmoid_inverse_temperature,
)


def test_binary_surprise():
    """Test binary surprise."""
    s = binary_surprise(x=1.0, expected_mean=0.8, clipping=True)
    assert jnp.isclose(s, 0.22314355)

    s = binary_surprise(x=0.0, expected_mean=0.8, clipping=True)
    assert jnp.isclose(s, 1.60943791)

    s = binary_surprise(x=0.0, expected_mean=0.0, clipping=True)
    assert jnp.isclose(s, 1.0132795e-06)

    s = binary_surprise(x=0.0, expected_mean=1.0, clipping=True)
    assert jnp.isclose(s, 13.802319)

    s = binary_surprise(x=0.0, expected_mean=1.0, clipping=False)
    assert jnp.isinf(s)

    s = binary_surprise(x=0.0, expected_mean=1.0, clipping=False)
    assert jnp.isinf(s)


def test_gaussian_surprise():
    """Test the Gaussian surprise function."""
    surprise = gaussian_surprise(
        x=jnp.array([1.0, 1.0]),
        expected_mean=jnp.array([0.0, 0.0]),
        expected_precision=jnp.array([1.0, 1.0]),
    )
    assert jnp.all(jnp.isclose(surprise, 1.4189385))


def test_multivariate_normal():
    """Test the MultivariateNormal node."""
    ss = MultivariateNormal.sufficient_statistics_from_observations(
        jnp.array([1.0, 2.0])
    )
    assert jnp.isclose(ss, jnp.array([1.0, 2.0, 1.0, 2.0, 4.0], dtype="float32")).all()

    bm = MultivariateNormal.base_measure(2)
    assert bm == 0.15915494309189535

    mean = jnp.array([0.0, 1.0])
    covariance = jnp.array([[2.0, 3.0], [3.0, 4.0]])
    ss = MultivariateNormal.sufficient_statistics_from_parameters(mean, covariance)
    assert jnp.isclose(ss, jnp.array([0.0, 1.0, 2.0, 3.0, 5.0], dtype="float32")).all()

    mean, covariance = MultivariateNormal.parameters_from_sufficient_statistics(
        ss, dimension=2
    )
    assert jnp.isclose(mean, jnp.array([0.0, 1.0], dtype="float32")).all()
    assert jnp.isclose(
        covariance, jnp.array([[2.0, 3.0], [3.0, 4.0]], dtype="float32")
    ).all()


def test_normal():
    """Test the Normal node."""
    ss = Normal.sufficient_statistics_from_observations(jnp.array(1.0))
    assert jnp.isclose(ss, jnp.array([1.0, 1.0], dtype="float32")).all()

    bm = Normal.base_measure()
    assert bm == 0.3989423

    ess = Normal.sufficient_statistics_from_parameters(mean=0.0, variance=1.0)
    assert jnp.isclose(ess, jnp.array([0.0, 1.0], dtype="float32")).all()

    par = Normal.parameters_from_sufficient_statistics(xis=[5.0, 29.0])
    assert jnp.isclose(jnp.array(par), jnp.array([5.0, 4.0], dtype="float32")).all()


def test_gaussian_predictive_distribution():
    """Test the Gaussian predictive distribution function."""
    pdf = gaussian_predictive_distribution(x=1.5, xi=[0.0, 1 / 8], nu=5.0)
    assert jnp.isclose(pdf, jnp.array(0.00845728, dtype="float32"))


def test_binary_surprise_finite_precision():
    """Test the binary surprise finite precision function."""
    surprise = binary_surprise_finite_precision(
        value=1.0,
        expected_mean=0.0,
        expected_precision=1.0,
        eta0=0.0,
        eta1=1.0,
    )
    assert surprise == 1.4189385


def test_sigmoid_inverse_temperature():
    """Test the sigmoid inverse temperature function."""
    s = sigmoid_inverse_temperature(x=0.4, temperature=6.0)
    assert jnp.isclose(s, jnp.array(0.08070617906683485, dtype="float32"))


@pytest.mark.parametrize("precision", [0.25, 1.0, 4.0, 100.0])
def test_gaussian_density_integrates_to_one(float_dtype, precision):
    """The total probability mass is one at any precision."""
    mean, sigma = 0.7, 1.0 / np.sqrt(precision)
    grid = np.linspace(mean - 10 * sigma, mean + 10 * sigma, 20001)
    density = gaussian_density(jnp.asarray(grid, dtype=float_dtype), mean, precision)
    mass = integrate.simpson(np.asarray(density, dtype=np.float64), x=grid)
    np.testing.assert_allclose(mass, 1.0, rtol=0.0, atol=2e-6)


def test_gaussian_density_matches_scipy_and_surprise(float_dtype):
    """The density is SciPy's normal density and the exponent of minus the surprise."""
    precision = np.array([0.25, 1.0, 4.0, 100.0])
    mean = np.array([-0.5, 0.0, 0.3, 1.0])
    x = mean + np.array([-1.5, -0.2, 0.7, 1.2]) / np.sqrt(precision)
    args = [jnp.asarray(a, dtype=float_dtype) for a in (x, mean, precision)]
    rtol = 3e-13 if float_dtype == jnp.float64 else 3e-6
    density = gaussian_density(*args)
    expected = stats.norm.pdf(x, loc=mean, scale=1.0 / np.sqrt(precision))
    np.testing.assert_allclose(density, expected, rtol=rtol)
    np.testing.assert_allclose(density, jnp.exp(-gaussian_surprise(*args)), rtol=rtol)


@pytest.mark.parametrize("precision", [0.25, 4.0, 16.0])
def test_binary_surprise_finite_precision_is_a_normal_mixture(float_dtype, precision):
    """The surprise is minus the log density of a normalised two-component mixture."""
    value, probability = np.array([-0.5, 0.5, 1.5]), np.array([0.2, 0.8, 0.5])
    sigma = 1.0 / np.sqrt(precision)
    actual = binary_surprise_finite_precision(
        jnp.asarray(value, dtype=float_dtype),
        jnp.asarray(probability, dtype=float_dtype),
        precision,
    )
    mixture = probability * stats.norm.pdf(value, 1.0, sigma) + (
        1 - probability
    ) * stats.norm.pdf(value, 0.0, sigma)
    np.testing.assert_allclose(actual, -np.log(mixture), rtol=3e-6, atol=3e-7)

    grid = np.linspace(-10 * sigma, 1.0 + 10 * sigma, 20001)
    surprise = binary_surprise_finite_precision(
        jnp.asarray(grid, dtype=float_dtype), 0.3, precision
    )
    mass = integrate.simpson(np.exp(-np.asarray(surprise, dtype=np.float64)), x=grid)
    np.testing.assert_allclose(mass, 1.0, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize("scale", [0.1, 1.0, 100.0, 1e6])
def test_dirichlet_self_kl_is_zero(float_dtype, scale):
    """Identical distributions have zero divergence, including where gamma overflows."""
    alpha = jnp.asarray([scale, scale, 2 * scale], dtype=float_dtype)
    assert float(dirichlet_kullback_leibler(alpha, alpha)) == 0.0


@pytest.mark.parametrize(
    "alpha,beta",
    [
        ([0.2, 0.7], [0.9, 0.3]),
        ([1.0, 1.0], [2.0, 3.0]),
        ([100.0, 100.0], [80.0, 120.0]),
        ([1000.0, 2000.0, 3000.0], [1200.0, 2300.0, 2700.0]),
    ],
)
def test_dirichlet_kl_matches_scipy(float_dtype, alpha, beta):
    """The divergence matches the closed form evaluated with SciPy's log gamma."""
    a, b = np.asarray(alpha), np.asarray(beta)
    expected = (
        special.gammaln(a.sum())
        - special.gammaln(b.sum())
        + np.sum(special.gammaln(b) - special.gammaln(a))
        + np.sum((a - b) * (special.digamma(a) - special.digamma(a.sum())))
    )
    actual = dirichlet_kullback_leibler(
        jnp.asarray(a, dtype=float_dtype), jnp.asarray(b, dtype=float_dtype)
    )
    x64 = float_dtype == jnp.float64
    np.testing.assert_allclose(
        actual, expected, rtol=3e-12 if x64 else 3e-4, atol=2e-11 if x64 else 2e-3
    )


def test_dirichlet_kl_matches_quadrature(float_dtype):
    """A two-category divergence matches the integrated log density ratio."""
    alpha, beta = np.array([2.0, 3.0]), np.array([4.0, 5.0])

    def integrand(x):
        log_p = stats.dirichlet.logpdf([x, 1.0 - x], alpha)
        return np.exp(log_p) * (log_p - stats.dirichlet.logpdf([x, 1.0 - x], beta))

    expected, error = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-11)
    assert error < 1e-8
    actual = dirichlet_kullback_leibler(
        jnp.asarray(alpha, dtype=float_dtype), jnp.asarray(beta, dtype=float_dtype)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def _lambert_arguments(dtype):
    """Zero, one, e and a logarithmic sweep up to near the largest finite float."""
    top = 300 if dtype == jnp.float64 else 30
    values = np.concatenate(([0.0, 1.0, np.e], np.logspace(-12, top, 65)))
    return jnp.asarray(values, dtype=dtype)


def test_lambert_w0_matches_scipy(float_dtype):
    """Values and the custom derivative match the principal branch over the range."""
    z = _lambert_arguments(float_dtype)
    w, dw = jax.jvp(jax.jit(lambert_w0), (z,), (jnp.ones_like(z),))
    expected = special.lambertw(np.asarray(z, dtype=np.float64)).real
    x64 = float_dtype == jnp.float64
    assert w.dtype == float_dtype
    assert w[0] == 0.0 and dw[0] == 1.0
    np.testing.assert_allclose(w, expected, rtol=4e-13 if x64 else 4e-6, atol=0.0)
    np.testing.assert_allclose(
        dw, np.exp(-expected) / (1.0 + expected), rtol=7e-13 if x64 else 7e-6
    )


def test_lambert_w0_log(float_dtype):
    """From log z, W solves log W + W = log z, including where z overflows."""
    z = _lambert_arguments(float_dtype)
    np.testing.assert_array_equal(lambert_w0_log(jnp.log(z)), lambert_w0(z))

    log_z = jnp.asarray([-jnp.inf, -30.0, 0.0, 1.0, 50.0, 300.0, 700.0], float_dtype)
    w, dw = jax.jvp(lambert_w0_log, (log_z,), (jnp.ones_like(log_z),))
    assert w[0] == 0.0 and dw[0] == 0.0
    w = np.asarray(w[1:], dtype=np.float64)
    rtol = 1e-12 if float_dtype == jnp.float64 else 2e-6
    np.testing.assert_allclose(np.log(w) + w, log_z[1:], rtol=rtol, atol=rtol)
    np.testing.assert_allclose(dw[1:], w / (1.0 + w), rtol=rtol)
