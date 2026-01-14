# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
import jax.numpy as jnp

from pyhgf.math import (
    MultivariateNormal,
    Normal,
    binary_surprise,
    binary_surprise_finite_precision,
    gaussian_predictive_distribution,
    gaussian_surprise,
    sigmoid_inverse_temperature,
)


def test_binary_surprise():
    """Test binary surprise."""
    s = binary_surprise(x=1.0, expected_mean=0.8, clipping=True)
    assert jnp.isclose(s, 0.22314355)

    s = binary_surprise(x=0.0, expected_mean=0.8, clipping=True)
    assert jnp.isclose(s, 1.60943791)

    s = binary_surprise(x=0.0, expected_mean=0.0, clipping=True)
    assert jnp.isclose(s, 1.0000005e-06)

    s = binary_surprise(x=0.0, expected_mean=1.0, clipping=True)
    assert jnp.isclose(s, 13.81551056)

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
