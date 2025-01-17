# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Tuple, Union

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma, gamma, gammaln
from jax.typing import ArrayLike


class MultivariateNormal:
    """The multivariate normal as an exponential family distribution.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Exponential_family

    """

    @staticmethod
    def sufficient_statistics_from_observations(x: ArrayLike) -> Array:
        """Compute the expected sufficient statistics from a single observation."""
        return jnp.hstack([x, jnp.outer(x, x)[jnp.tril_indices(x.shape[0])]])

    @staticmethod
    def sufficient_statistics_from_parameters(
        mean: ArrayLike, covariance: ArrayLike
    ) -> Array:
        """Compute the expected sufficient statistics from distribution parameter.

        Parameters
        ----------
        mean :
            Mean of the Gaussian distribution.
        covariance :
            Variance of the Gaussian distribution.

        Returns
        -------
        xis :
            The sufficient statistics.

        """
        return jnp.append(
            mean,
            (covariance + jnp.outer(mean, mean))[jnp.tril_indices(covariance.shape[0])],
        )

    @staticmethod
    def base_measure(k: int) -> float:
        """Compute the base measures for the multivariate normal."""
        return (2 * jnp.pi) ** (-k / 2)

    @staticmethod
    def parameters_from_sufficient_statistics(
        xis: ArrayLike, dimension: int
    ) -> Tuple[Array, Array]:
        """Compute the distribution parameters from the sufficient statistics.

        Parameters
        ----------
        xis :
            The sufficient statistics.
        dimension :
            The dimension of the multivariate normal distribution.

        Returns
        -------
        means, covariance :
            The parameters of the distribution (mean and covariance).

        """
        mean = xis[:dimension]
        covariance = jnp.zeros((dimension, dimension))
        covariance = covariance.at[jnp.tril_indices(dimension)].set(
            xis[dimension:] - jnp.outer(mean, mean)[jnp.tril_indices(dimension)]
        )
        covariance += covariance.T - jnp.diag(covariance.diagonal())

        return mean, covariance


class Normal:
    """The univariate normal as an exponential family distribution.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Exponential_family

    """

    @staticmethod
    def sufficient_statistics_from_observations(x: float) -> Array:
        """Compute the expected sufficient statistics from a single observation."""
        return jnp.array([x, x**2])

    @staticmethod
    def sufficient_statistics_from_parameters(mean: float, variance: float) -> Array:
        """Compute the expected sufficient statistics from distribution parameter.

        Parameters
        ----------
        mean :
            Mean of the Gaussian distribution.
        variance :
            Variance of the Gaussian distribution.

        Returns
        -------
        xis :
            The sufficient statistics.

        """
        return jnp.array([mean, mean**2 + variance])

    @staticmethod
    def base_measure() -> float:
        """Compute the base measure of the univariate normal."""
        return 1 / (jnp.sqrt(2 * jnp.pi))

    @staticmethod
    def parameters_from_sufficient_statistics(xis: ArrayLike) -> Tuple[float, float]:
        """Compute the distribution parameters from the sufficient statistics.

        Parameters
        ----------
        xis :
            The sufficient statistics.

        Returns
        -------
        mean, variance :
            The parameters of the distribution (mean and variance).

        """
        mean = xis[0]
        variance = xis[1] - (mean**2)

        return mean, variance


def gaussian_predictive_distribution(x: float, xi: ArrayLike, nu: float) -> float:
    r"""Density of the Gaussian-predictive distribution.

    This distribution is parametrized by hyperparameters from the exponential family as:

    .. math::

        \begin{cases}
            \mathcal{NP}(x, \xi, \nu) :=
            \sqrt{\frac{1}{\pi(\nu+1)(\xi_{x^2}-\xi_{x}^2)}}
            \frac{\Gamma(\frac{\nu+2}{2})}{\Gamma(\frac{\nu+1}{2})}
            \left( 1+\frac{(x-\xi_{x})^2}{(\nu+1)(\xi_{x^2}-\xi_x^2)} \right)
            ^{-\frac{\nu+2}{2}}
        \end{cases}

    Parameters
    ----------
    x :
        The point at which the density is evaluated.
    xi :
        Hyperparameter updated by the sufficient statistics of the observed variables.
    nu :
        Hyperparameter over the number of valid observation (pseudo-counts).

    Returns
    -------
    y :
        The probability density evaluated at *x*.

    References
    ----------
    .. [1] Mathys, C., & Weber, L. (2020). Hierarchical Gaussian filtering of sufficient
       statistic time series for active inference. In Communications in Computer and
       Information Science. Active Inference (pp. 52–58).
       doi:10.1007/978-3-030-64919-7_7

    """
    return (
        jnp.sqrt(1 / (jnp.pi * (nu + 1) * (xi[1] - xi[0] ** 2)))
        * jnp.exp(
            (gammaln((nu + 2) / 2) - gammaln((nu + 1) / 2))
        )  # use gammaln to avoid numerical overflow
        * (1 + ((x - xi[0]) ** 2) / ((nu + 1) * (xi[1] - xi[0] ** 2)))
        ** (-(nu + 2) / 2)
    )


def gaussian_density(x: ArrayLike, mean: ArrayLike, precision: ArrayLike) -> ArrayLike:
    """Gaussian density as defined by mean and precision."""
    return precision / jnp.sqrt(2 * jnp.pi) * jnp.exp(-precision / 2 * (x - mean) ** 2)


def sigmoid(
    x: Union[ArrayLike, float],
    lower_bound: Union[ArrayLike, float] = 0.0,
    upper_bound: Union[ArrayLike, float] = 1.0,
) -> ArrayLike:
    """Logistic sigmoid function."""
    return (upper_bound - lower_bound) / (1 + jnp.exp(-x)) + lower_bound


def binary_surprise(
    x: Union[float, ArrayLike], expected_mean: Union[float, ArrayLike]
) -> ArrayLike:
    r"""Surprise at a binary outcome.

    The surprise ellicited by a binary observation :math:`x` under the expected
    probability :math:`\hat{\mu}` is given by:

    .. math::

       \begin{cases}
            -\log(\hat{\mu}),& \text{if } x=1\\
            -\log(1 - \hat{\mu}), & \text{if } x=0\\
        \end{cases}

    Parameters
    ----------
    x :
        The outcome.
    expected_mean :
        The mean of the Bernoulli distribution.

    Returns
    -------
    surprise :
        The binary surprise.


    Examples
    --------
    >>> from pyhgf.binary import binary_surprise
    >>> binary_surprise(x=1.0, expected_mean=0.7)
    `Array(0.35667497, dtype=float32, weak_type=True)`

    """
    return jnp.where(
        x, -jnp.log(expected_mean), -jnp.log(jnp.array(1.0) - expected_mean)
    )


def gaussian_surprise(
    x: Union[float, ArrayLike],
    expected_mean: Union[float, ArrayLike],
    expected_precision: Union[float, ArrayLike],
) -> Array:
    r"""Surprise at an outcome under a Gaussian prediction.

    The surprise elicited by an observation :math:`x` under a Gaussian distribution
    with expected mean :math:`\hat{\mu}` and expected precision :math:`\hat{\pi}` is
    given by:

    .. math::

       \frac{1}{2} (\log(2 \pi) - \log(\hat{\pi}) + \hat{\pi}(x - \hat{\mu})^2)

    where :math:`\pi` is the mathematical constant.

    Parameters
    ----------
    x :
        The outcome.
    expected_mean :
        The expected mean of the Gaussian distribution.
    expected_precision :
        The expected precision of the Gaussian distribution.

    Returns
    -------
    surprise :
        The Gaussian surprise.

    Examples
    --------
    >>> from pyhgf.math import gaussian_surprise
    >>> gaussian_surprise(x=2.0, expected_mean=0.0, expected_precision=1.0)
    `Array(2.9189386, dtype=float32, weak_type=True)`

    """
    return jnp.array(0.5) * (
        jnp.log(jnp.array(2.0) * jnp.pi)
        - jnp.log(expected_precision)
        + expected_precision * jnp.square(jnp.subtract(x, expected_mean))
    )


def dirichlet_kullback_leibler(alpha_1: ArrayLike, alpha_2: ArrayLike) -> Array:
    r"""Compute the Kullback-Leibler divergence between two Dirichlet distributions.

    The Kullback-Leibler divergence from the distribution :math:`Q` to the distribution
    :math:`P`, two Dirichlet distributions parametrized by :math:`\alpha_2` and
    :math:`\alpha_1` (respectively) is given by the following equation:

    .. math::
       KL[P||Q] = \ln{\frac{\Gamma(\sum_{i=1}^k\alpha_{1i})}
         {\Gamma(\sum_{i=1}^k\alpha_{2i})}} +
         \sum_{i=1}^k \ln{\frac{\Gamma(\alpha_{2i})}{\Gamma(\alpha_{1i})}} +
         \sum_{i=1}^k(\alpha_{1i} -
         \alpha_{2i})\left[\psi(\alpha_{1i})-\psi(\sum_{i=1}^k\alpha_{1i})\right]

    Parameters
    ----------
    alpha_1 :
        The concentration parameters for the distribution :math:`P`.
    alpha_2 :
        The concentration parameters for the distribution :math:`Q`.

    Returns
    -------
    kl :
        The Kullback-Leibler divergence of distribution :math:`P` from distribution
        :math:`Q`.

    References
    ----------
    .. [1] https://statproofbook.github.io/P/dir-kl.html
    .. [2] Penny, William D. (2001): "KL-Divergences of Normal, Gamma, Dirichlet and
       Wishart densities" ; in: University College, London , p. 2, eqs. 8-9 ;
       URL: https://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps .

    """
    return (
        jnp.log(gamma(alpha_1.sum()) / gamma(alpha_2.sum()))
        + jnp.sum(jnp.log(gamma(alpha_2) / gamma(alpha_1)))
        + jnp.sum((alpha_1 - alpha_2) * (digamma(alpha_1) - digamma(alpha_1.sum())))
    )


def binary_surprise_finite_precision(
    value: Union[ArrayLike, float],
    expected_mean: Union[ArrayLike, float],
    expected_precision: Union[ArrayLike, float],
    eta0: Union[ArrayLike, float] = 0.0,
    eta1: Union[ArrayLike, float] = 1.0,
) -> Array:
    r"""Compute the binary surprise with finite precision.

    Parameters
    ----------
    value :
        The observed value.
    expected_mean :
        The expected probability of observing category 1.
    expected_precision :
        The precision of the underlying normal distributions.
    eta0 :
        The first possible value.
    eta1 :
        The second possible value.

    Returns
    -------
    surprise :
        The binary surprise under finite precision.

    """
    return -jnp.log(
        expected_mean * gaussian_density(value, eta1, expected_precision)
        + (1 - expected_mean) * gaussian_density(value, eta0, expected_precision)
    )


def sigmoid_inverse_temperature(x: float, temperature: float) -> float:
    """Compute the sigmoid response function with inverse temperature parameter."""
    return (x**temperature) / (x**temperature + (1 - x) ** temperature)
