# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

r"""Coupling functions with their derivatives and Gaussian moments.

A coupling function :math:`g` links a value parent to its value child: the child is
predicted from :math:`g(x_b)`, with :math:`x_b` the value of the parent. A
:class:`Coupling` holds the function with its first two derivatives and the moments of
:math:`g(x_b)` for a Gaussian belief on :math:`x_b`, so that the update steps read them
instead of differentiating the function.
"""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import ndtr
from jax.typing import ArrayLike


def leaky_relu_moments(
    mean: ArrayLike, variance: ArrayLike, negative_slope: float
) -> tuple[Array, Array, Array]:
    r"""Compute the Gaussian moments of a leaky rectified linear unit in closed form.

    For :math:`x \sim \mathcal{N}(\mu, \sigma^2)` and
    :math:`g(x) = \max(x, 0) + s \min(x, 0)`, with :math:`z = \mu / \sigma`,
    :math:`\varphi` the density and :math:`\Phi` the distribution function of the
    standard normal distribution:

    .. math::

        \mathbb{E}[g(x)] &= s \mu + (1 - s) \, \sigma \,
            \left( \varphi(z) + z \Phi(z) \right), \\
        \mathbb{E}[g'(x)] &= s + (1 - s) \, \Phi(z), \\
        \mathrm{Var}[g(x)] &= \sigma^2 \left( s^2 + 2 s (1 - s) \Phi(z)
            + (1 - s)^2 r(z) \right),

    with :math:`r(z) = z^2 \Phi(z) \Phi(-z) + \Phi(z)
    + z \varphi(z) \left( \Phi(-z) - \Phi(z) \right) - \varphi(z)^2` the variance of
    the rectified linear unit for :math:`\sigma = 1`. The mean of the rectified
    linear unit (:math:`s = 0`) is given in appendix A of [1]_. The negative slope
    :math:`s = 0` gives the rectified linear unit and :math:`s = 1` the identity.

    Parameters
    ----------
    mean :
        The mean :math:`\mu` of the Gaussian.
    variance :
        The variance :math:`\sigma^2` of the Gaussian. With a variance of 0 and
        :math:`\mu \neq 0`, the three moments are :math:`g(\mu)`, 0 and
        :math:`g'(\mu)`.
    negative_slope :
        The slope :math:`s` of the function for negative values.

    Returns
    -------
    mean :
        The mean :math:`\mathbb{E}[g(x)]`.
    variance :
        The variance :math:`\mathrm{Var}[g(x)]`.
    slope :
        The mean slope :math:`\mathbb{E}[g'(x)]`.

    References
    ----------
    .. [1] Wu, A., Nowozin, S., Meeds, E., Turner, R. E., Hernández-Lobato, J. M., &
       Gaunt, A. L. (2019). Deterministic variational inference for robust Bayesian
       neural networks. International Conference on Learning Representations.
       arXiv:1810.03958
    """
    mean = jnp.asarray(mean, dtype=float)
    variance = jnp.maximum(jnp.asarray(variance, dtype=float), 0.0)
    # The floor keeps the square root differentiable at a variance of 0, and the
    # clipping of z gives the limits of a point belief without forming 0 * inf.
    tiny = jnp.finfo(mean.dtype).tiny
    sigma = jnp.sqrt(jnp.maximum(variance, tiny))
    z = jnp.clip(mean / sigma, -40.0, 40.0)
    cdf, survival = ndtr(z), ndtr(-z)
    pdf = jnp.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    slope = negative_slope + (1.0 - negative_slope) * cdf
    expected_value = negative_slope * mean + (1.0 - negative_slope) * (
        mean * cdf + sigma * pdf
    )
    # Variance of the rectified linear unit for a unit variance, written without the
    # cancellation of E[g^2] - E[g]^2.
    relu_variance = z**2 * cdf * survival + cdf + z * pdf * (survival - cdf) - pdf**2
    expected_variance = variance * (
        negative_slope**2
        + 2.0 * negative_slope * (1.0 - negative_slope) * cdf
        + (1.0 - negative_slope) ** 2 * relu_variance
    )
    return expected_value, jnp.maximum(expected_variance, 0.0), slope


def gauss_hermite_moments(
    f: Callable,
    df: Callable,
    mean: ArrayLike,
    variance: ArrayLike,
    n_points: int = 20,
) -> tuple[Array, Array, Array]:
    r"""Compute the Gaussian moments of a function by Gauss-Hermite quadrature.

    For :math:`x \sim \mathcal{N}(\mu, \sigma^2)`, the expectation of a function
    :math:`h` is approximated with the :math:`n` nodes :math:`\xi_i` and weights
    :math:`w_i` of the probabilists' Gauss-Hermite rule:

    .. math::

        \mathbb{E}[h(x)] \approx \sum_{i=1}^{n} w_i \, h(\mu + \sigma \xi_i).

    The rule is applied to :math:`g`, to :math:`(g - \mathbb{E}[g])^2` and to
    :math:`g'`. It is exact when :math:`h` is a polynomial of degree at most
    :math:`2n - 1`. Its error grows with :math:`\sigma` for a function that
    saturates: with 20 nodes, the largest error on the moments of the hyperbolic
    tangent for :math:`\mu \in [-3, 3]` is below :math:`10^{-4}` for
    :math:`\sigma \leq 1` and of the order of :math:`10^{-2}` for :math:`\sigma = 2`.

    Parameters
    ----------
    f :
        The function :math:`g`, applied elementwise.
    df :
        The derivative :math:`g'`, applied elementwise.
    mean :
        The mean :math:`\mu` of the Gaussian.
    variance :
        The variance :math:`\sigma^2` of the Gaussian.
    n_points :
        The number :math:`n` of quadrature nodes.

    Returns
    -------
    mean :
        The mean :math:`\mathbb{E}[g(x)]`.
    variance :
        The variance :math:`\mathrm{Var}[g(x)]`.
    slope :
        The mean slope :math:`\mathbb{E}[g'(x)]`.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_points)
    weights = weights / np.sqrt(2.0 * np.pi)

    mean = jnp.asarray(mean, dtype=float)
    variance = jnp.maximum(jnp.asarray(variance, dtype=float), 0.0)
    # The floor keeps the square root differentiable at a variance of 0.
    sigma = jnp.sqrt(jnp.maximum(variance, jnp.finfo(mean.dtype).tiny))
    points = mean[..., None] + sigma[..., None] * jnp.asarray(nodes, dtype=mean.dtype)
    weights = jnp.asarray(weights, dtype=mean.dtype)

    values = f(points)
    expected_value = jnp.sum(weights * values, axis=-1)
    expected_variance = jnp.sum(
        weights * (values - expected_value[..., None]) ** 2, axis=-1
    )
    slope = jnp.sum(weights * df(points), axis=-1)
    return expected_value, expected_variance, slope


class Coupling:
    r"""A coupling function with its derivatives and its Gaussian moments.

    The object is callable and returns :math:`g(x)`, so it is used wherever a
    coupling function is expected. Two objects are equal only when they are the same
    object. A network stores its coupling functions as static fields of its compiled
    update functions: reusing one object across networks reuses the compiled
    functions, and a different object compiles them again. The couplings of
    :data:`COUPLINGS` are created once, and :meth:`from_callable` returns the same
    object for the same function.

    Parameters
    ----------
    name :
        The name of the coupling function.
    f :
        The function :math:`g`, applied elementwise to an array.
    df :
        The first derivative :math:`g'`, applied elementwise to an array.
    d2f :
        The second derivative :math:`g''`, applied elementwise to an array.
    moments :
        A function of the mean and the variance of a Gaussian that returns the mean
        :math:`\mathbb{E}[g(x)]`, the variance :math:`\mathrm{Var}[g(x)]` and the
        mean slope :math:`\mathbb{E}[g'(x)]` in closed form. If ``None`` (default),
        the moments are computed by :func:`gauss_hermite_moments`.
    n_points :
        The number of nodes of the Gauss-Hermite quadrature. Only used when
        ``moments`` is ``None``.

    Examples
    --------
    >>> from pyhgf.coupling import resolve_coupling_fn
    >>> coupling = resolve_coupling_fn("relu")
    >>> float(coupling(-1.0)), float(coupling.df(2.0))
    (0.0, 1.0)
    >>> expected_value, variance, slope = coupling.moments(0.0, 1.0)
    >>> round(float(slope), 3)
    0.5
    """

    def __init__(
        self,
        name: str,
        f: Callable,
        df: Callable,
        d2f: Callable,
        moments: Optional[Callable] = None,
        n_points: int = 20,
    ):
        self.name = name
        self.__name__ = name
        self.f = f
        self.df = df
        self.d2f = d2f
        self.closed_form_moments = moments
        self.n_points = n_points

    def __call__(self, x: ArrayLike) -> Array:
        """Evaluate the coupling function.

        Parameters
        ----------
        x :
            The values at which the function is evaluated.

        Returns
        -------
        values :
            The values :math:`g(x)`, with the shape of ``x``.
        """
        return self.f(x)

    def __repr__(self) -> str:
        """Return the name of the coupling function."""
        return f"Coupling({self.name})"

    def __copy__(self) -> "Coupling":
        """Return the object itself, so that a copy compares equal to it."""
        return self

    def __deepcopy__(self, memo: dict) -> "Coupling":
        """Return the object itself, so that a copy compares equal to it."""
        return self

    def moments(
        self, mean: ArrayLike, variance: ArrayLike
    ) -> tuple[Array, Array, Array]:
        r"""Compute the moments of the coupling function under a Gaussian.

        For :math:`x \sim \mathcal{N}(\mu, \sigma^2)`, the mean slope is the slope
        of the linear function of :math:`x` that is the closest to :math:`g(x)` in
        mean squared error (Stein's lemma):

        .. math::

            \mathbb{E}[g'(x)] = \frac{\mathrm{Cov}[x, g(x)]}{\sigma^2}.

        Parameters
        ----------
        mean :
            The mean :math:`\mu` of the Gaussian.
        variance :
            The variance :math:`\sigma^2` of the Gaussian, with the shape of
            ``mean``.

        Returns
        -------
        mean :
            The mean :math:`\mathbb{E}[g(x)]`.
        variance :
            The variance :math:`\mathrm{Var}[g(x)]`.
        slope :
            The mean slope :math:`\mathbb{E}[g'(x)]`.
        """
        if self.closed_form_moments is not None:
            return self.closed_form_moments(mean, variance)
        return gauss_hermite_moments(self.f, self.df, mean, variance, self.n_points)

    @classmethod
    def from_callable(cls, fn: Callable, n_points: int = 20) -> "Coupling":
        r"""Wrap a function in a coupling with derivatives from automatic differentiation.

        The function is applied elementwise to an array. Its derivatives are
        computed by automatic differentiation, one element at a time, and its
        moments by Gauss-Hermite quadrature. The same function and number of nodes return the
        same object, so that networks built from one function share their compiled
        update functions. A function of :mod:`jax` that has an entry in
        :data:`COUPLINGS` returns this entry when ``n_points`` is the default.

        Parameters
        ----------
        fn :
            The coupling function, or a :class:`Coupling`, which is returned
            unchanged.
        n_points :
            The number of nodes of the Gauss-Hermite quadrature.

        Returns
        -------
        coupling :
            The coupling.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from pyhgf.coupling import Coupling
        >>> coupling = Coupling.from_callable(jnp.sin)
        >>> coupling is Coupling.from_callable(jnp.sin)
        True
        >>> round(float(coupling.df(0.0)), 3)
        1.0
        """
        if isinstance(fn, cls):
            return fn
        try:
            cached = FROM_CALLABLE_CACHE.get((fn, n_points))
        except TypeError:  # the function cannot be hashed and is not cached
            cached = None
        if cached is not None:
            return cached

        def df(x):
            x = jnp.asarray(x, dtype=float)
            return jax.vmap(jax.grad(fn))(x.ravel()).reshape(x.shape)

        def d2f(x):
            x = jnp.asarray(x, dtype=float)
            return jax.vmap(jax.grad(jax.grad(fn)))(x.ravel()).reshape(x.shape)

        coupling = cls(
            name=getattr(fn, "__name__", type(fn).__name__),
            f=fn,
            df=df,
            d2f=d2f,
            n_points=n_points,
        )
        try:
            FROM_CALLABLE_CACHE[(fn, n_points)] = coupling
        except TypeError:
            pass
        return coupling


#: The couplings returned by :meth:`Coupling.from_callable`, by function and number
#: of quadrature nodes.
FROM_CALLABLE_CACHE: dict[tuple[Callable, int], Coupling] = {}

#: The identity coupling, :math:`g(x) = x`, with closed-form moments.
IDENTITY = Coupling(
    name="identity",
    f=lambda x: x,
    df=lambda x: jnp.ones_like(jnp.asarray(x, dtype=float)),
    d2f=lambda x: jnp.zeros_like(jnp.asarray(x, dtype=float)),
    moments=lambda mean, variance: (
        jnp.asarray(mean, dtype=float),
        jnp.asarray(variance, dtype=float),
        jnp.ones_like(jnp.asarray(mean, dtype=float)),
    ),
)

#: The rectified linear unit, :math:`g(x) = \max(x, 0)`, with :math:`g'(0) = 0` and
#: closed-form moments.
RELU = Coupling(
    name="relu",
    f=jax.nn.relu,
    df=lambda x: jnp.where(jnp.asarray(x, dtype=float) > 0.0, 1.0, 0.0),
    d2f=lambda x: jnp.zeros_like(jnp.asarray(x, dtype=float)),
    moments=partial(leaky_relu_moments, negative_slope=0.0),
)

#: The leaky rectified linear unit with a negative slope of 0.01, with
#: :math:`g'(0) = 1` and closed-form moments.
LEAKY_RELU = Coupling(
    name="leaky_relu",
    f=jax.nn.leaky_relu,
    df=lambda x: jnp.where(jnp.asarray(x, dtype=float) >= 0.0, 1.0, 0.01),
    d2f=lambda x: jnp.zeros_like(jnp.asarray(x, dtype=float)),
    moments=partial(leaky_relu_moments, negative_slope=0.01),
)

#: The hyperbolic tangent, with analytic derivatives and moments by quadrature.
TANH = Coupling(
    name="tanh",
    f=jnp.tanh,
    df=lambda x: (1.0 + jnp.tanh(x)) * (1.0 - jnp.tanh(x)),
    d2f=lambda x: -2.0 * jnp.tanh(x) * (1.0 + jnp.tanh(x)) * (1.0 - jnp.tanh(x)),
)

#: The logistic sigmoid, with analytic derivatives and moments by quadrature.
SIGMOID = Coupling(
    name="sigmoid",
    f=jax.nn.sigmoid,
    df=lambda x: jax.nn.sigmoid(x) * (1.0 - jax.nn.sigmoid(x)),
    d2f=lambda x: (
        jax.nn.sigmoid(x) * (1.0 - jax.nn.sigmoid(x)) * (1.0 - 2.0 * jax.nn.sigmoid(x))
    ),
)

#: The coupling functions that can be named with a string. ``"linear"`` is a second
#: name of the identity. The identity, the rectified linear unit and the leaky
#: rectified linear unit have analytic derivatives and closed-form moments; the
#: hyperbolic tangent and the logistic sigmoid have analytic derivatives and moments
#: by quadrature; the others have derivatives from automatic differentiation and
#: moments by quadrature.
COUPLINGS: dict[str, Coupling] = {
    "identity": IDENTITY,
    "linear": IDENTITY,
    "relu": RELU,
    "leaky_relu": LEAKY_RELU,
    "tanh": TANH,
    "sigmoid": SIGMOID,
    "gelu": Coupling.from_callable(jax.nn.gelu),
    "elu": Coupling.from_callable(jax.nn.elu),
    "silu": Coupling.from_callable(jax.nn.silu),
    "swish": Coupling.from_callable(jax.nn.silu),
}

# A function of jax with an analytic entry resolves to this entry.
FROM_CALLABLE_CACHE.update({
    (jax.nn.relu, 20): RELU,
    (jax.nn.leaky_relu, 20): LEAKY_RELU,
    (jnp.tanh, 20): TANH,
    (jax.nn.tanh, 20): TANH,
    (jax.nn.sigmoid, 20): SIGMOID,
})


def resolve_coupling_fn(
    name_or_fn: Optional[str | Callable],
) -> Optional[Coupling]:
    """Resolve the name of a coupling function, or a function, to a coupling.

    Parameters
    ----------
    name_or_fn :
        The name of a coupling function (a key of :data:`COUPLINGS`), a function,
        a :class:`Coupling`, or ``None``.

    Returns
    -------
    coupling :
        The entry of :data:`COUPLINGS` for a name, the coupling returned by
        :meth:`Coupling.from_callable` for a function, and ``None`` for ``None``.

    Raises
    ------
    ValueError
        If the name is not a key of :data:`COUPLINGS`.
    TypeError
        If the argument is not ``None``, a string or a callable.

    Examples
    --------
    >>> from pyhgf.coupling import resolve_coupling_fn
    >>> resolve_coupling_fn("gelu")
    Coupling(gelu)
    >>> resolve_coupling_fn("identity") is resolve_coupling_fn("identity")
    True
    >>> resolve_coupling_fn(None) is None
    True
    """
    if name_or_fn is None:
        return None

    if isinstance(name_or_fn, str):
        if name_or_fn not in COUPLINGS:
            raise ValueError(
                f"Unknown coupling function: '{name_or_fn}'. "
                f"Valid names: {sorted(COUPLINGS.keys())}. "
                f"Alternatively, pass a callable directly."
            )
        return COUPLINGS[name_or_fn]

    if callable(name_or_fn):
        return Coupling.from_callable(name_or_fn)

    raise TypeError(
        f"coupling_fn must be None, a string name, or a callable; "
        f"got {type(name_or_fn).__name__}"
    )
