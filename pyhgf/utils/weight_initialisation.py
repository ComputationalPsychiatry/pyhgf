# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Weight initialisation strategies for predictive-coding neural networks.

Each function takes the fan-in (*n_parents*) and fan-out (*n_children*) of a layer and
returns a 1-D ``numpy`` array of length ``n_parents * n_children`` (row-major, one
weight per parent→child connection) that can be used as initial
``value_coupling_parents`` / ``value_coupling_children`` vectors.

Available strategies
--------------------
* **Xavier (Glorot)** — :func:`xavier_init`
* **He (Kaiming)** — :func:`he_init`
* **Orthogonal** — :func:`orthogonal_init`
* **Sparse** — :func:`sparse_init`

:func:`_init_matrix` assembles one weight matrix from a strategy, keeping the bias
column out of the statistics.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np


def xavier_init(
    n_parents: int,
    n_children: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""Xavier / Glorot uniform initialisation.

    Draws weights from :math:`\\mathcal{U}(-a, a)` where
    :math:`a = \\sqrt{6 / (n_{\\text{parents}} + n_{\\text{children}})}`.

    Parameters
    ----------
    n_parents :
        Number of parent (input) nodes — fan-in.
    n_children :
        Number of child (output) nodes — fan-out.
    seed :
        Optional random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight vector of length ``n_parents * n_children``.
    """
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (n_parents + n_children))
    return rng.uniform(-limit, limit, size=n_parents * n_children)


def he_init(
    n_parents: int,
    n_children: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""He / Kaiming normal initialisation.

    Draws weights from :math:`\\mathcal{N}(0, \\sigma^2)` where
    :math:`\\sigma = \\sqrt{2 / n_{\\text{parents}}}`.  Designed for layers
    followed by ReLU activations.

    Parameters
    ----------
    n_parents :
        Number of parent (input) nodes — fan-in.
    n_children :
        Number of child (output) nodes — fan-out.
    seed :
        Optional random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight vector of length ``n_parents * n_children``.
    """
    rng = np.random.default_rng(seed)
    std = np.sqrt(2.0 / n_parents)
    return rng.normal(0.0, std, size=n_parents * n_children)


def orthogonal_init(
    n_parents: int,
    n_children: int,
    gain: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Orthogonal initialisation.

    Takes the orthogonal factor of a random matrix's SVD, scaled by *gain*. Every
    singular value is then exactly one, so the layer applies the *same* gain to every
    direction of its input rather than stretching some and squashing others. The returned
    vector is row-major over an ``(n_children, n_parents)`` matrix, which is the shape
    callers reshape it to. Orthogonality means ``W.T @ W = I`` when there are at least as
    many children as parents, so every input direction keeps its length. With fewer children
    than parents the layer is a projection and no matrix can preserve every direction; the
    best available is ``W @ W.T = I``, which preserves length within the row space.

    Parameters
    ----------
    n_parents :
        Number of parent (input) nodes — fan-in.
    n_children :
        Number of child (output) nodes — fan-out.
    gain :
        Multiplicative scaling factor (default 1.0). A norm-preserving nonlinearity
        would need none, but the ReLU family roughly halves the variance passing
        through it, so ``gain=sqrt(2)`` is the usual compensation.
    seed :
        Optional random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight vector of length ``n_parents * n_children``, row-major over an
        ``(n_children, n_parents)`` matrix.
    """
    rng = np.random.default_rng(seed)
    # Draw tall-or-square so ``full_matrices=False`` yields orthonormal columns, then
    # orient the result as (n_children, n_parents) before flattening.
    if n_children >= n_parents:
        a = rng.standard_normal((n_children, n_parents))
        u, _, _ = np.linalg.svd(a, full_matrices=False)
        q = u  # (n_children, n_parents), orthonormal columns
    else:
        a = rng.standard_normal((n_parents, n_children))
        u, _, _ = np.linalg.svd(a, full_matrices=False)
        q = u.T  # (n_children, n_parents), orthonormal rows
    return (gain * q).ravel()


def sparse_init(
    n_parents: int,
    n_children: int,
    sparsity: float = 0.9,
    std: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""Sparse initialisation.

    Most weights are set to zero; only a fraction ``1 - sparsity`` of
    entries are drawn from :math:`\\mathcal{N}(0, \\text{std}^2)`.

    Parameters
    ----------
    n_parents :
        Number of parent (input) nodes — fan-in.
    n_children :
        Number of child (output) nodes — fan-out.
    sparsity :
        Fraction of weights set to zero (default 0.9 → 90 % zeros).
    std :
        Standard deviation of the non-zero entries (default 0.01).
    seed :
        Optional random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight vector of length ``n_parents * n_children``.
    """
    rng = np.random.default_rng(seed)
    size = n_parents * n_children
    weights = np.zeros(size)
    n_nonzero = max(1, int(round((1.0 - sparsity) * size)))
    indices = rng.choice(size, size=n_nonzero, replace=False)
    weights[indices] = rng.normal(0.0, std, size=n_nonzero)
    return weights


def _init_matrix(
    init_fn: Callable,
    n_children: int,
    n_parents: int,
    add_constant_input: bool,
    seed: int,
    kwargs: dict,
) -> jnp.ndarray:
    r"""Draw one ``weights_mean`` matrix, keeping the bias column out of the statistics.

    The bias column is not a connection to a parent, so it takes no part in either
    half of an initialisation scheme. It is **excluded from the fan-in** and **initialised
    to zero**.

    Parameters
    ----------
    init_fn :
        One of the strategies in :mod:`pyhgf.utils.weight_initialisation`.
    n_children, n_parents :
        Shape of the matrix, ``n_parents`` including the bias column when present.
    add_constant_input :
        Whether the last column is the bias.
    seed :
        Seed forwarded to *init_fn*.
    kwargs :
        Extra keyword arguments forwarded to *init_fn*.

    Returns
    -------
    jnp.ndarray
        The initialised matrix, shape ``(n_children, n_parents)``.
    """
    n_real = n_parents - 1 if add_constant_input else n_parents
    flat = init_fn(n_real, n_children, seed=seed, **kwargs)
    weights = np.asarray(flat).reshape(n_children, n_real)
    if add_constant_input:
        weights = np.concatenate([weights, np.zeros((n_children, 1))], axis=1)
    return jnp.asarray(weights)
