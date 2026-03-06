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
"""

from __future__ import annotations

from typing import Optional

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

    Generates a random matrix, computes its SVD, and returns the
    (semi-)orthogonal factor scaled by *gain*.  This preserves gradient
    norms during backpropagation.

    Parameters
    ----------
    n_parents :
        Number of parent (input) nodes — fan-in.
    n_children :
        Number of child (output) nodes — fan-out.
    gain :
        Multiplicative scaling factor (default 1.0).
    seed :
        Optional random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Weight vector of length ``n_parents * n_children``.
    """
    rng = np.random.default_rng(seed)
    # Always create a tall-or-square matrix so full_matrices=False gives
    # the right number of orthogonal columns.
    if n_parents >= n_children:
        a = rng.standard_normal((n_parents, n_children))
        u, _, _ = np.linalg.svd(a, full_matrices=False)
        q = u  # (n_parents, n_children)
    else:
        a = rng.standard_normal((n_children, n_parents))
        u, _, _ = np.linalg.svd(a, full_matrices=False)
        q = u.T  # (n_parents, n_children)
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
