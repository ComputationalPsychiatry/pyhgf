# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Exactness witnesses for ``kind="natural"`` (NG-VI weight update).

These pin the properties derived in the natural-gradient notebook
(``docs/source/notebooks/0.5-Natural_gradients.md``, §6/§9): the Sherman-Morrison
update is the *exact* natural gradient of the value-coupled linear-Gaussian
layer, reducing to the diagonal-Fisher and prior-metric forms in the
appropriate limits.
"""

import jax
import jax.numpy as jnp
import pytest

from pyhgf.typing.vectorised import LayerState
from pyhgf.updates.vectorized.learning import (
    _softmax_fisher_inverse,
    vectorized_weight_gradient,
    vectorized_weight_gradient_factors,
)


@pytest.fixture(autouse=True)
def _x64():
    """Enable float64 for these exactness checks, then restore the session default.

    The tolerances below are at the float64 floor; leaving x64 on would leak
    into the shared pytest session and shift dtypes in unrelated tests.
    """
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _state(mean, precision, expected_mean, expected_precision):
    """Minimal value-level LayerState (volatility fields unused here)."""
    return LayerState(
        mean=mean,
        precision=precision,
        expected_mean=expected_mean,
        expected_precision=expected_precision,
        conditional_expected_precision=expected_precision,
        effective_precision=expected_precision,
        value_prediction_error=jnp.zeros_like(mean),
        mean_vol=None,
        precision_vol=None,
        expected_mean_vol=None,
        expected_precision_vol=None,
        effective_precision_vol=None,
        volatility_prediction_error=None,
    )


@pytest.fixture
def layers():
    """Build a random parent/child pair with positive precisions and GELU coupling."""
    d_p, d_c = 5, 4
    k = jax.random.split(jax.random.PRNGKey(0), 5)
    a = jax.random.normal(k[0], (d_p,))
    pi_p = jnp.exp(jax.random.normal(k[1], (d_p,)))
    x = jax.random.normal(k[2], (d_c,))
    xhat = jax.random.normal(k[3], (d_c,))
    pi_c = jnp.exp(jax.random.normal(k[4], (d_c,)))
    parent = _state(a, pi_p, a, pi_p)
    child = _state(x, pi_c, xhat, pi_c)
    return parent, child, jax.nn.gelu


def test_natural_matches_boxed_formula(layers):
    """Full gradient equals the boxed Sherman-Morrison expression (§6)."""
    parent, child, g = layers
    a = g(parent.mean)
    pe = child.mean - child.expected_mean
    s = jnp.sum(a**2 / parent.expected_precision)
    shrink = child.expected_precision / (1.0 + child.expected_precision * s)
    ref = (pe * shrink)[:, None] * (a / parent.expected_precision)[None, :]  # ascent
    got = vectorized_weight_gradient(parent, child, g, kind="natural")  # descent
    assert jnp.allclose(got, -ref, atol=1e-12)


def test_natural_factored_matches_full(layers):
    """Separable (u, v) factors reproduce the full gradient."""
    parent, child, g = layers
    full = vectorized_weight_gradient(parent, child, g, kind="natural")
    u, v = vectorized_weight_gradient_factors(parent, child, g, kind="natural")
    assert jnp.allclose(u[:, None] * v[None, :], full, atol=1e-12)


def test_natural_one_step_reaches_posterior_mean(layers):
    """One natural step from W=0 lands on the conjugate posterior mean (§9)."""
    parent, child, g = layers
    a = g(parent.mean)
    P = jnp.diag(parent.expected_precision)
    # Prediction from W=0 is 0, so expected_mean must be 0 for delta = x.
    child0 = _state(child.mean, child.precision, jnp.zeros_like(child.mean),
                    child.expected_precision)
    step = -vectorized_weight_gradient(parent, child0, g, kind="natural")  # ascent
    for i in range(child.mean.shape[0]):
        fisher = child.expected_precision[i] * jnp.outer(a, a)
        m_post = jnp.linalg.solve(
            P + fisher, child.expected_precision[i] * child.mean[i] * a
        )
        assert jnp.allclose(step[i], m_post, atol=1e-10)


def test_natural_reduces_to_diagonal_for_onehot_parent(layers):
    """One active parent dim: the exact update equals the diagonal-Fisher form.

    With no off-diagonal Fisher mass, natural coincides with
    pi_c*PE*a_j / (pi_p + pi_c*a_j^2).
    """
    _, child, _ = layers
    onehot = jnp.zeros(5).at[2].set(1.3)
    pi_p = jnp.exp(jnp.arange(5.0))
    parent = _state(onehot, pi_p, onehot, pi_p)
    nat = vectorized_weight_gradient(parent, child, lambda z: z, kind="natural")
    pi_c = child.expected_precision
    pe = child.mean - child.expected_mean
    diagonal = -(  # descent sign
        (pi_c * pe)[:, None]
        * onehot[None, :]
        / (pi_p[None, :] + pi_c[:, None] * onehot[None, :] ** 2)
    )
    assert jnp.allclose(nat, diagonal, atol=1e-12)


def test_natural_reduces_to_prior_metric_for_weak_data(layers):
    """Uninformative data (pi_c -> 0): shrinkage -> 1, leaving the prior metric.

    The update reduces to pi_c*PE*a_j / pi_p (no data-curvature correction).
    """
    parent, child, g = layers
    weak = _state(child.mean, child.precision * 1e-8, child.expected_mean,
                  child.expected_precision * 1e-8)
    nat = vectorized_weight_gradient(parent, weak, g, kind="natural")
    a = g(parent.mean)
    pe = weak.mean - weak.expected_mean
    prior_metric = -(  # descent sign
        (weak.expected_precision * pe)[:, None] * (a / parent.expected_precision)[None, :]
    )
    assert jnp.allclose(nat, prior_metric, atol=1e-8)


def test_natural_vanishes_for_rigid_weights(layers):
    """Infinite prior precision pins the weights: natural update -> 0 (§9)."""
    parent, child, g = layers
    rigid = _state(parent.mean, parent.precision * 1e12, parent.expected_mean,
                   parent.expected_precision * 1e12)
    nat = vectorized_weight_gradient(rigid, child, g, kind="natural")
    assert jnp.max(jnp.abs(nat)) < 1e-8


def test_natural_constant_node_shape_and_factoring(layers):
    """Constant (bias) node participates with prior precision 1."""
    parent, child, g = layers
    full = vectorized_weight_gradient(
        parent, child, g, kind="natural", parent_has_constant=True
    )
    u, v = vectorized_weight_gradient_factors(
        parent, child, g, kind="natural", parent_has_constant=True
    )
    assert full.shape == (child.mean.shape[0], parent.mean.shape[0] + 1)
    assert jnp.allclose(u[:, None] * v[None, :], full, atol=1e-12)


def test_natural_is_separable():
    """``natural`` is registered as a separable kind (efficient batch path)."""
    from pyhgf.updates.vectorized.learning import SEPARABLE_KINDS

    assert "natural" in SEPARABLE_KINDS


# ---------------------------------------------------------------------------
# Categorical (softmax) natural gradient
# ---------------------------------------------------------------------------


@pytest.fixture
def categorical():
    """Build a softmax child (probabilities in expected_mean, one-hot in mean)."""
    K, d = 6, 5
    k = jax.random.split(jax.random.PRNGKey(3), 4)
    a = jax.random.normal(k[0], (d,))
    pi_p = jnp.exp(jax.random.normal(k[1], (d,)))
    p = jax.nn.softmax(jax.random.normal(k[2], (K,)))
    y = jax.nn.one_hot(2, K)
    parent = _state(a, pi_p, a, pi_p)
    child = _state(y, jnp.ones(K), p, jnp.ones(K))
    return parent, child, jax.nn.gelu


def test_softmax_fisher_inverse_solves_damped_system(categorical):
    """The helper returns x with (diag(p) + lam - p pᵀ) x = pe."""
    _, child, _ = categorical
    p, pe = child.expected_mean, child.mean - child.expected_mean
    lam = 1e-2
    x = _softmax_fisher_inverse(p, pe, lam)
    fisher = jnp.diag(p) + lam * jnp.eye(p.shape[0]) - jnp.outer(p, p)
    assert jnp.allclose(fisher @ x, pe, atol=1e-12)


def test_categorical_natural_matches_kfac_closed_form(categorical):
    """Full gradient equals (Fisher-preconditioned error) ⊗ P^{-1}a / (1+s)."""
    parent, child, g = categorical
    p, pe = child.expected_mean, child.mean - child.expected_mean
    a = g(parent.mean)
    s = jnp.sum(a**2 / parent.expected_precision)
    x = _softmax_fisher_inverse(p, pe, 1e-2)
    ref = (x / (1.0 + s))[:, None] * (a / parent.expected_precision)[None, :]  # ascent
    got = vectorized_weight_gradient(
        parent, child, g, kind="natural", child_is_categorical=True
    )
    assert jnp.allclose(got, -ref, atol=1e-12)


def test_categorical_natural_factored_matches_full(categorical):
    """The categorical natural gradient stays separable."""
    parent, child, g = categorical
    full = vectorized_weight_gradient(
        parent, child, g, kind="natural", child_is_categorical=True
    )
    u, v = vectorized_weight_gradient_factors(
        parent, child, g, kind="natural", child_is_categorical=True
    )
    assert jnp.allclose(u[:, None] * v[None, :], full, atol=1e-12)


def test_categorical_natural_stable_for_near_impossible_class(categorical):
    """A vanishing predicted probability does not blow the update up."""
    parent, _, g = categorical
    K = 6
    p = jnp.array([0.98, 0.019, 1e-8, 4e-4, 4e-4, 2e-4])
    p = p / p.sum()
    child = _state(jax.nn.one_hot(2, K), jnp.ones(K), p, jnp.ones(K))
    got = vectorized_weight_gradient(
        parent, child, g, kind="natural", child_is_categorical=True
    )
    assert jnp.all(jnp.isfinite(got))
    # Damping caps the 1/p amplification at ~1/lambda; nowhere near 1/1e-8.
    assert float(jnp.max(jnp.abs(got))) < 1e3


def test_softmax_fisher_inverse_recentres_as_damping_vanishes(categorical):
    """With lam -> 0 the solution is the zero-sum pseudoinverse of the residual."""
    _, child, _ = categorical
    p, pe = child.expected_mean, child.mean - child.expected_mean
    x = _softmax_fisher_inverse(p, pe, 1e-10)
    fisher0 = jnp.diag(p) - jnp.outer(p, p)  # singular multinomial Fisher
    assert jnp.allclose(fisher0 @ x, pe, atol=1e-6)  # solves on the range
    assert abs(float(jnp.sum(x))) < 1e-5  # minimum-norm (zero-sum) solution


def test_categorical_natural_differs_from_gaussian_treatment(categorical):
    """The softmax metric is not the unit-precision Gaussian metric."""
    parent, child, g = categorical
    cat = vectorized_weight_gradient(
        parent, child, g, kind="natural", child_is_categorical=True
    )
    gauss = vectorized_weight_gradient(
        parent, child, g, kind="natural", child_is_categorical=False
    )
    assert not jnp.allclose(cat, gauss, atol=1e-3)


# ---------------------------------------------------------------------------
# precision_weighted uses the posterior precision (backprop-parity choice)
# ---------------------------------------------------------------------------


def test_precision_weighted_uses_posterior_precision():
    """`precision_weighted` weights by the posterior precision, not the predicted one.

    A moved interior belief shifts by (routed error) / posterior precision, so
    weighting by that same posterior precision cancels the division and recovers
    backprop at any precision setting (see test_input_side_gradient_precision_
    cancellation). `natural` — a conjugate update — uses the predicted precision.
    """
    d_p, d_c = 5, 4
    k = jax.random.split(jax.random.PRNGKey(7), 6)
    a = jax.random.normal(k[0], (d_p,))
    pi_p = jnp.exp(jax.random.normal(k[1], (d_p,)))
    x = jax.random.normal(k[2], (d_c,))
    xhat = jax.random.normal(k[3], (d_c,))
    pi_post = jnp.exp(jax.random.normal(k[4], (d_c,)))  # posterior
    pi_pred = jnp.exp(jax.random.normal(k[5], (d_c,)))  # predicted (expected)
    parent = _state(a, pi_p, a, pi_p)
    child = _state(x, pi_post, xhat, pi_pred)  # distinct posterior vs predicted
    got = vectorized_weight_gradient(parent, child, lambda z: z, kind="precision_weighted")
    ref = -((x - xhat) * pi_post)[:, None] * a[None, :]  # descent, posterior precision
    assert jnp.allclose(got, ref, atol=1e-12)
    wrong = -((x - xhat) * pi_pred)[:, None] * a[None, :]
    assert not jnp.allclose(got, wrong, atol=1e-3)


# ---------------------------------------------------------------------------
# Binary (Bernoulli) natural gradient — curvature rho = sigma(1 - sigma)
# ---------------------------------------------------------------------------


@pytest.fixture
def binary():
    """Build a binary (sigmoid) child: expected_mean = sigma, expected_precision = sigma(1-sigma)."""
    d_p, d_c = 5, 4
    k = jax.random.split(jax.random.PRNGKey(11), 4)
    a = jax.random.normal(k[0], (d_p,))
    pi_p = jnp.exp(jax.random.normal(k[1], (d_p,)))
    sigma = jax.nn.sigmoid(jax.random.normal(k[2], (d_c,)))
    x = (jax.random.uniform(k[3], (d_c,)) < 0.5).astype(jnp.float64)
    parent = _state(a, pi_p, a, pi_p)
    child = _state(x, jnp.ones(d_c), sigma, sigma * (1.0 - sigma))
    return parent, child, jax.nn.gelu


def test_binary_natural_matches_bernoulli_fisher(binary):
    """Binary natural equals (P + rho a aT)^-1 (x - sigma) a, rho = sigma(1-sigma)."""
    parent, child, g = binary
    a = g(parent.mean)
    P = jnp.diag(parent.expected_precision)
    sigma = child.expected_mean
    delta = child.mean - child.expected_mean
    got = vectorized_weight_gradient(parent, child, g, kind="natural", child_is_binary=True)
    for i in range(child.mean.shape[0]):
        rho = sigma[i] * (1.0 - sigma[i])
        ref_i = jnp.linalg.solve(P + rho * jnp.outer(a, a), delta[i] * a)  # ascent
        assert jnp.allclose(-got[i], ref_i, atol=1e-10)


def test_binary_natural_factored_matches_full(binary):
    """The binary natural gradient stays separable."""
    parent, child, g = binary
    full = vectorized_weight_gradient(parent, child, g, kind="natural", child_is_binary=True)
    u, v = vectorized_weight_gradient_factors(
        parent, child, g, kind="natural", child_is_binary=True
    )
    assert jnp.allclose(u[:, None] * v[None, :], full, atol=1e-12)


def test_binary_natural_differs_from_unit_curvature(binary):
    """Curvature rho = sigma(1-sigma) is not the old rho = 1 convention."""
    parent, child, g = binary
    a = g(parent.mean)
    s = jnp.sum(a**2 / parent.expected_precision)
    delta = child.mean - child.expected_mean
    got = vectorized_weight_gradient(parent, child, g, kind="natural", child_is_binary=True)
    old = -(delta / (1.0 + s))[:, None] * (a / parent.expected_precision)[None, :]
    assert not jnp.allclose(got, old, atol=1e-3)
