# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pyhgf.model import DeepNetwork
from pyhgf.model.fused import FusedPipeline
from pyhgf.model.hybrid import DeepNetworkAdapter
from pyhgf.typing.vectorised import LayerStack, LayerState
from pyhgf.updates.vectorised.learning import (
    create_input_gram,
    create_step_gram,
    learning_weights_vectorised,
    resolve_synaptic_uncertainty_settings,
    vectorised_synaptic_uncertainty_update,
)


def _importance(parent_state, child_state, coupling_fn, **kwargs):
    """Return the importance factors ``(p, h**2)`` through the module's entry point.

    The entry point hands back the shared parent-side activation; the importance's own
    parent factor is its square.
    """
    _, h, p = learning_weights_vectorised(
        parent_state, child_state, coupling_fn, **kwargs
    )
    return p, h**2


def _states(parent_mean, child_precision):
    """Parent and child LayerStates carrying the given evidence precision.

    The child's evidence precision is its posterior minus its marginal predicted
    precision, so the posterior is set to the requested value above a unit prediction.
    Zero effective precision means no process noise, so the evidence reaches the weight
    unsoftened and equals ``child_precision``.
    """
    parent = dataclasses.replace(
        LayerState.create(len(parent_mean), has_volatility_parent=False),
        mean=jnp.asarray(parent_mean),
    )
    child = dataclasses.replace(
        LayerState.create(len(child_precision), has_volatility_parent=False),
        expected_precision=jnp.ones(len(child_precision)),
        precision=jnp.asarray(child_precision) + 1.0,
        effective_precision=jnp.zeros(len(child_precision)),
    )
    return parent, child


def test_effective_evidence_precision():
    """The evidence precision is recovered from the sweeps' cached fields.

    Read through the entry point's importance factor, which is where the quantity now
    lives.
    """
    n = 3
    base = LayerState.create(n, has_volatility_parent=False)
    parent = dataclasses.replace(
        LayerState.create(2, has_volatility_parent=False), mean=jnp.ones(2)
    )

    def xi_tilde(child):
        _, _, p = learning_weights_vectorised(parent, child, lambda m: m)
        return p

    # Evidence is posterior minus marginal predicted; with no process noise it
    # reaches the weight unsoftened.
    state = dataclasses.replace(
        base,
        expected_precision=jnp.array([1.0, 2.0, 0.5]),
        precision=jnp.array([5.0, 6.0, 2.5]),
        effective_precision=jnp.zeros(n),
    )
    np.testing.assert_allclose(xi_tilde(state), [4.0, 4.0, 2.0], rtol=1e-6)

    # Process noise adds to the evidence *variance*: with omega = gamma / pi
    # tilde, xi_tilde = xi / (1 + omega xi). Here omega = 0.25 throughout.
    state = dataclasses.replace(
        base,
        expected_precision=jnp.array([1.0, 2.0, 0.5]),
        precision=jnp.array([5.0, 6.0, 2.5]),
        effective_precision=0.25 * jnp.array([1.0, 2.0, 0.5]),
    )
    xi = np.array([4.0, 4.0, 2.0])
    np.testing.assert_allclose(xi_tilde(state), xi / (1.0 + 0.25 * xi), rtol=1e-6)

    # A clamped observed layer has its marginal predicted precision forced to
    # its posterior, so no evidence arrives and its own precision stands in.
    clamped = dataclasses.replace(
        base,
        expected_precision=jnp.array([3.0, 3.0, 3.0]),
        precision=jnp.array([3.0, 3.0, 3.0]),
        effective_precision=jnp.zeros(n),
    )
    np.testing.assert_allclose(xi_tilde(clamped), [3.0] * 3)

    # Clipping can drive the posterior below the prediction; the increment must
    # stay non-negative rather than turning into a precision subtraction.
    clipped = dataclasses.replace(
        base,
        expected_precision=jnp.array([4.0, 4.0, 4.0]),
        precision=jnp.array([1.0, 1.0, 1.0]),
        effective_precision=jnp.zeros(n),
    )
    assert np.all(np.asarray(xi_tilde(clipped)) >= 0.0)

    # The same floor also guards the softening denominator, 1 / (1 + omega * xi).
    # Left unfloored, a negative evidence would inflate the gradient instead of
    # attenuating it, and a large enough one would flip its sign.
    clipped_noisy = dataclasses.replace(
        clipped,
        mean=jnp.array([1.0, -2.0, 0.5]),
        expected_mean=jnp.zeros(n),
        effective_precision=0.25 * jnp.array([4.0, 4.0, 4.0]),
    )
    u_ev, _, _ = learning_weights_vectorised(
        parent, clipped_noisy, lambda m: m, kind="synaptic_uncertainty"
    )
    u_pw, _, _ = learning_weights_vectorised(
        parent, clipped_noisy, lambda m: m, kind="precision_weighted"
    )
    np.testing.assert_allclose(u_ev, u_pw, rtol=1e-6)

    # A supplied evidence replaces the cached difference and is softened by the
    # same process noise. Asserted here because the networks the walk is
    # exercised on end to end carry no volatility parent, so their omega is
    # identically zero and the softening is inert in them.
    noisy = dataclasses.replace(
        base,
        expected_precision=jnp.array([1.0, 2.0, 0.5]),
        precision=jnp.array([5.0, 6.0, 2.5]),
        effective_precision=0.25 * jnp.array([1.0, 2.0, 0.5]),
    )
    carried = jnp.array([2.0, 8.0, 1.0])
    _, _, p = learning_weights_vectorised(
        parent, noisy, lambda m: m, child_evidence=carried
    )
    carried_np = np.asarray(carried)
    np.testing.assert_allclose(p, carried_np / (1.0 + 0.25 * carried_np), rtol=1e-6)


def _step(weights, precision_delta, gradient, importance, **kwargs):
    """One belief step, returning ``(update, new_delta)``.

    The rule returns the new mean; the tests below are written against the *change* in
    the mean, which is what the update rule states.
    """
    settings = resolve_synaptic_uncertainty_settings(kwargs)
    new_weights, new_delta = vectorised_synaptic_uncertainty_update(
        weights, precision_delta, gradient, importance, settings
    )
    return new_weights - weights, new_delta


def test_importance_factors_kernel():
    """The importance kernel returns (evidence precision, h^2)."""
    parent, child = _states([0.5, -2.0, 3.0], [4.0, 0.25])
    identity = lambda m: m  # noqa: E731

    p, q = _importance(parent, child, identity)
    np.testing.assert_allclose(p, [4.0, 0.25])
    np.testing.assert_allclose(q, [0.25, 4.0, 9.0])

    # Constant input appends a unit activation, so the bias weight counts as usage 1.
    p, q = _importance(parent, child, identity, parent_has_constant=True)
    np.testing.assert_allclose(q, [0.25, 4.0, 9.0, 1.0])

    # A clamped discrete child contributes the curvature of its own likelihood,
    # p(1 - p), read off its expected mean rather than any precision field.
    probs = jnp.array([0.25, 0.9])
    discrete = dataclasses.replace(child, expected_mean=probs)
    for child_kind in ("binary", "categorical"):
        p, _ = _importance(parent, discrete, identity, child_kind=child_kind)
        np.testing.assert_allclose(p, probs * (1.0 - probs), rtol=1e-6)

    # The "standard" kind keeps the unit-precision convention throughout.
    p, _ = _importance(parent, child, identity, kind="standard")
    np.testing.assert_allclose(p, [1.0, 1.0])
    p, _ = _importance(parent, discrete, identity, kind="standard", child_kind="binary")
    np.testing.assert_allclose(p, [1.0, 1.0])

    # A nonlinear coupling squares the coupled activation, not the raw mean.
    p, q = _importance(parent, child, jnp.tanh)
    np.testing.assert_allclose(q, np.tanh([0.5, -2.0, 3.0]) ** 2, rtol=1e-6)

    # Non-finite entries are zeroed rather than propagated.
    bad_parent = dataclasses.replace(parent, mean=jnp.array([jnp.inf, 1.0, 2.0]))
    _, q = _importance(bad_parent, child, identity)
    np.testing.assert_allclose(q, [0.0, 1.0, 4.0])

    with pytest.raises(ValueError):
        _importance(parent, child, identity, kind="nope")


def test_settings_validation():
    """Invalid configurations are rejected when the settings are resolved."""
    with pytest.raises(ValueError, match="window"):
        resolve_synaptic_uncertainty_settings({})
    with pytest.raises(ValueError, match="at least 1"):
        resolve_synaptic_uncertainty_settings({"window": 0.5})
    with pytest.raises(ValueError, match="learning_rate"):
        resolve_synaptic_uncertainty_settings({"window": 100, "learning_rate": 0.0})
    with pytest.raises(ValueError, match="increment_scale"):
        resolve_synaptic_uncertainty_settings({"window": 100, "increment_scale": 0.0})
    with pytest.raises(ValueError, match="Unknown learning_kwargs"):
        resolve_synaptic_uncertainty_settings({"window": 100, "nonsense": 1})

    # The gradient the rule descends is fixed, so it is not a setting: even the
    # value it is fixed to is rejected as an unknown key.
    with pytest.raises(ValueError, match="Unknown learning_kwargs"):
        resolve_synaptic_uncertainty_settings({
            "window": 100,
            "kind": "synaptic_uncertainty",
        })

    # The leak and update form are no longer settings, so naming either is an
    # unknown key rather than a choice.
    for gone in ("leak", "update_form", "autoconnection", "tonic_variance"):
        with pytest.raises(ValueError, match="Unknown learning_kwargs"):
            resolve_synaptic_uncertainty_settings({
                "window": 100,
                gone: "synaptic_uncertainty",
            })


def test_sgd_reduction_at_init():
    """With no accumulated importance the rule is plain SGD at prior_variance.

    The only departure is the mean reversion toward prior_mean, at rate 1/window when
    the precision sits at the prior.
    """
    window, prior_variance = 1000.0, 0.3
    w = jnp.array([[0.5, -1.0, 2.0], [0.0, 0.1, -0.2]])
    g = jnp.array([[1.0, -2.0, 0.5], [0.3, 0.0, -1.5]])
    precision_delta = jnp.zeros_like(w)

    # Compared on the mean the rule returns rather than on a difference: where
    # the step is small next to the weight, recovering it by subtraction costs
    # more float32 accuracy than the step itself carries.
    settings = resolve_synaptic_uncertainty_settings({
        "window": window,
        "prior_variance": prior_variance,
    })
    new_weights, new_delta = vectorised_synaptic_uncertainty_update(
        w, precision_delta, g, (jnp.zeros(2), jnp.zeros(3)), settings
    )
    np.testing.assert_allclose(
        new_weights, w + (-prior_variance * g - w / window), rtol=1e-5
    )
    # Zero increment leaves the precision at the prior, i.e. no accumulation at
    # all. Asserted on the delta itself: adding the prior back first would apply
    # the tolerance to a base of 1 / prior_variance and accept a small non-zero
    # accumulation.
    np.testing.assert_allclose(new_delta, 0.0, atol=1e-12)


def test_learning_rate_scales_steps_but_not_forgetting():
    """The multiplier scales the gradient step and leaves the leak alone.

    Doubling it doubles the gradient part of the update, leaves the reversion toward the
    prior untouched, and does not touch the precision at all, so the depth of protection
    and the forgetting window are unchanged.
    """
    window, prior_variance = 500.0, 0.1
    w = jnp.array([[0.4, -0.7]])
    g = jnp.array([[1.0, -0.5]])
    precision_delta = jnp.zeros_like(w)
    importance = (jnp.array([1.5]), jnp.array([0.5, 2.0]))

    updates, deltas = {}, {}
    for alpha in (1.0, 2.0):
        updates[alpha], deltas[alpha] = _step(
            w,
            precision_delta,
            g,
            importance,
            window=window,
            prior_variance=prior_variance,
            learning_rate=alpha,
        )

    reversion = -w / window  # prior mean 0, precision at its prior value
    gradient_part = {a: updates[a] - reversion for a in (1.0, 2.0)}
    np.testing.assert_allclose(gradient_part[2.0], 2.0 * gradient_part[1.0], rtol=1e-6)
    np.testing.assert_allclose(deltas[2.0], deltas[1.0], rtol=1e-9)


def test_increment_scale_deepens_protection_only():
    """The scale multiplies the precision gain and leaves the first step alone.

    Under the rule's leak the mean update divides by the pre-update precision, so at the
    prior the first step is identical at any scale; the precision gain is multiplied
    exactly.
    """
    w = jnp.zeros((1, 2))
    g = jnp.array([[1.0, -0.5]])
    precision_delta = jnp.zeros_like(w)
    importance = (jnp.array([2.0]), jnp.array([0.5, 1.5]))

    results = {}
    for scale in (1.0, 30.0):
        results[scale] = _step(
            w,
            precision_delta,
            g,
            importance,
            window=1000.0,
            prior_variance=0.5,
            increment_scale=scale,
        )

    np.testing.assert_allclose(results[30.0][0], results[1.0][0], rtol=1e-6)
    np.testing.assert_allclose(results[30.0][1], 30.0 * results[1.0][1], rtol=1e-6)


def test_synaptic_uncertainty_fixed_point_is_linear_in_evidence():
    """Under the rule's leak the precision converges to prior + window * increment."""
    window, prior_variance = 50.0, 1.0
    settings = resolve_synaptic_uncertainty_settings({
        "window": window,
        "prior_variance": prior_variance,
    })
    w = jnp.zeros((1, 2))
    precision_delta = jnp.zeros_like(w)
    p, q = jnp.array([2.0]), jnp.array([0.5, 1.5])
    g = jnp.zeros((1, 2))
    for _ in range(2000):
        w, precision_delta = vectorised_synaptic_uncertainty_update(
            w, precision_delta, g, (p, q), settings
        )

    expected = 1.0 / prior_variance + window * p[0] * q
    np.testing.assert_allclose(
        precision_delta[0] + 1.0 / prior_variance, expected, rtol=1e-3
    )


def test_layer_stack_shapes():
    """Stacked elements (leading slice axis) broadcast through the update."""
    w = jnp.zeros((4, 2, 3))  # (n_slices, n_children, n_parents)
    precision_delta = jnp.zeros_like(w)
    g = jnp.ones((4, 2, 3))
    importance = (jnp.ones((4, 2)), jnp.ones((4, 3)))

    update, new_delta = _step(
        w, precision_delta, g, importance, window=100.0, prior_variance=1.0
    )
    assert update.shape == (4, 2, 3)
    assert new_delta.shape == (4, 2, 3)
    assert bool(jnp.all(new_delta + 1.0 > 1.0))


def test_synaptic_uncertainty_matches_the_increment_it_divides():
    """The rule's gradient and its increment are one energy's two derivatives.

    The weight-belief step is (gradient / accumulated precision), so the two have to be
    built from the same evidence or the ratio is not a Newton step. The gradient's
    child-side factor is the posterior precision softened by the process noise; the
    increment's is the evidence precision softened by the same noise. Their ratio must
    therefore be the posterior precision over the evidence precision, with no leftover
    noise term.
    """
    n = 3
    base = LayerState.create(n, has_volatility_parent=False)
    state = dataclasses.replace(
        base,
        mean=jnp.array([1.0, -2.0, 0.5]),
        expected_mean=jnp.zeros(n),
        expected_precision=jnp.array([1.0, 2.0, 0.5]),
        precision=jnp.array([5.0, 6.0, 2.5]),
        effective_precision=0.25 * jnp.array([1.0, 2.0, 0.5]),
    )
    parent = dataclasses.replace(
        LayerState.create(2, has_volatility_parent=False), mean=jnp.array([1.0, -1.0])
    )
    identity = lambda m: m  # noqa: E731

    u_ev, _, _ = learning_weights_vectorised(
        parent, state, identity, kind="synaptic_uncertainty"
    )
    u_pw, _, _ = learning_weights_vectorised(
        parent, state, identity, kind="precision_weighted"
    )
    _, _, p_inc = learning_weights_vectorised(
        parent, state, identity, kind="synaptic_uncertainty"
    )

    evidence = np.asarray(state.precision - state.expected_precision)
    softening = 1.0 / (1.0 + 0.25 * evidence)
    # The gradient is the precision-weighted one charged the process noise.
    np.testing.assert_allclose(u_ev, np.asarray(u_pw) * softening, rtol=1e-6)
    # Ratio of the two child-side factors: posterior over evidence precision,
    # the softening having cancelled.
    np.testing.assert_allclose(
        np.asarray(u_ev) / np.asarray(p_inc),
        -np.asarray(state.mean) * np.asarray(state.precision) / evidence,
        rtol=1e-6,
    )

    # Where there is no process noise the two precision kinds coincide.
    quiet = dataclasses.replace(state, effective_precision=jnp.zeros(n))
    np.testing.assert_allclose(
        learning_weights_vectorised(
            parent, quiet, identity, kind="synaptic_uncertainty"
        )[0],
        learning_weights_vectorised(parent, quiet, identity, kind="precision_weighted")[
            0
        ],
        rtol=1e-9,
    )


def test_gradient_kind_is_pinned():
    """The rule descends one gradient, whatever the network was built around.

    The mean update divides the child's evidence by the weight's own precision,
    so the numerator has to carry that evidence: the precision-weighted form is
    the only one that does. A network whose plain learning kind is "standard"
    still gets the pinned gradient once the belief rule is selected.
    """
    dn = DeepNetwork().add_layer(size=2).add_layer(size=3)

    gradient_kind, settings = dn._resolve_learning(
        "synaptic_uncertainty", {"window": 100.0}
    )
    assert gradient_kind == "synaptic_uncertainty"
    assert not hasattr(settings, "kind")

    # A plain kind still passes through untouched, carrying no settings.
    assert dn._resolve_learning("standard", None) == ("standard", None)


def test_deepnetwork_integration():
    """Both learning paths carry the belief on the layers, not beside them.

    The sequential (fit) and batch-synchronous (batch_update) paths accumulate precision
    above the prior on the elements that hold weights, need no optimiser, and leave
    plain optax members working unchanged.
    """
    n_targets, n_h, n_input = 2, 3, 1
    np.random.seed(7)
    x = np.random.randn(8, n_input)
    y = np.random.randn(8, n_targets)
    kwargs = {"window": 100.0, "prior_variance": 0.1}

    def build():
        return (
            DeepNetwork()
            .add_layer(size=n_targets)
            .add_layer(size=n_h)
            .add_layer(size=n_input)
            .weight_initialisation("xavier", key=jax.random.key(0))
        )

    # Sequential path: the belief is installed on first use and no optimiser
    # state is created, since the rule carries its own step size.
    dn = build()
    dn.fit(x=x, y=y, learning_kind="synaptic_uncertainty", learning_kwargs=kwargs)
    assert dn.opt_state is None
    accumulated = [
        e
        for e in (
            getattr(elem, "weights_precision_delta", None) for elem in dn.state.layers
        )
        if e is not None
    ]
    assert len(accumulated) == 2
    assert all(bool(jnp.all(jnp.isfinite(e))) for e in accumulated)
    assert any(bool(jnp.any(e > 1e-6)) for e in accumulated)
    # One precision per weight: the belief sits on exactly the elements holding a
    # weight matrix, matches its shape, and only ever accumulates above the prior,
    # which the field stores as the delta over it.
    for elem in dn.state.layers:
        precision_delta = getattr(elem, "weights_precision_delta", None)
        assert (precision_delta is None) == (elem.weights_mean is None)
        if precision_delta is not None:
            assert precision_delta.shape == elem.weights_mean.shape
            assert bool(jnp.all(precision_delta >= 0.0))
    assert bool(jnp.all(jnp.isfinite(dn.predict(np.array([[0.5]])))))

    # Batched path.
    dn_batch = build()
    dn_batch.batch_update(
        x, y, learning_kind="synaptic_uncertainty", learning_kwargs=kwargs
    )
    accumulated = [
        e
        for e in (
            getattr(elem, "weights_precision_delta", None)
            for elem in dn_batch.state.layers
        )
        if e is not None
    ]
    assert any(bool(jnp.any(e > 1e-6)) for e in accumulated)

    # Plain optax path is untouched, and carries no belief.
    dn_sgd = build()
    dn_sgd.fit(x=x, y=y, optimiser=optax.sgd(0.1))
    assert all(
        e is None
        for e in (
            getattr(elem, "weights_precision_delta", None)
            for elem in dn_sgd.state.layers
        )
    )
    assert bool(jnp.all(jnp.isfinite(dn_sgd.predict(np.array([[0.5]])))))

    # A gradient kind without an optimiser is refused rather than silently idle.
    with pytest.raises(ValueError, match="needs an optimiser"):
        build().fit(x=x, y=y)
    # learning_kwargs is only meaningful for the belief rule.
    with pytest.raises(ValueError, match="only used by"):
        build().fit(x=x, y=y, optimiser=optax.sgd(0.1), learning_kwargs=kwargs)


def test_install_weight_belief_is_explicit_and_idempotent():
    """The belief can be installed ahead of the first fit, and only once."""
    net = (
        DeepNetwork()
        .add_layer(size=2)
        .add_layer(size=3)
        .weight_initialisation("xavier", key=jax.random.key(0))
    )
    assert all(
        e is None
        for e in (
            getattr(elem, "weights_precision_delta", None) for elem in net.state.layers
        )
    )

    net.install_weight_belief()
    precision_delta = tuple(
        getattr(elem, "weights_precision_delta", None) for elem in net.state.layers
    )
    assert precision_delta[0] is None  # the bottom layer holds no weights
    np.testing.assert_allclose(precision_delta[1], 0.0)
    assert precision_delta[1].shape == net.state.layers[1].weights_mean.shape

    # Installing again leaves an accumulated belief alone.
    net.state = dataclasses.replace(
        net.state,
        layers=(
            net.state.layers[0],
            dataclasses.replace(
                net.state.layers[1],
                weights_precision_delta=jnp.full_like(precision_delta[1], 2.5),
            ),
        ),
    )
    net.install_weight_belief()
    np.testing.assert_allclose(
        tuple(
            getattr(elem, "weights_precision_delta", None) for elem in net.state.layers
        )[1],
        2.5,
    )

    with pytest.raises(ValueError, match="hold no weight matrix"):
        net.install_weight_belief(layers=[0])


def test_evidence_walk_curvature():
    """The walk's increment is the Gauss-Newton diagonal; the sweep's is inflated.

    The importance increment is meant to be the curvature the data impose on a
    weight. Recovering the evidence from the filter's cache as
    ``precision - expected_precision`` does not deliver that above the observed
    layer, because the chain it is read from is seeded with the clamped layer's
    unit precision rather than with the categorical curvature ``p(1 - p)``: the
    interior increment comes out several times too large. Carrying the evidence
    instead reproduces the autodiff Gauss-Newton diagonal.
    """
    from jax.nn import leaky_relu

    from pyhgf.updates.vectorised.learning.factors import (
        _importance_pair,
        _weight_quantities,
    )
    from pyhgf.utils.vectorised_belief_propagation import (
        _prediction_sweep,
        _update_sweep,
    )

    def build():
        net = (
            DeepNetwork(coupling_fn=leaky_relu)
            .add_layer(size=3, kind="categorical", volatility_parent=False)
            .add_layer(size=8, volatility_parent=False)
            .add_layer(size=2, volatility_parent=False)
        )
        # The tolerance on the interior matrix is calibrated on this draw: across
        # keys its median ratio ranges from about 0.7 to 3, while the head stays
        # within 0.1 of 1.
        return net.weight_initialisation("he", key=jax.random.key(2))

    net = build()
    x = jnp.asarray([0.7, -0.4])
    y = jnp.asarray([0.0, 1.0, 0.0])

    def increments(learning_kind):
        swept = _update_sweep(_prediction_sweep(net.state, x), y)
        factors = _weight_quantities(swept, learning_kind)
        pairs = [_importance_pair(f) for f in factors[1:]]
        return [np.asarray(p[:, None] * h2[None, :]) for p, h2 in pairs]

    # Ground truth: J^T (diag(p) - p p^T) J on the diagonal, by autodiff over the
    # same weights, with the constant input node written out explicitly.
    weights = [jnp.asarray(w) for w in net.state.weights]

    def logits(params, features):
        h = jnp.concatenate([leaky_relu(features), jnp.ones(1)])
        h = leaky_relu(params[1] @ h)
        return params[0] @ jnp.concatenate([h, jnp.ones(1)])

    probs = jax.nn.softmax(logits(weights, x))
    hessian = jnp.diag(probs) - jnp.outer(probs, probs)
    jacobian = jax.jacobian(logits)(weights, x)
    exact = [np.asarray(jnp.einsum("cij,cd,dij->ij", j, hessian, j)) for j in jacobian]

    def median_ratio(got, want):
        live = want > 1e-12
        return float(np.median(got[live] / want[live]))

    walked = increments("synaptic_uncertainty")

    # Both matrices reproduce the Gauss-Newton diagonal: the head because its
    # child is the clamped layer whose curvature seeds the walk, and the one
    # above it because the walk carries that seed up rather than re-reading a
    # chain the filter seeded at unit precision.
    for got, want in zip(walked, exact):
        assert median_ratio(got, want) == pytest.approx(1.0, abs=0.2)


def test_evidence_walk_through_a_layer_stack():
    """The walk carries the evidence through a stack's slices, by scan.

    Slice k's evidence is a function of slice k-1's, so it cannot be mapped over the
    slices the way the factors are; the scan that carries it must leave the stack with
    the same evidence a chain of plain layers would.
    """
    from pyhgf.updates.vectorised.learning.factors import _weight_quantities
    from pyhgf.utils.vectorised_belief_propagation import (
        _prediction_sweep,
        _update_sweep,
    )

    def build(stacked: bool):
        net = (
            DeepNetwork(coupling_fn=jax.nn.leaky_relu)
            .add_layer(size=3, kind="categorical", volatility_parent=False)
            .add_layer(size=6, add_constant_input=True, volatility_parent=False)
        )
        sizes = [6] * 5
        if stacked:
            net = net.add_layer_stack(
                layer_sizes=sizes, add_constant_input=True, volatility_parent=False
            )
        else:
            for size in sizes:
                net = net.add_layer(
                    size=size, add_constant_input=True, volatility_parent=False
                )
        return net.add_layer(size=2, add_constant_input=False).weight_initialisation(
            "he", key=jax.random.key(3)
        )

    stacked, plain = build(True), build(False)
    assert any(type(e).__name__ == "LayerStack" for e in stacked.state.layers)
    assert not any(type(e).__name__ == "LayerStack" for e in plain.state.layers)

    x = jnp.asarray([0.6, -0.2])
    y = jnp.asarray([0.0, 0.0, 1.0])

    def top_factors(net):
        swept = _update_sweep(_prediction_sweep(net.state, x), y)
        return _weight_quantities(swept, "synaptic_uncertainty")[-1]

    # The element above the stack sees the evidence the scan carried out of it,
    # so its importance factor is the check that the recursion was not mapped.
    got, want = np.asarray(top_factors(stacked)[2]), np.asarray(top_factors(plain)[2])

    # Compared as a ratio, and with the scale asserted first. Six pullbacks leave
    # the evidence around 1e-9, so any absolute tolerance loose enough to look
    # reasonable is orders of magnitude larger than the quantity itself and the
    # comparison passes whatever the stack did.
    assert np.all(want > 0.0)
    np.testing.assert_allclose(got / want, 1.0, rtol=1e-4)


def test_evidence_pullback_charges_the_process_noise_it_crosses():
    """The recursion softens the evidence once per level, not only at the increment.

    A message leaving a node crosses that node's random walk, so the precision that
    arrives above is ``xi / (1 + Omega * xi)`` and not ``xi``. The softening is part of
    the pullback, which makes it compound over depth; charging it only where the
    evidence feeds a weight increment would leave every level above the first uncharged.
    """
    from pyhgf.updates.vectorised.learning import evidence_pullback

    n_children, n_parents = 4, 3
    evidence = jnp.asarray([0.4, 0.9, 0.2, 0.7])
    weights = jnp.asarray(
        np.linspace(-1.0, 1.0, n_children * n_parents).reshape(n_children, n_parents)
    )
    parent = dataclasses.replace(
        LayerState.create(n_parents, has_volatility_parent=False),
        expected_mean=jnp.asarray([0.3, -0.5, 1.2]),
    )

    def pulled(process_noise):
        return np.asarray(
            evidence_pullback(
                parent_state=parent,
                child_evidence=evidence,
                child_process_noise=jnp.full(n_children, process_noise),
                weights=weights,
                coupling_fn=lambda m: m,
            )
        )

    # An identity coupling has unit derivative, so the pullback is the squared
    # weights applied to the softened evidence and nothing else.
    omega = 1.5
    softened = np.asarray(evidence) / (1.0 + omega * np.asarray(evidence))
    expected = np.asarray(weights).T ** 2 @ softened
    np.testing.assert_allclose(pulled(omega), expected, rtol=1e-6)

    # Without process noise the softening is exactly one, so a clamped layer, which
    # undergoes no random walk, seeds the walk unchanged.
    raw = np.asarray(weights).T ** 2 @ np.asarray(evidence)
    np.testing.assert_allclose(pulled(0.0), raw, rtol=1e-6)
    assert np.all(pulled(omega) < pulled(0.0))


@pytest.mark.parametrize("kind", ["binary", "categorical"])
@pytest.mark.parametrize("mask", [None, [1.0, 0.0, 1.0]])
def test_batch_importance_is_the_mean_likelihood_hessian(kind, mask):
    """On a clamped layer, the batch increment is the weighted mean Hessian diagonal."""
    x = jnp.array([[0.0, 0.5], [2.0, 1.0], [-1.0, 0.5]])
    if kind == "binary":
        weights = jnp.array([[0.8, -0.3]])
        targets = jnp.array([[0.0], [1.0], [0.0]])
    else:
        weights = jnp.array([[0.2, -0.7], [1.1, 0.3], [-0.4, 0.8]])
        targets = jax.nn.one_hot(jnp.array([0, 2, 1]), 3)
    row_weights = jnp.ones(3) if mask is None else jnp.asarray(mask)

    def mean_negative_log_likelihood(flat_weights):
        logits = x @ flat_weights.reshape(weights.shape).T
        if kind == "binary":
            per_row = (jax.nn.softplus(logits) - targets * logits).sum(axis=-1)
        else:
            per_row = jax.scipy.special.logsumexp(logits, axis=-1) - (
                targets * logits
            ).sum(axis=-1)
        return (row_weights * per_row).sum() / row_weights.sum()

    hessian = jax.hessian(mean_negative_log_likelihood)(weights.reshape(-1))
    net = (
        DeepNetwork()
        .add_layer(
            weights.shape[0],
            kind=kind,
            add_constant_input=False,
            volatility_parent=False,
        )
        .add_layer(2, add_constant_input=False, volatility_parent=False)
    )
    parent = dataclasses.replace(net.state.layers[1], weights_mean=weights)
    net.state = dataclasses.replace(net.state, layers=(net.state.layers[0], parent))
    net.batch_update(
        x,
        targets,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 100.0},
        update_precisions=False,
        sample_weight=None if mask is None else row_weights,
    )
    np.testing.assert_allclose(
        net.state.layers[1].weights_precision_delta,
        jnp.diag(hessian).reshape(weights.shape),
        rtol=2e-5,
        atol=1e-7,
    )


def test_zero_total_weight_keeps_the_prior_dynamics():
    """Without data, the weight belief still decays and reverts towards its prior."""
    net = (
        DeepNetwork()
        .add_layer(1, kind="binary", add_constant_input=False, volatility_parent=False)
        .add_layer(1, add_constant_input=False, volatility_parent=False)
        .install_weight_belief()
    )
    parent = dataclasses.replace(
        net.state.layers[1], weights_precision_delta=jnp.array([[2.0]])
    )
    net.state = dataclasses.replace(net.state, layers=(net.state.layers[0], parent))
    net.batch_update(
        jnp.array([[0.0], [2.0]]),
        jnp.array([[0.0], [1.0]]),
        sample_weight=jnp.zeros(2),
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 10.0},
        update_precisions=False,
    )
    np.testing.assert_allclose(net.state.layers[1].weights_precision_delta, [[1.8]])
    np.testing.assert_allclose(net.state.layers[1].weights_mean, [[1.0 - 1.0 / 30.0]])


def _pipeline_weight_beliefs(stacked, weight_reuse):
    """Weight means and precisions of every slice after one fused pipeline step."""
    net = DeepNetwork(coupling_fn=jnp.tanh).add_layer(
        2, kind="categorical", add_constant_input=False, volatility_parent=False
    )
    kwargs = dict(add_constant_input=False, volatility_parent=False, precision=2.0)
    if stacked:
        net.add_layer_stack([2] * 5, **kwargs)
    else:
        for _ in range(5):
            net.add_layer(2, **kwargs)
    net = net.add_layer(2, **kwargs).weight_initialisation(
        "xavier", key=jax.random.key(7)
    )
    assert stacked == any(isinstance(layer, LayerStack) for layer in net.state.layers)
    pipeline = FusedPipeline(
        DeepNetworkAdapter(
            net,
            learning_kind="synaptic_uncertainty",
            learning_kwargs={"window": 100.0},
            update_precisions=False,
            weight_reuse=weight_reuse,
        ),
        error_normalisation="rows",
    )
    pipeline.step(
        jnp.array([[0.5, -0.1], [1.3, 0.4], [-0.8, 0.7]]),
        jax.nn.one_hot(jnp.array([0, 1, 0]), 2),
    )
    pipeline.merge()
    beliefs = []
    for layer in net.state.layers:
        if layer.weights_mean is None:
            continue
        pairs = (layer.weights_mean, layer.weights_precision_delta)
        beliefs.extend(zip(*pairs) if isinstance(layer, LayerStack) else [pairs])
    return beliefs


def test_batch_importance_reaches_stacks_and_weight_reuse():
    """A LayerStack learns as its unrolled layers, and reuse scales every slice."""
    stacked = _pipeline_weight_beliefs(True, 3.0)
    unrolled = _pipeline_weight_beliefs(False, 3.0)
    single_use = _pipeline_weight_beliefs(True, 1.0)
    assert len(stacked) == len(unrolled) == len(single_use) == 6
    for (w, p), (w_ref, p_ref), (_, p_one) in zip(stacked, unrolled, single_use):
        np.testing.assert_allclose(w, w_ref, rtol=2e-4, atol=1e-7)
        np.testing.assert_allclose(p, p_ref, rtol=2e-4, atol=1e-9)
        np.testing.assert_allclose(p, 3.0 * p_one, rtol=2e-4, atol=1e-9)


def _recording_network(kind="volatile", precision=2.0, record=True):
    """Two layers, the top one clamped to three inputs plus a bias column.

    A Gaussian (``"volatile"``) bottom layer without a volatility parent is seeded with
    its own precision, one constant across units and samples, which is the case where
    the accumulated precision of every row is that constant times the diagonal of the
    input Gram matrix.
    """
    bottom = dict(add_constant_input=False, volatility_parent=False)
    if kind == "volatile":
        bottom["precision"] = precision
    net = (
        DeepNetwork(coupling_fn=jnp.tanh)
        .add_layer(2 if kind == "volatile" else 3, kind=kind, **bottom)
        .add_layer(3, add_constant_input=True, volatility_parent=False)
        .weight_initialisation("xavier", key=jax.random.key(3))
        .install_weight_belief()
    )
    return _install_records(net) if record else net


def _install_records(
    net,
    input_gram=True,
    step_gram=False,
    rate=0.1,
    rows=False,
    batch=False,
):
    """Install recorded fields on every Layer holding weights, as a caller does.

    ``input_gram`` installs ``weights_input_gram`` and ``weights_gradient`` at zero.
    ``step_gram`` installs ``weights_step_gram`` and ``weights_step_mass`` at zero, with
    ``weights_step_rate`` at ``rate`` (left out when ``None``), ``weights_step_rows`` at
    zero (``rows``) and ``weights_batch_gram`` at zero (``batch``).
    """
    layers = []
    for elem in net.state.layers:
        if isinstance(elem, LayerStack) or elem.weights_mean is None:
            layers.append(elem)
            continue
        weights = elem.weights_mean
        fields = {}
        if input_gram:
            fields["weights_input_gram"] = create_input_gram(weights)
            fields["weights_gradient"] = jnp.zeros_like(weights)
        if step_gram:
            gram, mass = create_step_gram(weights)
            fields.update(weights_step_gram=gram, weights_step_mass=mass)
            if rate is not None:
                fields["weights_step_rate"] = jnp.asarray(rate, weights.dtype)
            if rows:
                fields["weights_step_rows"] = jnp.zeros(
                    weights.shape[:-1], weights.dtype
                )
            if batch:
                fields["weights_batch_gram"] = create_input_gram(weights)
        layers.append(dataclasses.replace(elem, **fields))
    net.state = dataclasses.replace(net.state, layers=tuple(layers))
    return net


def _augmented_inputs(x):
    """Return the coupled activations of the clamped top layer, bias appended."""
    return jnp.concatenate([jnp.tanh(x), jnp.ones((x.shape[0], 1))], axis=1)


def test_input_gram_is_the_weighted_mean_outer_product():
    """Each batch step adds the weighted mean of h h^T and decays at 1 / window."""
    x = jnp.array([[0.0, 0.5, -1.0], [2.0, 1.0, 0.3], [-1.0, 0.5, 0.8]])
    second_x = x[::-1] * 0.5
    targets = jax.nn.one_hot(jnp.array([0, 2, 1]), 3)
    row_weights = jnp.array([1.0, 0.0, 3.0])
    settings = {"window": 8.0, "increment_scale": 0.7}
    net = _recording_network(kind="categorical")
    for inputs in (x, second_x):
        net.batch_update(
            inputs,
            targets,
            learning_kind="synaptic_uncertainty",
            learning_kwargs=settings,
            update_precisions=False,
            sample_weight=row_weights,
        )

    def mean_outer(inputs):
        h = _augmented_inputs(inputs)
        return jnp.einsum("b,bi,bj->ij", row_weights / row_weights.sum(), h, h)

    first = 0.7 * mean_outer(x)
    expected = first + 0.7 * mean_outer(second_x) - first / 8.0
    np.testing.assert_allclose(
        net.state.layers[1].weights_input_gram, expected, rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("path", ["fit", "batch_update"])
def test_input_gram_diagonal_is_the_accumulated_precision(path):
    """With a constant evidence precision, every row of the precision is xi diag(G).

    Both learning paths advance the two fields with the same increment scale and the
    same window, so the diagonal of the Gram matrix differs from each row of the
    accumulated precision only by the evidence precision of the clamped layer.
    """
    x = jax.random.normal(jax.random.key(0), (6, 3))
    y = jax.random.normal(jax.random.key(1), (6, 2))
    settings = {"window": 20.0, "increment_scale": 0.7}
    net = _recording_network(precision=2.0)
    for step in range(3):
        if path == "fit":
            net.fit(
                x + step,
                y,
                learning_kind="synaptic_uncertainty",
                learning_kwargs=settings,
                update_precisions=False,
            )
        else:
            net.batch_update(
                x + step,
                y,
                learning_kind="synaptic_uncertainty",
                learning_kwargs=settings,
                update_precisions=False,
            )
    top = net.state.layers[1]
    assert bool(jnp.all(jnp.diag(top.weights_input_gram) > 0))
    np.testing.assert_allclose(
        top.weights_input_gram, top.weights_input_gram.T, rtol=1e-6
    )
    np.testing.assert_allclose(
        top.weights_precision_delta,
        2.0
        * jnp.broadcast_to(
            jnp.diag(top.weights_input_gram), top.weights_precision_delta.shape
        ),
        rtol=1e-5,
    )


@pytest.mark.parametrize("path", ["fit", "batch_update"])
def test_recorded_fields_leave_the_update_unchanged(path):
    """Recording changes no weight or precision, and records the gradient divided.

    From a fresh install with a zero prior mean, one step moves the weights by
    ``-learning_rate * g / prior_precision - weights / window``, so the recorded
    gradient ``g`` is recovered from the weights before and after the step.
    """
    x = jax.random.normal(jax.random.key(4), (5, 3))
    y = jax.random.normal(jax.random.key(5), (5, 2))
    settings = {"window": 10.0, "prior_variance": 0.5, "learning_rate": 0.3}
    learned = []
    for record in (False, True):
        net = _recording_network(record=record)
        before = np.asarray(net.state.layers[1].weights_mean)
        getattr(net, path)(
            x[:1] if path == "fit" else x,
            y[:1] if path == "fit" else y,
            learning_kind="synaptic_uncertainty",
            learning_kwargs=settings,
            update_precisions=False,
        )
        learned.append(net.state.layers[1])
    plain, recorded = learned
    assert plain.weights_input_gram is None and plain.weights_gradient is None
    np.testing.assert_array_equal(plain.weights_mean, recorded.weights_mean)
    np.testing.assert_array_equal(
        plain.weights_precision_delta, recorded.weights_precision_delta
    )
    moved = before - before / 10.0 - recorded.weights_mean
    np.testing.assert_allclose(
        recorded.weights_gradient, moved * 2.0 / 0.3, rtol=1e-4, atol=1e-6
    )


def test_input_gram_through_the_pipeline_with_weight_reuse():
    """The fused pipeline advances the Gram matrix, scaled by weight reuse as Delta is.

    Weight reuse multiplies the importance increment, so it multiplies the Gram
    increment by the same factor and the identity of the previous test still holds.
    """
    x = jnp.array([[0.5, -0.1, 0.2], [1.3, 0.4, -0.6], [-0.8, 0.7, 0.1]])
    net = _recording_network(precision=2.0)
    pipeline = FusedPipeline(
        DeepNetworkAdapter(
            net,
            learning_kind="synaptic_uncertainty",
            learning_kwargs={"window": 100.0},
            update_precisions=False,
            weight_reuse=3.0,
        ),
        error_normalisation="rows",
    )
    pipeline.step(x, jnp.zeros((3, 2)))
    pipeline.merge()
    top = net.state.layers[1]
    h = _augmented_inputs(x)
    np.testing.assert_allclose(
        top.weights_input_gram, 3.0 * h.T @ h / 3.0, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        top.weights_precision_delta,
        2.0
        * jnp.broadcast_to(
            jnp.diag(top.weights_input_gram), top.weights_precision_delta.shape
        ),
        rtol=1e-5,
    )
    assert bool(jnp.any(top.weights_gradient != 0))


def test_step_gram_requires_a_rate():
    """A running Gram matrix installed without its mixing rate is rejected."""
    net = _install_records(
        _recording_network(record=False), input_gram=False, step_gram=True, rate=None
    )
    x = jax.random.normal(jax.random.key(7), (6, 3))
    y = jax.random.normal(jax.random.key(8), (6, 2))
    with pytest.raises(ValueError, match="weights_step_rate"):
        net.batch_update(
            x,
            y,
            learning_kind="synaptic_uncertainty",
            learning_kwargs={"window": 1e6},
            update_precisions=False,
        )


def test_step_gram_is_the_bias_corrected_average_of_the_batches():
    """After one step G / m is that batch's Gram matrix; after two, their average."""
    x = jax.random.normal(jax.random.key(7), (6, 3))
    y = jax.random.normal(jax.random.key(8), (6, 2))
    settings = {"window": 1e6}
    net = _install_records(
        _recording_network(precision=1.0, record=False),
        input_gram=False,
        step_gram=True,
        rate=0.25,
    )
    h1, h2 = _augmented_inputs(x), _augmented_inputs(x * 0.5 + 1.0)
    first, second = h1.T @ h1 / 6, h2.T @ h2 / 6
    net.batch_update(
        x,
        y,
        learning_kind="synaptic_uncertainty",
        learning_kwargs=settings,
        update_precisions=False,
    )
    top = net.state.layers[1]
    np.testing.assert_allclose(
        top.weights_step_gram / top.weights_step_mass, first, rtol=1e-5
    )
    net.batch_update(
        x * 0.5 + 1.0,
        y,
        learning_kind="synaptic_uncertainty",
        learning_kwargs=settings,
        update_precisions=False,
    )
    top = net.state.layers[1]
    expected = (0.75 * 0.25 * first + 0.25 * second) / (0.75 * 0.25 + 0.25)
    np.testing.assert_allclose(
        top.weights_step_gram / top.weights_step_mass, expected, rtol=1e-5
    )
    assert top.weights_input_gram is None


def _evidence_network(kind="volatile", rate=0.1):
    """Return the recording network with the row evidence and batch Gram recorded."""
    return _install_records(
        _recording_network(kind=kind, record=False),
        step_gram=True,
        rate=rate,
        rows=True,
        batch=True,
    )


def test_gram_evidence_is_off_by_default():
    """The evidence weighting of the Gram matrices is opt-in."""
    resolve = resolve_synaptic_uncertainty_settings
    assert resolve({"window": 10}).gram_evidence is False
    assert resolve({"window": 10, "gram_evidence": True}).gram_evidence is True


def test_evidence_gram_weights_samples_by_their_evidence_at_the_unweighted_trace():
    """At a categorical child each sample counts by its mean p(1 - p).

    The batch Gram matrix is sum_t w_t r_t h_t h_t^T rescaled to the trace of the
    unweighted one, with r_t the mean over classes of p(1 - p) for the predicted
    probabilities p. The same matrix is recorded, mixed into the step Gram matrix and
    added to the input Gram matrix, and the running row evidence is proportional to
    sum_t w_t p_a,t (1 - p_a,t) ||h_t||^2.
    """
    x = jax.random.normal(jax.random.key(21), (5, 3)) * 2.0
    targets = jax.nn.one_hot(jnp.array([0, 2, 1, 1, 0]), 3)
    sample_weight = jnp.array([1.0, 2.0, 0.5, 1.0, 3.0])
    net = _evidence_network("categorical")
    probs = net.predict(x)
    net.batch_update(
        x,
        targets,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 1e6, "gram_evidence": True},
        update_precisions=False,
        sample_weight=sample_weight,
    )
    h = _augmented_inputs(x)
    w = sample_weight / sample_weight.sum()
    evidence = probs * (1.0 - probs)
    r = evidence.mean(axis=1)
    norms = (h**2).sum(axis=1)
    expected = jnp.einsum("b,bi,bj->ij", w * r, h, h) * (w @ norms) / ((w * r) @ norms)
    plain = jnp.einsum("b,bi,bj->ij", w, h, h)
    assert not np.allclose(expected, plain, rtol=1e-3)
    top = net.state.layers[1]
    np.testing.assert_allclose(top.weights_batch_gram, expected, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(jnp.trace(top.weights_batch_gram), w @ norms, rtol=1e-5)
    np.testing.assert_allclose(
        top.weights_step_gram / top.weights_step_mass, expected, rtol=1e-4, atol=1e-7
    )
    np.testing.assert_allclose(top.weights_input_gram, expected, rtol=1e-4, atol=1e-7)
    rows = jnp.einsum("b,ba,b->a", w, evidence, norms)
    np.testing.assert_allclose(
        top.weights_step_rows / top.weights_step_rows.mean(),
        rows / rows.mean(),
        rtol=1e-4,
    )


def test_constant_evidence_leaves_the_gram_unchanged():
    """A continuous clamped child gives one constant evidence, which changes nothing."""
    x = jax.random.normal(jax.random.key(22), (6, 3))
    y = jax.random.normal(jax.random.key(23), (6, 2))
    recorded = []
    for flag in (False, True):
        net = _evidence_network("volatile")
        net.batch_update(
            x,
            y,
            learning_kind="synaptic_uncertainty",
            learning_kwargs={"window": 1e6, "gram_evidence": flag},
            update_precisions=False,
        )
        recorded.append(net.state.layers[1])
    plain, weighted = recorded
    h = _augmented_inputs(x)
    np.testing.assert_allclose(weighted.weights_batch_gram, h.T @ h / 6, rtol=1e-5)
    np.testing.assert_allclose(
        weighted.weights_batch_gram, plain.weights_batch_gram, rtol=1e-5
    )
    np.testing.assert_allclose(
        weighted.weights_step_rows, weighted.weights_step_rows.mean(), rtol=1e-5
    )


def test_step_rows_mix_at_the_step_rate():
    """The row evidence is mixed in at the layer's step rate, as the Gram matrix is.

    The child is continuous and clamped at precision 2, so every row's batch evidence is
    2 times the batch mean of ||h||^2.
    """
    x = jax.random.normal(jax.random.key(24), (6, 3))
    y = jax.random.normal(jax.random.key(25), (6, 2))
    net = _evidence_network("volatile", rate=0.5)
    batches = (x, x * 0.5 + 1.0)
    for inputs in batches:
        net.batch_update(
            inputs,
            y,
            learning_kind="synaptic_uncertainty",
            learning_kwargs={"window": 1e6},
            update_precisions=False,
        )
    first, second = (
        2.0 * float((_augmented_inputs(b) ** 2).sum(axis=1).mean()) for b in batches
    )
    np.testing.assert_allclose(
        net.state.layers[1].weights_step_rows,
        np.full(2, 0.5 * 0.5 * first + 0.5 * second),
        rtol=1e-5,
    )


@pytest.mark.parametrize("path", ["fit", "batch_update"])
def test_one_to_one_layer_records_only_its_edges(path):
    """A one-to-one layer records the diagonal Gram matrices and edge-only row evidence.

    Row a of a one-to-one weight matrix has the single input h_a, so its input-side
    curvature is h_a^2 and its row evidence is the batch mean of p_a h_a^2. The inputs
    are correlated, so the products h_i h_j with i != j are far from zero, yet none is
    recorded, and the recorded gradient is masked to the diagonal. The clamped child
    is continuous at precision 2, so p = 2 on every row.
    """
    x = jax.random.normal(jax.random.key(31), (6, 3)) * 0.3 + jax.random.normal(
        jax.random.key(32), (6, 1)
    )
    y = jax.random.normal(jax.random.key(33), (6, 3))
    if path == "fit":
        x, y = x[:1], y[:1]
    net = (
        DeepNetwork(coupling_fn=jnp.tanh)
        .add_layer(3, add_constant_input=False, volatility_parent=False, precision=2.0)
        .add_layer(
            3, add_constant_input=False, fully_connected=False, volatility_parent=False
        )
        .weight_initialisation("xavier", key=jax.random.key(34))
        .install_weight_belief()
    )
    net = _install_records(net, step_gram=True, rows=True, batch=True)
    getattr(net, path)(
        x,
        y,
        learning_kind="synaptic_uncertainty",
        learning_kwargs={"window": 1e6},
        update_precisions=False,
    )
    top = net.state.layers[1]
    squares = jnp.mean(jnp.tanh(x) ** 2, axis=0)
    full = jnp.tanh(x).T @ jnp.tanh(x) / x.shape[0]
    assert float(jnp.min(jnp.abs(full - jnp.diag(jnp.diag(full))) + jnp.eye(3))) > 0.05
    for gram in (
        top.weights_batch_gram,
        top.weights_input_gram,
        top.weights_step_gram / top.weights_step_mass,
    ):
        np.testing.assert_allclose(gram, jnp.diag(squares), rtol=1e-5, atol=1e-7)
    off_diagonal = ~np.eye(3, dtype=bool)
    np.testing.assert_array_equal(np.asarray(top.weights_gradient)[off_diagonal], 0.0)
    assert bool(jnp.all(jnp.diag(top.weights_gradient) != 0))
    # Mixed in from zero at the installed rate 0.1.
    np.testing.assert_allclose(top.weights_step_rows, 0.1 * 2.0 * squares, rtol=1e-5)
