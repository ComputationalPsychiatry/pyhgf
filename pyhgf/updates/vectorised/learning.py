# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Vectorised weight learning for deep predictive coding networks."""

import math
from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import grad as jgrad
from jax import vmap

from pyhgf.typing.vectorised import LayerState

# The accepted weight-update kinds. All three share the same base term and
# factorise into a child-side and a parent-side vector (a rank-one product),
# which is why the gradient below is assembled as a single outer product.
SEPARABLE_KINDS: tuple = ("standard", "precision_weighted", "synaptic_uncertainty")

#: Floor applied to precisions before any division.
_EPS = 1e-30


def learning_weights_vectorised(
    parent_state: LayerState,
    child_state: LayerState,
    coupling_fn: Callable,
    kind: str = "precision_weighted",
    parent_has_constant: bool = False,
    child_kind: str = "continuous",
    child_evidence: Optional[jnp.ndarray] = None,
) -> tuple:
    r"""Per-layer weight-learning factors for the vectorised deep network.

    The single entry point of this module. One weight update has two halves and
    this returns the pieces of both: the descent gradient that moves the
    weight's mean, and the importance increment that raises its precision. They
    are the first and second derivative of one layer-local energy, computed in
    one pass.

    Both are rank-one and **share their parent-side factor**. The gradient is
    :math:`u \otimes h` and the increment is :math:`p \otimes h^2`, with the
    same coupled parent activation :math:`h_i = g(\mu_i)`, so three vectors
    carry both quantities. Returning factors rather than assembled matrices
    lets a batched caller average over samples and contract once
    (``einsum('bi,bj->ij') / batch``): the same arithmetic, without
    materialising one weight-matrix-sized array per sample.

    Both are strictly *local*: what one weight gets reads only the prediction
    error and precision at its child and the activation at its parent, never a
    reduction over the other parents or children in the layer.

    **The child-side gradient factor** starts from the value prediction error
    :math:`\delta = \mu_\text{child} - \hat{\mu}_\text{child}`, and *kind*
    decides what scales it:

    - ``"standard"`` takes :math:`\delta` alone, the raw gradient of an
      *unweighted* squared error. This coincides with the free-energy gradient
      only where the child precision is one (the unit-precision output or
      categorical convention).
    - ``"precision_weighted"`` scales by the child's *posterior* precision,
      :math:`\delta\,\pi_a`. This is the **backprop-parity** mode: a moved
      interior belief shifts by (routed error) / posterior precision, so
      weighting by that same posterior precision cancels the division and
      reproduces the backpropagated gradient node for node at *any* precision
      setting.
    - ``"synaptic_uncertainty"`` charges the process noise between the weight and
      its child on top, :math:`\delta\,\pi_a / (1 + \Omega_a \xi_a)`, so the
      gradient and the increment become the two derivatives of one energy.
      This is the kind the weight-belief rule descends, and the only one it
      can: dividing it by the weight's own precision is the Gaussian
      posterior step, so both halves of that update read the same evidence.
      It is *not* backprop parity.

    A binary child drops the precision factor either way: its ``precision``
    field holds the Bernoulli variance :math:`p(1-p)`, which cancels through
    the sigmoid in the gradient, so keeping it would count the same term
    twice. That cancellation belongs to the *first* derivative only, and the
    curvature below keeps it.

    **The child-side importance factor** is the exact Hessian diagonal of the
    layer-local variational energy: that energy is exactly quadratic in the
    weights for any coupling function, because the coupling acts on the
    activation while the map :math:`w_{ai} \mapsto w_{ai} h_i` stays linear, so
    one observation adds :math:`\tilde{\xi}_a h_i^2` with no approximation and
    no sampling. Where the evidence comes from depends on the child:

    - A **binary** or **categorical** child is where the data are clamped, so
      the evidence is the curvature of its own likelihood, per unit
      :math:`p_a(1 - p_a)` with :math:`p_a` the child's expected mean. For the
      categorical case the Hessian of the log-likelihood with respect to the
      logits is :math:`\operatorname{diag}(\mathbf{p}) -
      \mathbf{p}\mathbf{p}^{\top}`, whose diagonal is that expression. No
      label enters it, since averaging the outer product of the residual over
      labels drawn from the model returns the same matrix, so this is the
      model's own expected curvature rather than an estimate built from
      observed errors.
    - A **continuous** child passes on the evidence it received from below,
      softened by the process noise it had to cross,
      :math:`\tilde\xi_a = (1/\xi_a + \Omega_a)^{-1}`, with
      :math:`\Omega_a = \gamma_a / \hat\pi_a` recovered from the effective
      precision the prediction sweep writes as
      :math:`\gamma_a = \Omega_a \hat\pi_a`. The evidence :math:`\xi_a`
      itself is supplied by the caller through *child_evidence*, carried up from
      the clamped layer by :func:`evidence_pullback`, and the weight-belief rule
      always supplies it.

      Without it, :math:`\xi_a` falls back to :math:`\pi_a - \hat\pi_a` from
      the cache, which is the same quantity in exact arithmetic *only* where the
      filter's own precision chain was seeded with the clamped layer's likelihood
      curvature. It is not: a clamped categorical layer enters that chain at unit
      precision, because that convention is what makes the message it routes exact
      cross-entropy backpropagation. Reading the difference also subtracts two
      large nearly-equal precisions, which loses every significant digit once a
      layer's precision has accumulated. The fallback is kept for direct callers
      reading the layer-local quantity; it is not what the rule descends.

    Parameters
    ----------
    parent_state :
        Current state of the parent layer.
    child_state :
        Current state of the child layer (with observations), after the update
        sweep has written its posterior.
    coupling_fn :
        Coupling function applied to parent means.
    kind :
        The metric the gradient is expressed in, one of
        :data:`SEPARABLE_KINDS`. It also sets the importance convention:
        ``"standard"`` uses unit observation precision on both halves.
    parent_has_constant :
        If True, the parent layer has a constant input node (mean = 1.0,
        precision = 1.0) appended to its activations after coupling.
    child_kind :
        The child layer's node kind, ``"binary"``, ``"categorical"`` or
        anything else for a continuous one.
    child_evidence :
        The child's evidence precision :math:`\xi_a`, supplied by a caller that
        carries it up the stack itself (:func:`evidence_pullback`) rather than
        letting it be recovered here as :math:`\pi_a - \hat\pi_a`. When given,
        it replaces that difference in the importance factor and the fallback
        below never applies, since a carried evidence is non-negative by
        construction. The gradient factor is unaffected: it reads the child's
        posterior precision either way. ``None`` (default) recovers the evidence
        from the cache.

    Returns
    -------
    factors :
        The triple ``(u, h, p)``. The child-side gradient factor :math:`u` and
        the child-side importance factor
        :math:`p` have shape ``(n_children,)``; the shared parent-side factor
        :math:`h` has shape ``(n_parents[+1],)``. The gradient is
        ``u[:, None] * h[None, :]`` and the increment
        ``p[:, None] * (h ** 2)[None, :]``. Non-finite entries are zeroed, so
        optax never propagates a NaN or an inf through its moment
        accumulators.

    Raises
    ------
    ValueError
        If *kind* is unrecognised.
    """
    if kind not in SEPARABLE_KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {SEPARABLE_KINDS}.")

    # The shared parent-side factor, h_i = g(mu_i). A constant input node
    # contributes a fixed activation of 1.0, so its weight counts as usage 1.
    h = coupling_fn(parent_state.mean)
    if parent_has_constant:
        h = jnp.concatenate([h, jnp.ones(1)])
    h = jnp.where(jnp.isfinite(h), h, 0.0)

    # The evidence the child received from below, and the factor by which the
    # process noise between the weight and the child attenuates it. Both halves
    # of the update read these, so they are formed once. The evidence is
    # floored at zero: clipping can drive a posterior precision below its
    # prediction, which must not turn the increment into a subtraction, nor the
    # softening below into an amplification.
    if child_evidence is None:
        evidence = child_state.precision - child_state.expected_precision
    else:
        evidence = child_evidence
    floored = jnp.maximum(evidence, 0.0)

    # The child's process noise. The prediction sweep caches the effective
    # precision gamma = Omega * pi_hat, so Omega = gamma / pi_hat. It is
    # zero at a clamped layer, which undergoes no random walk.
    process_noise = child_state.effective_precision / jnp.maximum(
        child_state.expected_precision, _EPS
    )

    # A message crossing that noise has the two variances convolved, so an
    # evidence precision xi arrives as xi_tilde = (1/xi + Omega)^-1
    # = xi / (1 + Omega xi). The denominator is kept on its own because both
    # halves need it: the increment is xi_tilde h^2, while the gradient factor
    # divides the child's displacement by the same quantity.
    softening = 1.0 / (1.0 + process_noise * floored)

    # Child-side gradient factor, in descent form: sign-flipped from the
    # natural "ascent" formulation so it composes with standard optax
    # (apply_updates performs weights + updates, and sgd(lr).update returns
    # -lr * grad).
    u = child_state.mean - child_state.expected_mean
    if kind != "standard" and child_kind != "binary":
        u = u * child_state.precision  # posterior precision
        if kind == "synaptic_uncertainty":
            u = u * softening
    u = -jnp.where(jnp.isfinite(u), u, 0.0)

    # Child-side importance factor. A caller advancing only the means ignores
    # it; the two array operations below are removed with it by dead-code
    # elimination, so gating them on a flag would buy nothing.
    if kind == "standard":
        p = jnp.ones_like(child_state.mean)
    elif child_evidence is not None:
        # A carried evidence is already the right quantity for any child: the
        # walk seeds a clamped discrete layer with its own p(1-p) and pulls that
        # up, so no per-kind branch and no fallback are needed here.
        p = floored * softening
    elif child_kind in ("binary", "categorical"):
        # Clamped discrete child: the curvature of its own likelihood.
        prob = child_state.expected_mean
        p = prob * (1.0 - prob)
    else:
        # A clamped continuous layer receives no message from below, so no
        # evidence arrives and its own precision stands in. That fallback is
        # the uniform clamp-precision convention and carries no per-unit
        # curvature; seeding such a layer with the true curvature of its
        # likelihood is separate work.
        p = jnp.where(evidence > 0.0, floored * softening, child_state.precision)
    p = jnp.where(jnp.isfinite(p), p, 0.0)

    return u, h, p


# ----------------------------------------------------------------- variational

#: The learning kind whose gradient is not rank-one and is assembled by
#: :func:`vectorised_variational_gradient` instead of the factors above.
VARIATIONAL_KIND: str = "variational"

#: The learning kind whose gradient ignores the inference: the gradient of the
#: precision-weighted squared error of the network's forward pass, by automatic
#: differentiation (see
#: :func:`pyhgf.utils.vectorised_belief_propagation._backprop_gradients`).
BACKPROP_KIND: str = "backprop"


def relaxed_effective_precision(
    child_state: LayerState,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Relaxed effective precision of a child layer, one value per unit.

    .. math::

        s_a = \frac{\bar{\pi}_a\, \pi^y_a}{\bar{\pi}_a + \pi^y_a},
        \qquad \pi^y_a = \max(\pi_a - \hat{\pi}_a, 0),

    with :math:`\bar{\pi}_a` the conditional expected precision of the child and
    :math:`\pi^y_a` the evidence it received from below. :math:`s_a` is the precision
    of the prediction of unit :math:`a` under the observations: one over the sum of the
    variance of the prediction given the parents, :math:`1 / \bar{\pi}_a`, and the
    variance the evidence leaves, :math:`1 / \pi^y_a`. It is :math:`\bar{\pi}_a` for the
    clamped observed layer (:math:`\pi^y_a \to \infty`, the observations are known
    exactly) and under ``mean_field_updates``, whose posterior has no covariance
    between layers.

    Parameters
    ----------
    child_state :
        The child layer after the update sweep, as its parent sees it.
    child_is_input_layer :
        Whether the child is the clamped observed layer.
    mean_field_updates :
        Whether the beliefs follow the mean-field updates.

    Returns
    -------
    jnp.ndarray
        :math:`s_a`, shape ``(n_child,)``.
    """
    conditional = child_state.conditional_expected_precision
    if child_is_input_layer or mean_field_updates:
        return conditional
    evidence = jnp.maximum(child_state.precision - child_state.expected_precision, 0.0)
    return conditional * evidence / jnp.maximum(conditional + evidence, _EPS)


def vectorised_variational_gradient(
    parent_state: LayerState,
    child_state: LayerState,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
    child_is_input_layer: bool = False,
    mean_field_updates: bool = False,
) -> jnp.ndarray:
    r"""Descent gradient of one weight matrix under the posterior of the beliefs.

    The weights :math:`W` predict child :math:`a` from the parents :math:`b` with
    precision :math:`\bar{\pi}_a` (the child's conditional expected precision),
    :math:`x_a \sim \mathcal{N}(\sum_b W_{ab}\, g(x_b) + c_a, 1/\bar{\pi}_a)`. This
    returns the gradient of the expected log-likelihood of that factor under the
    posterior of the beliefs (the M-step of variational expectation maximisation),
    linearising :math:`g` at the posterior means:

    .. math::

        \frac{\partial}{\partial W_{ab}} = \bar{\pi}_a \Big( \rho_a\, g(\mu_b)
            + g'(\mu_b)\, \mathrm{Cov}(x_a, x_b)
            - W_{ab}\, g'(\mu_b)^2\, \mathrm{Var}(x_b) \Big),

    with the units of the parent layer independent of each other, :math:`\rho_a` the
    posterior mean of the child's noise :math:`x_a - \sum_b W_{ab}\, g(x_b) - c_a`,
    and :math:`\mathrm{Var}(x_b) = 1/\pi_b` the parent's posterior variance. On the
    relaxed edge the posterior covariance is :math:`\mathrm{Cov}(x_a, x_b) =
    \bar{\pi}_a W_{ab}\, g'(\mu_b) / ((\bar{\pi}_a + \pi^y_a)\, \pi_b)`, with
    :math:`\pi^y_a = \pi_a - \hat{\pi}_a` the evidence the child received from below,
    so the two variance terms combine into

    .. math::

        \frac{\partial}{\partial W_{ab}} = \bar{\pi}_a\, \rho_a\, g(\mu_b)
            - W_{ab}\, s_a\, \frac{g'(\mu_b)^2}{\pi_b},
        \qquad s_a = \frac{\bar{\pi}_a\, \pi^y_a}{\bar{\pi}_a + \pi^y_a}.

    :math:`s_a` is the relaxed effective precision of the child. It is
    :math:`\bar{\pi}_a` for the clamped observed layer (:math:`\pi^y_a \to \infty`, the
    observations are known exactly) and under ``mean_field_updates``, whose posterior
    has no covariance between layers. The second term shrinks each weight in
    proportion to the posterior variance of its parent. The bias column takes
    :math:`\bar{\pi}_a\, \rho_a`.

    The noise is read at the parents' posterior means. Given them, the child's prior
    is :math:`\mathcal{N}(\hat{\mu}_a + \Delta_a, 1/\bar{\pi}_a)`, with
    :math:`\Delta_a = \sum_b W_{ab}\, (g(\mu_b) - g(\hat{\mu}_b))` the shift of its
    prediction, and the evidence from below brings the precision :math:`\pi^y_a` and
    the information :math:`\pi_a \delta_a`, with :math:`\delta_a = \mu_a -
    \hat{\mu}_a` the child's prediction error. Combining the two gives

    .. math::

        \bar{\pi}_a\, \rho_a = \frac{\bar{\pi}_a}{\bar{\pi}_a + \pi^y_a}
            \big( \pi_a \delta_a - \pi^y_a \Delta_a \big)
            = \frac{\bar{\pi}_a\, \pi_a}{\bar{\pi}_a + \pi^y_a}\, \delta_a
            - s_a\, \Delta_a,

    the prediction error the child sends its parents, weighted by the relaxed gain of
    :func:`~pyhgf.updates.vectorised.volatile.posterior.vectorised_posterior_update_mean_value_level`,
    less the part of it that the parents' moves explain. For the clamped observed
    layer and under ``mean_field_updates`` this is the difference of the posterior
    means, :math:`\rho_a = \mu_a - \sum_b W_{ab}\, g(\mu_b) - c_a`. On a hidden child
    with more than one unit the two differ: the child's mean is formed with the
    diagonal of its predicted covariance, while the parents' moves also carry the
    evidence its sibling units received, so the difference of the means keeps a share
    of those moves that does not shrink as the hidden precisions grow. With
    :math:`\rho_a` read at the parents' posterior means, under the relaxed updates,
    the gradient tends to that of the forward pass (the ``"backprop"`` kind) as the
    conditional expected precisions of the hidden layers grow, provided the variance
    of the top layer, clamped to the predictors, stays negligible against theirs.

    Parameters
    ----------
    parent_state :
        The parent layer after the update sweep (posterior means and precisions).
    child_state :
        The child layer after the update sweep, as its parent sees it (see
        :func:`pyhgf.utils.vectorised_belief_propagation._child_view`).
    weights :
        The weight matrix predicting the child, shape ``(n_child, n_parent)`` or
        ``(n_child, n_parent + 1)`` with the bias as the last column.
    coupling_fn :
        Coupling function applied to the parent means.
    parent_has_constant :
        Whether the last column of ``weights`` is the bias.
    child_is_input_layer :
        Whether the child is the clamped observed layer.
    mean_field_updates :
        Whether the beliefs follow the mean-field updates.

    Returns
    -------
    jnp.ndarray
        The descent gradient (the negative of the derivative above), shape of
        ``weights``, so that it composes with ``optax.apply_updates`` as the other
        learning kinds do.
    """
    n_parents = parent_state.mean.shape[-1]
    matrix = weights[:, :n_parents]
    activation = coupling_fn(parent_state.mean)
    prediction = jnp.matmul(matrix, activation)
    if parent_has_constant:
        prediction = prediction + weights[:, -1]
    conditional = child_state.conditional_expected_precision
    shrink = relaxed_effective_precision(
        child_state, child_is_input_layer, mean_field_updates
    )
    if child_is_input_layer or mean_field_updates:
        # The noise is the difference of the posterior means: the observations are
        # known exactly, and the mean-field posterior has no covariance between
        # layers.
        weighted_residual = conditional * (child_state.mean - prediction)
    else:
        # The child's noise given its parents' posterior means: the prediction error
        # it sent up, weighted by the relaxed gain, less s_a times the shift of its
        # prediction.
        evidence = jnp.maximum(
            child_state.precision - child_state.expected_precision, 0.0
        )
        gain = (
            conditional
            * child_state.precision
            / jnp.maximum(conditional + evidence, _EPS)
        )
        weighted_residual = gain * (
            child_state.mean - child_state.expected_mean
        ) - shrink * (prediction - child_state.expected_mean)

    slope = vmap(jgrad(coupling_fn))(parent_state.mean)
    parent_variance = slope**2 / jnp.maximum(parent_state.precision, _EPS)

    ascent = weighted_residual[:, None] * activation[None, :] - matrix * (
        shrink[:, None] * parent_variance[None, :]
    )
    if parent_has_constant:
        ascent = jnp.concatenate([ascent, weighted_residual[:, None]], axis=1)
    return -jnp.where(jnp.isfinite(ascent), ascent, 0.0)


# ------------------------------------------------------------------ kronecker

#: The learning kind of the Kronecker-factored weight belief
#: (see :func:`resolve_kronecker_settings`).
KRONECKER_KIND: str = "kronecker"

#: The gradients the Kronecker-factored weight belief can descend.
KRONECKER_GRADIENTS: tuple = (
    VARIATIONAL_KIND,
    "precision_weighted",
    "standard",
    BACKPROP_KIND,
)

#: Settings of the Kronecker-factored weight belief and their defaults, as
#: ``learning_kwargs`` accepts them (see :func:`resolve_kronecker_settings`).
KRONECKER_DEFAULTS: dict = {
    "learning_rate": 0.03,
    "memory": 100.0,
    "damping": 0.1,
    "absolute_damping": 0.0,
    "trust_region": 1e-3,
    "gradient": VARIATIONAL_KIND,
    "bias_correction": False,
    "anchor": 0.0,
    "anchor_prior": 0.0,
}


class KroneckerSettings(NamedTuple):
    r"""Validated settings of the Kronecker-factored weight belief.

    Built from a ``learning_kwargs`` dictionary by :func:`resolve_kronecker_settings`;
    every field is documented there.
    """

    learning_rate: float
    memory: float
    damping: float
    trust_region: float
    gradient: str = VARIATIONAL_KIND
    bias_correction: bool = False
    absolute_damping: float = 0.0
    anchor: float = 0.0
    anchor_prior: float = 0.0


def resolve_kronecker_settings(
    learning_kwargs: Optional[dict] = None,
) -> KroneckerSettings:
    r"""Validate the Kronecker-factored weight belief's settings and fill in defaults.

    Each weight matrix :math:`W`, predicting child layer :math:`a` from parent layer
    :math:`b`, carries a belief whose precision is the Kronecker product
    :math:`\Omega \otimes \Gamma` of two factors: the output factor :math:`\Omega`,
    shape ``(n_child, n_child)``, and the input factor :math:`\Gamma`, shape
    ``(n_parent[+1], n_parent[+1])``, the bias included. Both are running means, with
    memory :math:`N`, of what the update sweep leaves on each sample:

    .. math::

        \Omega &\leftarrow \Big(1 - \frac{1}{N}\Big) \Omega
            + \frac{1}{N} M_a, \\
        \Gamma &\leftarrow \Big(1 - \frac{1}{N}\Big) \Gamma
            + \frac{1}{N} \Big( \bar{g}\, \bar{g}^\top
            + \begin{pmatrix} G \Sigma_b G & 0 \\ 0 & 0 \end{pmatrix} \Big),

    with :math:`\bar{g} = (g(\mu_b), 1)` the coupled parent activation and the
    constant input, :math:`G = \operatorname{diag}(g'(\mu_b))`, and :math:`\Sigma_b =
    \operatorname{diag}(1 / \pi_b)` the posterior covariance of the parent layer.
    :math:`M_a = \operatorname{diag}(s_a)` is the precision the observations give the
    prediction of layer :math:`a`, with :math:`s_a` the relaxed effective precision
    (:func:`relaxed_effective_precision`). For the observed layer :math:`M_a` is the
    observation precision. The factors start at the identity, so the first steps are
    gradient steps at rate :math:`\eta / (1 + \lambda)^2`. With ``bias_correction``
    they start at zero instead and are kept as decayed sums, :math:`\Omega \leftarrow (1
    - 1/N)\, \Omega + M_a` and likewise for :math:`\Gamma`, beside a mass :math:`m
    \leftarrow (1 - 1/N)\, m + 1`, also started at zero. Each factor is divided by the
    mass before the step, so the step uses the weighted mean of the increments mixed in
    so far from the first one on. With :math:`N = \infty` the sums do not decay, and
    that mean is the plain mean of every increment since the first.

    The mean takes one step against the descent gradient :math:`\nabla` of the
    chosen ``gradient``, preconditioned by the damped factors:

    .. math::

        \tilde{\Omega} &= \Omega + \Big(\lambda\, \omega
            + \frac{\sqrt{\gamma}}{\rho}\Big) I,
        \qquad
        \tilde{\Gamma} = \Gamma + \big(\lambda\, g + \rho \sqrt{\gamma}\big) I,
        \qquad
        \omega = \frac{\operatorname{tr} \Omega}{n_a}, \quad
        g = \frac{\operatorname{tr} \Gamma}{n_b}, \quad
        \rho = \sqrt{g / \omega}, \\
        D &= \tilde{\Omega}^{-1} \nabla \tilde{\Gamma}^{-1}, \\
        W &\leftarrow W - c\, \eta\, D, \qquad
        c = \min\Big(1, \sqrt{\kappa / (\eta^2 q)}\Big), \quad
        q = \sum \langle D, \tilde{\Omega} D \tilde{\Gamma} \rangle,

    the sum running over every weight matrix of the network, so a single scale
    :math:`c` bounds the step :math:`\Delta w` of all of them together to
    :math:`\Delta w^\top (\tilde{\Omega} \otimes \tilde{\Gamma}) \Delta w \leq
    \kappa`.

    The relative damping :math:`\lambda` scales with the factors, so it vanishes when
    they do. The absolute damping :math:`\gamma` does not: the product of its two shifts
    is :math:`\gamma I`, so :math:`\tilde{\Omega} \otimes \tilde{\Gamma}` approximates
    the curvature plus a fixed precision :math:`\gamma`, with :math:`\rho` splitting it
    between the factors in proportion to their scales (factored Tikhonov damping, Section
    6.3 of [1]_). The output factor of a categorical observed layer vanishes as its
    predictions become confident, because :math:`\operatorname{diag}(p) - p p^\top \to
    0` as :math:`p` approaches a one-hot vector. The descent gradient vanishes at the
    same rate, so with :math:`\gamma = 0` the step on the logits keeps a constant size
    however well the observations are already predicted. With :math:`\gamma > 0` it
    shrinks to a gradient step of rate :math:`\eta / \gamma` once :math:`\omega g \ll
    \gamma`.

    With an ``anchor`` :math:`\alpha > 0`, the belief also keeps a slow mean
    :math:`\bar{W}`, the mean of a Gaussian belief about the weights whose precision
    is the decayed sum of every increment rather than their mean: the damped factors
    times their mass, :math:`m\, (\tilde{\Omega} \otimes \tilde{\Gamma})`. Each step
    combines that belief with the batch by the Gaussian update, the batch's gradient
    carried from the weights :math:`W` it was read at to :math:`\bar{W}` through the
    batch's own curvature :math:`M_a \otimes Q`, with :math:`Q` the input increment
    above:

    .. math::

        \bar{W} \leftarrow \bar{W} + \frac{1}{m + m_0}\, \tilde{\Omega}^{-1}
            \big( M_a (W - \bar{W}) Q - \nabla \big) \tilde{\Gamma}^{-1}.

    :math:`m_0` (``anchor_prior``) counts the initial weights, where the slow mean
    starts, as that many batches of evidence: with :math:`m_0 = 0` the first batch
    sets the slow mean to a full Newton step from its own curvature alone. The
    weights then take their step and are pulled toward the slow mean,
    :math:`W \leftarrow W' - \alpha\, (W' - \bar{W})` with :math:`W'` the stepped
    weights. The pull is the preconditioned step of a Gaussian prior on the weights
    with mean :math:`\bar{W}` and precision :math:`\beta\, (\Omega \otimes \Gamma)`:
    under the metric :math:`(1 + \beta)(\Omega \otimes \Gamma)` its Newton step is
    :math:`\alpha\, (W - \bar{W})` with :math:`\alpha = \beta / (1 + \beta)`, the
    Kronecker-factored Laplace prior of [4]_ held at the slow mean instead of at a
    snapshot. For a network of one linear layer, with :math:`\lambda \to 0` and
    :math:`m_0 = 0`, the slow mean is the least-squares solution of every batch since the first
    (recursive least squares), whichever path the weights took, since the weights
    cancel from the bracket. :math:`\alpha = 0` (default) keeps no slow mean and
    :math:`\alpha = 1` sets the weights to it after every step.

    The product :math:`\Omega \otimes \Gamma` is the Kronecker-factored approximation
    of the expected curvature of the log-likelihood with respect to :math:`W`, which
    replaces the mean of :math:`M_a \otimes \bar{g} \bar{g}^\top` by the product of the
    two means [1]_. With :math:`\eta = 1/N`, :math:`\lambda = 0` and no trust region,
    the step is the mean update of a Kalman filter on the weights under that
    approximation, the online natural gradient [2]_. The natural-gradient step of
    predictive coding in [3]_ preconditions by the running mean of
    :math:`\bar{g} \bar{g}^\top` alone, the case :math:`\Omega \propto I` and
    :math:`\Sigma_b = 0`.

    Parameters
    ----------
    learning_kwargs :
        The settings, as ``DeepNetwork.fit(learning_kwargs=...)`` takes them.
        Recognised keys, with the defaults of :data:`KRONECKER_DEFAULTS`:

        ``learning_rate``
            Step size :math:`\eta`. Positive.
        ``memory``
            Memory :math:`N` of both factors, in learning steps. At least 1, or
            ``float("inf")`` for factors that never forget, which requires
            ``bias_correction`` (factors started at the identity would never move).
        ``damping``
            Relative damping :math:`\lambda`, the fraction of the mean eigenvalue of
            each factor added to its diagonal before the solve. Positive.
        ``absolute_damping``
            Absolute damping :math:`\gamma`, the fixed precision added to the
            Kronecker product, split between the factors by :math:`\rho`. Zero (default)
            or positive.
        ``trust_region``
            Bound :math:`\kappa` on the step's squared length under the damped
            factors. Positive.
        ``gradient``
            The descent gradient the step preconditions, one of
            :data:`KRONECKER_GRADIENTS`: ``"variational"`` (default; see
            :func:`vectorised_variational_gradient`), ``"precision_weighted"`` or
            ``"standard"`` (see :func:`learning_weights_vectorised`), or
            ``"backprop"``, the gradient of the forward pass's squared error that
            ignores the inference (online learning only; see
            :func:`pyhgf.utils.vectorised_belief_propagation._backprop_gradients`).
        ``bias_correction``
            Whether the factors start at zero and are divided by their mass, rather
            than starting at the identity (default ``False``).
        ``anchor``
            Pull :math:`\alpha` of the weights toward the slow mean after each step,
            between 0 (default, no slow mean) and 1. Requires ``bias_correction``,
            because the slow mean's precision is the factors times their mass.
        ``anchor_prior``
            :math:`m_0`, the evidence the initial weights count for in the slow
            mean, in batches. Zero (default) or positive.

    Returns
    -------
    KroneckerSettings
        The validated settings.

    Raises
    ------
    ValueError
        If a key is unrecognised, ``memory`` is below 1 or infinite without
        ``bias_correction``, a positive quantity is not positive, ``gradient`` is
        not one of :data:`KRONECKER_GRADIENTS`, or ``anchor`` is outside 0 to 1 or
        positive without ``bias_correction``.

    References
    ----------
    .. [1] Martens, J., & Grosse, R. (2015). Optimizing neural networks with
       Kronecker-factored approximate curvature. Proceedings of the 32nd International
       Conference on Machine Learning, 2408 to 2417.
    .. [2] Ollivier, Y. (2018). Online natural gradient as a Kalman filter. Electronic
       Journal of Statistics, 12(2), 2930 to 2961. doi:10.1214/18-EJS1468
    .. [3] Beumer, L., de Haan, S., Pokel, N., Aceituno, P. V., & Grewe, B. F. (2026).
       Natural gradient learning in predictive coding through lateral circuits.
       arXiv:2611.16666
    .. [4] Ritter, H., Botev, A., & Barber, D. (2018). Online structured Laplace
       approximations for overcoming catastrophic forgetting. Advances in Neural
       Information Processing Systems 31, 3738 to 3748.
    """
    settings = dict(KRONECKER_DEFAULTS)
    given = dict(learning_kwargs or {})
    unknown = sorted(set(given) - set(KRONECKER_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown learning_kwargs for learning_kind='kronecker': {unknown}. "
            f"Expected a subset of {sorted(KRONECKER_DEFAULTS)}."
        )
    settings.update(given)
    learning_rate = float(settings["learning_rate"])
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    memory = float(settings["memory"])
    if not memory >= 1:
        raise ValueError(f"memory must be at least 1, got {memory}.")
    bias_correction = bool(settings["bias_correction"])
    if math.isinf(memory) and not bias_correction:
        raise ValueError(
            "An infinite memory requires bias_correction=True: factors that start at "
            "the identity would never move."
        )
    damping = float(settings["damping"])
    if damping <= 0:
        raise ValueError(f"damping must be positive, got {damping}.")
    absolute_damping = float(settings["absolute_damping"])
    if absolute_damping < 0:
        raise ValueError(
            f"absolute_damping must be zero or positive, got {absolute_damping}."
        )
    trust_region = float(settings["trust_region"])
    if trust_region <= 0:
        raise ValueError(f"trust_region must be positive, got {trust_region}.")
    gradient = str(settings["gradient"])
    if gradient not in KRONECKER_GRADIENTS:
        raise ValueError(
            f"gradient must be one of {KRONECKER_GRADIENTS}, got {gradient!r}."
        )
    anchor = float(settings["anchor"])
    if not 0.0 <= anchor <= 1.0:
        raise ValueError(f"anchor must be between 0 and 1, got {anchor}.")
    if anchor > 0 and not bias_correction:
        raise ValueError(
            "anchor requires bias_correction=True: the slow mean's precision is the "
            "factors times their mass."
        )
    anchor_prior = float(settings["anchor_prior"])
    if anchor_prior < 0:
        raise ValueError(f"anchor_prior must be zero or positive, got {anchor_prior}.")
    return KroneckerSettings(
        learning_rate=learning_rate,
        memory=memory,
        damping=damping,
        trust_region=trust_region,
        gradient=gradient,
        bias_correction=bias_correction,
        absolute_damping=absolute_damping,
        anchor=anchor,
        anchor_prior=anchor_prior,
    )


def kronecker_factor_update(
    factor: jnp.ndarray,
    increment: jnp.ndarray,
    memory: float,
    weight=1.0,
    summed: bool = False,
) -> jnp.ndarray:
    r"""Mix one sample's increment into a Kronecker factor.

    :math:`F \leftarrow (1 - w/N) F + (w/N) \tfrac{1}{2}(H + H^\top)`, with :math:`N`
    the ``memory`` (see :func:`resolve_kronecker_settings`) and :math:`w` the
    ``weight``. With ``summed`` (the bias-corrected factors) the increment enters with
    weight :math:`w` instead of :math:`w/N`, which keeps :math:`F` a decayed sum that
    is finite and moving for :math:`N = \infty`. An increment with a non-finite entry
    leaves the factor unchanged.

    Parameters
    ----------
    factor :
        The running factor, a square matrix.
    increment :
        This sample's increment, same shape.
    memory :
        The memory :math:`N`, in learning steps.
    weight :
        :math:`w`, between zero and one: zero leaves the factor unchanged (a batch
        that carries no data), one mixes the increment in at the full rate.
    summed :
        Whether the factor is a decayed sum rather than a running mean.

    Returns
    -------
    jnp.ndarray
        The advanced factor.
    """
    rate = weight / memory
    gain = weight if summed else rate
    mixed = (1.0 - rate) * factor + gain * 0.5 * (increment + increment.T)
    return jnp.where(jnp.all(jnp.isfinite(increment)), mixed, factor)


def kronecker_mass_update(mass, memory: float, weight=1.0):
    r"""Advance the mass of bias-corrected Kronecker factors.

    :math:`m \leftarrow (1 - w/N)\, m + w`, the total weight the factors have summed,
    so that a factor divided by it is the weighted mean of its increments (see
    :func:`resolve_kronecker_settings`).
    """
    return (1.0 - weight / memory) * mass + weight


def kronecker_direction(
    output_factor: jnp.ndarray,
    input_factor: jnp.ndarray,
    gradient: jnp.ndarray,
    damping: float,
    absolute_damping: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Precondition one weight matrix's gradient by its damped Kronecker factors.

    Returns :math:`D = \tilde{\Omega}^{-1} \nabla \tilde{\Gamma}^{-1}` and
    :math:`\langle D, \tilde{\Omega} D \tilde{\Gamma} \rangle`, with each factor damped
    by :math:`\lambda` times its mean eigenvalue plus its share of the absolute
    damping :math:`\gamma` (see :func:`resolve_kronecker_settings`). Both damped
    factors are symmetric positive definite, so the two solves use their Cholesky
    factorisations.

    Parameters
    ----------
    output_factor :
        :math:`\Omega`, shape ``(n_child, n_child)``.
    input_factor :
        :math:`\Gamma`, shape ``(n_in, n_in)``.
    gradient :
        :math:`\nabla`, shape ``(n_child, n_in)``.
    damping :
        :math:`\lambda`.
    absolute_damping :
        :math:`\gamma`. Zero (default) leaves the relative damping alone.

    Returns
    -------
    tuple
        ``(direction, quadratic)``: :math:`D`, shape of ``gradient``, and the scalar
        :math:`\langle D, \tilde{\Omega} D \tilde{\Gamma} \rangle`.
    """
    output_scale = jnp.maximum(jnp.trace(output_factor) / output_factor.shape[-1], _EPS)
    input_scale = jnp.maximum(jnp.trace(input_factor) / input_factor.shape[-1], _EPS)
    # Factored Tikhonov damping: the two shifts multiply to gamma, and their split
    # follows the ratio of the factors' scales.
    ratio = jnp.sqrt(input_scale / output_scale)
    root = jnp.sqrt(absolute_damping)

    def damped(factor, shift):
        return factor + shift * jnp.eye(factor.shape[-1], dtype=factor.dtype)

    output_damped = damped(output_factor, damping * output_scale + root / ratio)
    input_damped = damped(input_factor, damping * input_scale + root * ratio)
    left = jax.scipy.linalg.cho_solve(
        jax.scipy.linalg.cho_factor(output_damped), gradient
    )
    direction = jax.scipy.linalg.cho_solve(
        jax.scipy.linalg.cho_factor(input_damped), left.T
    ).T
    quadratic = jnp.sum(direction * (output_damped @ direction @ input_damped))
    return direction, quadratic


def kronecker_step_scale(
    quadratic: jnp.ndarray, settings: KroneckerSettings, trust_region_scale=1.0
):
    r"""Scale :math:`c` of the trust region shared by every weight matrix.

    :math:`c = \min(1, \sqrt{s \kappa / (\eta^2 q)})`, with :math:`q` the sum over the
    weight matrices of the quadratics of :func:`kronecker_direction` (see
    :func:`resolve_kronecker_settings`) and :math:`s` the ``trust_region_scale``, which
    multiplies the bound :math:`\kappa` for one step (default 1). A non-finite
    :math:`q`, or an :math:`s` that is non-finite or not positive, gives :math:`c = 0`,
    so no weight moves.
    """
    eta = settings.learning_rate
    budget = settings.trust_region * jnp.maximum(trust_region_scale, 0.0)
    bounded = jnp.sqrt(budget / jnp.maximum(eta**2 * quadratic, _EPS))
    scale = jnp.minimum(1.0, bounded)
    finite = jnp.isfinite(quadratic) & jnp.isfinite(budget)
    return jnp.where(finite, scale, 0.0)


def kronecker_slow_mean_update(
    slow_mean: jnp.ndarray,
    weights: jnp.ndarray,
    gradient: jnp.ndarray,
    output_increment: jnp.ndarray,
    input_increment: jnp.ndarray,
    output_factor: jnp.ndarray,
    input_factor: jnp.ndarray,
    mass,
    damping: float,
    absolute_damping: float = 0.0,
    weight=1.0,
    prior_mass: float = 0.0,
) -> jnp.ndarray:
    r"""Combine the slow mean of a Kronecker-factored weight belief with one batch.

    .. math::

        \bar{W} \leftarrow \bar{W} + \frac{w}{m + m_0}\, \tilde{\Omega}^{-1}
            \big( M_a (W - \bar{W}) Q - \nabla \big) \tilde{\Gamma}^{-1},

    with :math:`M_a` and :math:`Q` the batch's output and input increments
    (symmetrised as :func:`kronecker_factor_update` mixes them), :math:`\nabla` the
    descent gradient read at the weights :math:`W`, :math:`\tilde{\Omega}` and
    :math:`\tilde{\Gamma}` the damped factors of :func:`kronecker_direction`, and
    :math:`m` the mass of the factors after this batch was mixed in and :math:`m_0`
    the evidence the initial weights count for, in batches. The bracket is minus the
    batch's gradient at :math:`\bar{W}`, carried there from :math:`W` through the
    batch curvature :math:`M_a \otimes Q`, and
    :math:`(m + m_0)\, (\tilde{\Omega} \otimes \tilde{\Gamma})` is the precision of
    the slow belief, this batch included, so the update is the Gaussian combination
    of that belief with the batch (see
    :func:`resolve_kronecker_settings`). An increment or a step with a non-finite
    entry leaves the slow mean unchanged.

    Parameters
    ----------
    slow_mean :
        :math:`\bar{W}`, shape ``(n_child, n_in)``.
    weights :
        :math:`W`, the weights the gradient was read at, same shape.
    gradient :
        :math:`\nabla`, same shape.
    output_increment, input_increment :
        :math:`M_a`, shape ``(n_child, n_child)``, and :math:`Q`, shape
        ``(n_in, n_in)``.
    output_factor, input_factor :
        The factors after this batch, divided by their mass, undamped.
    mass :
        :math:`m`, the mass of the factors after this batch.
    damping, absolute_damping :
        :math:`\lambda` and :math:`\gamma` of :func:`kronecker_direction`.
    weight :
        :math:`w`, between zero and one, the weight the batch was mixed into the
        factors with: zero (a batch that carries no data) leaves the slow mean
        unchanged.
    prior_mass :
        :math:`m_0`, zero (default) or positive.

    Returns
    -------
    jnp.ndarray
        The advanced slow mean, shape of ``slow_mean``.
    """
    output_increment = 0.5 * (output_increment + output_increment.T)
    input_increment = 0.5 * (input_increment + input_increment.T)
    bracket = output_increment @ (weights - slow_mean) @ input_increment - gradient
    direction, _ = kronecker_direction(
        output_factor, input_factor, bracket, damping, absolute_damping
    )
    advanced = slow_mean + (weight / jnp.maximum(mass + prior_mass, _EPS)) * direction
    finite = (
        jnp.all(jnp.isfinite(output_increment))
        & jnp.all(jnp.isfinite(input_increment))
        & jnp.all(jnp.isfinite(direction))
        & (mass > 0)
    )
    return jnp.where(finite, advanced, slow_mean)


# ------------------------------------------------------ synaptic uncertainty

#: Settings of the weight-belief rule and their defaults, as
#: ``learning_kwargs`` accepts them (see :func:`resolve_synaptic_uncertainty_settings`).
SYNAPTIC_UNCERTAINTY_DEFAULTS: dict = {
    "window": None,
    "prior_variance": 1.0,
    "prior_mean": 0.0,
    "learning_rate": 1.0,
    "increment_scale": 1.0,
    "importance_source": "evidence",
    "natural_memory": 10.0,
    "natural_ridge": 1e-2,
    "gram_evidence": False,
}

#: The accepted values of the ``importance_source`` setting.
IMPORTANCE_SOURCES = ("evidence",)

#: Power iterations used for the largest eigenvalue of :func:`step_gram_surprise`.
_POWER_ITERATIONS = 30


class SynapticUncertaintySettings(NamedTuple):
    r"""Validated settings of the weight-belief rule.

    Built from a ``learning_kwargs`` dictionary by
    :func:`resolve_synaptic_uncertainty_settings`; every field is documented there.
    """

    window: float
    prior_variance: float
    prior_mean: float
    learning_rate: float
    increment_scale: float
    prior_precision: float
    importance_source: str = "evidence"
    natural_memory: float = 10.0
    natural_ridge: float = 1e-2
    gram_evidence: bool = False


def resolve_synaptic_uncertainty_settings(
    learning_kwargs: Optional[dict] = None,
) -> SynapticUncertaintySettings:
    r"""Validate the weight-belief rule's settings and fill in the defaults.

    Each weight carries a Gaussian belief: the weight itself is the belief's
    mean and the layer's ``weights_precision_delta`` its precision above the
    prior. One update step does two things.

    **Mean.** The update is :math:`\Delta w = -\alpha g/\pi + \text{reversion}`,
    the incoming descent gradient :math:`g` divided by the weight's precision
    and scaled by ``learning_rate`` :math:`\alpha`, plus a pull of the mean
    toward ``prior_mean``. At the start :math:`\pi = \pi_p` everywhere, so the
    rule begins as plain gradient descent at rate
    :math:`\alpha \times \texttt{prior\_variance}` and departs from it only as
    precision accumulates.

    **Precision.** The increment is the curvature the data impose on that
    weight, delivered as the importance factors of
    :func:`learning_weights_vectorised` or as a full increment matrix. It then
    relaxes toward the prior in precision form,
    :math:`\pi \leftarrow \pi + H - (\pi - \pi_p)/N` with
    :math:`N = \texttt{window}`. The fixed point is
    :math:`\pi^\ast = N \bar{H} + \pi_p`, linear in the evidence at every
    scale, and the mean reversion is precision-scaled,
    :math:`(\pi_p / (N\pi))(\mu_p - w)`, so a weight the data have pinned is
    also pulled back more gently.

    Both halves are mean-field: each weight's gradient is divided by its own
    precision, and the joint structure across the weights feeding one child is
    not retained. A layer can record that structure without using it (its input
    Gram matrices, see :class:`~pyhgf.typing.vectorised.Layer`), for a step a
    caller takes from the recorded fields.

    Parameters
    ----------
    learning_kwargs :
        The settings, as ``DeepNetwork.fit(learning_kwargs=...)`` takes them.
        Recognised keys, with the defaults of :data:`SYNAPTIC_UNCERTAINTY_DEFAULTS`:

        ``window``
            The memory window :math:`N`, in update steps. Importance decays
            toward the prior at rate :math:`1/N`, so curvature older than
            roughly :math:`N` steps no longer protects a weight. Required,
            and at least 1.
        ``prior_variance``
            Variance of the per-weight prior belief, :math:`1/\pi_p`. Also the
            effective learning rate before any importance has accumulated, so a
            value that trains the model well under plain gradient descent is
            the natural setting.
        ``prior_mean``
            Mean the weights revert toward.
        ``learning_rate``
            Multiplier :math:`\alpha` on the gradient part of the mean update.
            It exists because the rule otherwise has no step size of its own:
            the effective rate is :math:`\alpha/\pi`, and once the accumulated
            curvature dominates the prior the rate is pinned at
            :math:`\alpha/(N\bar{H})`. It sets the overall scale of every step
            but not the *ratio* between the rate before and after importance
            accumulates, which is :math:`1 + N\bar{H}/\pi_p` and is the depth
            of protection the rule applies. It does not scale the reversion,
            which stays at its :math:`1/N` rate, so ``window`` keeps meaning
            one thing only: how long importance is remembered.
        ``increment_scale``
            Multiplier on the importance increment. It deepens protection
            without touching anything else: the precision reaches
            :math:`\pi_p + c\,N\bar{H}`, so the most protected weights slow by
            :math:`c` times more, while the step size before any importance
            accumulates, the reversion rate and the window are unchanged. A
            uniform scale leaves rank orderings unchanged.
        ``importance_source``
            Where the child-side factor :math:`p` of the importance increment
            :math:`p \otimes h^2` comes from. The only accepted value is
            ``"evidence"`` (the default): the evidence precision carried up each
            network from its clamped layer (see :func:`learning_weights_vectorised`),
            so a network whose clamped layer is continuous seeds it with that
            layer's own precision, one number for every unit.
        ``natural_memory``
            Memory :math:`M`, in learning steps, of the running input Gram matrix
            ``weights_step_gram``: each step mixes in the current batch at weight
            :math:`1/M` (see :func:`vectorised_step_gram_update`), unless the layer
            carries its own ``weights_step_rate``. At least 1; at 1 the running
            matrix is the current batch's alone.
        ``natural_ridge``
            Ridge :math:`\rho` of :func:`step_gram_surprise`, relative to the mean
            eigenvalue of the running Gram matrix: :math:`\delta = \rho
            \operatorname{tr} \bar G / d`. Positive.
        ``gram_evidence``
            Whether each sample's outer product :math:`h_t h_t^\top` enters the
            recorded Gram matrices (``weights_input_gram``, ``weights_step_gram``
            and ``weights_batch_gram``) weighted by the evidence its child layer
            received, :math:`r_t = \frac{1}{n} \sum_a p_{a,t}`: the mean over the
            child's :math:`n` units of the child-side importance factor :math:`p`,
            the evidence precision of :func:`learning_weights_vectorised`. Within
            each batch the weights are rescaled so that the weighted Gram matrix
            keeps the trace of the unweighted one, so the evidence decides which
            samples' inputs shape the matrix and leaves its scale alone. Where the
            evidence is one constant across samples (a continuous clamped child)
            the matrix is unchanged. At a categorical child :math:`r_t = (1 -
            \lVert \mathbf{p}_t \rVert^2) / K`, with :math:`\mathbf{p}_t` the
            predicted probabilities over :math:`K` classes, so a sample the model
            already predicts confidently counts less. ``False`` (the default)
            weights every sample equally. A single-sample step is unaffected
            either way, since the rescaling cancels the weight of a lone sample.

        The gradient the rule descends is not among them: it is fixed to
        ``"synaptic_uncertainty"``.

    Returns
    -------
    SynapticUncertaintySettings
        The validated settings, with the prior precision precomputed.

    Raises
    ------
    ValueError
        If a key is unrecognised, ``window`` is missing or below 1, or a
        positive quantity is not positive.
    """
    settings = dict(SYNAPTIC_UNCERTAINTY_DEFAULTS)
    given = dict(learning_kwargs or {})
    unknown = sorted(set(given) - set(SYNAPTIC_UNCERTAINTY_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown learning_kwargs for learning_kind="
            f"'synaptic_uncertainty': {unknown}. "
            f"Expected a subset of {sorted(SYNAPTIC_UNCERTAINTY_DEFAULTS)}."
        )
    settings.update(given)

    if settings["window"] is None:
        raise ValueError(
            "learning_kind='synaptic_uncertainty' requires "
            "learning_kwargs={'window': N, ...}: "
            "the memory window has no default, since it sets how long "
            "importance is remembered."
        )
    window = float(settings["window"])
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}.")
    prior_variance = float(settings["prior_variance"])
    if prior_variance <= 0:
        raise ValueError(f"prior_variance must be positive, got {prior_variance}.")
    learning_rate = float(settings["learning_rate"])
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
    increment_scale = float(settings["increment_scale"])
    if increment_scale <= 0:
        raise ValueError(f"increment_scale must be positive, got {increment_scale}.")
    natural_memory = float(settings["natural_memory"])
    if natural_memory < 1:
        raise ValueError(f"natural_memory must be at least 1, got {natural_memory}.")
    natural_ridge = float(settings["natural_ridge"])
    if natural_ridge <= 0:
        raise ValueError(f"natural_ridge must be positive, got {natural_ridge}.")
    importance_source = str(settings["importance_source"])
    if importance_source not in IMPORTANCE_SOURCES:
        raise ValueError(
            f"importance_source must be one of {IMPORTANCE_SOURCES}, got "
            f"{importance_source!r}."
        )
    return SynapticUncertaintySettings(
        window=window,
        prior_variance=prior_variance,
        prior_mean=float(settings["prior_mean"]),
        learning_rate=learning_rate,
        increment_scale=increment_scale,
        prior_precision=1.0 / prior_variance,
        importance_source=importance_source,
        natural_memory=natural_memory,
        natural_ridge=natural_ridge,
        gram_evidence=bool(settings["gram_evidence"]),
    )


def clamped_layer_evidence(child_state: LayerState, child_kind: str) -> jnp.ndarray:
    r"""Seed the evidence walk at the layer the data are clamped to.

    A **binary or categorical** layer contributes the exact, label-free curvature
    of its own likelihood, :math:`\xi_a = \hat{p}_a(1 - \hat{p}_a)`: the diagonal
    of :math:`\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top`, which
    is the Hessian of the categorical log-likelihood with respect to the logits.
    No observed label enters it, so it is the model's own expected curvature
    rather than one built from the errors that happened to occur.

    A **continuous** layer contributes its own precision, which for a Gaussian
    likelihood is that same curvature: the second derivative of
    :math:`-\log \mathcal{N}(y; \mu, 1/\pi)` with respect to :math:`\mu` is
    :math:`\pi`, whatever the residual. Nothing is approximated. What differs is
    that the value is a *constant*, and two things follow from that. It varies
    neither across units nor across samples, so unlike the discrete seed it says
    nothing about which outputs the data constrain hardest, and every per-weight
    difference the walk produces above such a layer comes from the activations
    and the weight pullback alone. And wherever the precision expresses "this
    layer is observed" rather than a modelled observation noise, its value is
    arbitrary; since the seed multiplies through the whole chain, it then fixes
    the scale of every accumulated weight precision above it, which trades
    against the prior precision.

    Parameters
    ----------
    child_state :
        State of the clamped layer, after the update sweep.
    child_kind :
        ``"binary"``, ``"categorical"``, or anything else for a continuous layer.

    Returns
    -------
    jnp.ndarray
        The evidence precision at each node of the clamped layer.
    """
    if child_kind in ("binary", "categorical"):
        prob = child_state.expected_mean
        evidence = prob * (1.0 - prob)
    else:
        evidence = child_state.precision
    return jnp.where(jnp.isfinite(evidence), evidence, 0.0)


def evidence_pullback(
    parent_state: LayerState,
    child_evidence: jnp.ndarray,
    child_process_noise: jnp.ndarray,
    weights: jnp.ndarray,
    coupling_fn: Callable,
    parent_has_constant: bool = False,
) -> jnp.ndarray:
    r"""Carry the evidence precision up one layer, by the squared-coupling recursion.

    The message leaving a child first crosses that child's own process noise, which
    convolves the two variances and softens the evidence to :math:`\tilde\xi_a`.
    What is left is then pulled back through the linear map the child depends on the
    parent by: a Gaussian likelihood pulled back through a coefficient :math:`c` has
    its precision multiplied by :math:`c^2`, and here
    :math:`c = W_{ai}\,g'(\hat\mu_i)`. Children are conditionally independent given
    the parent, so their pulled-back precisions add:

    .. math::

        \xi_i = g'(\hat\mu_i)^2 \sum_a W_{ai}^2\, \tilde\xi_a,
        \qquad
        \tilde\xi_a = \left(\frac{1}{\xi_a} + \Omega_a\right)^{-1}

    The softening applies once per level crossed, so it is part of the recursion and
    not only of the increment the evidence eventually feeds. It is exactly one at a
    clamped layer, which has no process noise, so it changes nothing at the seed.

    This is the upward mirror of the downward variance bleed-through, and
    computationally it is curvature backpropagation: per sample,
    :math:`\tilde\xi_a h_i^2` is the Gauss-Newton diagonal of the global loss
    with respect to :math:`W_{ai}`, under the same diagonal treatment (cross
    terms between different parents of one child are dropped).

    Computed from the evidence carried in the walk rather than recovered as
    :math:`\pi_a - \hat\pi_a` from the filter's cache. The two agree in exact
    arithmetic, but the difference of two large nearly-equal precisions loses
    every significant digit once a layer's precision has accumulated, whereas a
    carried quantity does not.

    Parameters
    ----------
    parent_state :
        State of the layer the evidence is being carried up to.
    child_evidence :
        Evidence precision at each node of the layer below, shape
        ``(n_children,)``, before the child's own process noise is charged.
    child_process_noise :
        That layer's process noise :math:`\Omega_a`, same shape, recovered by its
        caller from the cached effective precision as
        :math:`\Omega_a = \gamma_a / \hat\pi_a`.
    weights :
        The matrix connecting them, shape ``(n_children, n_parents[+1])``. A bias
        column is dropped: a constant input node is not a parent whose belief the
        evidence can be about.
    coupling_fn :
        Coupling applied to parent means, differentiated at the parent's expected
        mean.
    parent_has_constant :
        Whether *weights* carries that trailing bias column.

    Returns
    -------
    jnp.ndarray
        The evidence precision at each node of the parent layer, shape
        ``(n_parents,)``.
    """
    if parent_has_constant:
        weights = weights[..., :-1]

    # Charge the child's process noise, xi_tilde = xi / (1 + Omega xi). Once per
    # level, so it compounds with depth, and exactly the identity at a clamped
    # layer, whose process noise is zero. The floor keeps a negative evidence,
    # which precision clipping elsewhere in the sweep can produce, from
    # amplifying the message instead of attenuating it.
    floored = jnp.maximum(child_evidence, 0.0)
    softened = floored / (1.0 + child_process_noise * floored)
    coupling_prime = vmap(jgrad(coupling_fn))(parent_state.expected_mean)
    pulled = jnp.matmul(weights.T**2, softened) * coupling_prime**2
    return jnp.where(jnp.isfinite(pulled), jnp.maximum(pulled, 0.0), 0.0)


def vectorised_synaptic_uncertainty_update(
    weights: jnp.ndarray,
    precision_delta: jnp.ndarray,
    gradient: jnp.ndarray,
    importance,
    settings: SynapticUncertaintySettings,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""One weight matrix's belief update: new mean, new accumulated precision.

    The rule is stated in :func:`resolve_synaptic_uncertainty_settings`. A
    ``LayerStack`` carries a leading slice axis on every operand, which the ellipsis
    broadcasting here handles unchanged.

    Parameters
    ----------
    weights :
        The belief means, i.e. the element's ``weights_mean``.
    precision_delta :
        The accumulated precision above the prior, same shape.
    gradient :
        The descent gradient, same shape.
    importance :
        Either the factor pair ``(p, q)`` of one sample, from
        :func:`learning_weights_vectorised`, whose outer product is the increment;
        or a full increment matrix of the weights' shape, the weighted batch mean of
        the per-sample outer products.
    settings :
        The resolved settings.

    Returns
    -------
    tuple of jnp.ndarray
        The updated weights and the updated accumulated precision.
    """
    prior_precision = settings.prior_precision
    pi = jnp.maximum(precision_delta + prior_precision, _EPS)

    # The mean update divides by the pre-update precision
    # and the reversion is precision-scaled.
    reversion = (prior_precision / (settings.window * pi)) * (
        settings.prior_mean - weights
    )
    update = -settings.learning_rate * gradient / pi + reversion

    if isinstance(importance, tuple):
        # Rank-one importance increment H[a, i] = p[a] * q[i].
        p, q = importance
        increment = p[..., :, None] * q[..., None, :]
    else:
        increment = importance
    increment = settings.increment_scale * increment
    new_delta = precision_delta + increment - precision_delta / settings.window

    return weights + update, new_delta


def create_input_gram(weights: jnp.ndarray) -> jnp.ndarray:
    r"""Initialise the input Gram matrix of one weight matrix at zero.

    The Gram matrix spans the parent side of ``weights``, so its shape is the last
    axis of ``weights`` on both sides, ``(n_parents[+1], n_parents[+1])``, and a bias
    column is included. Zero is the state before any evidence, matching a freshly
    installed ``weights_precision_delta``.

    Parameters
    ----------
    weights :
        The weight matrix the Gram matrix accompanies, ``(n_children,
        n_parents[+1])``.

    Returns
    -------
    jnp.ndarray
        Zeros of shape ``(n_parents[+1], n_parents[+1])``, stored as a layer's
        ``weights_input_gram``.
    """
    n_inputs = weights.shape[-1]
    return jnp.zeros((n_inputs, n_inputs), dtype=weights.dtype)


def vectorised_input_gram_update(
    gram: jnp.ndarray,
    increment: jnp.ndarray,
    settings: SynapticUncertaintySettings,
) -> jnp.ndarray:
    r"""Advance the input Gram matrix of one weight matrix by one learning step.

    .. math::

        G \leftarrow G + c\,\overline{h h^\top} - \frac{G}{N}

    where :math:`\overline{h h^\top}` is the weighted batch mean of the outer
    product of the coupled parent activations :math:`h`, :math:`c` is the
    increment scale and :math:`N` the window. This is the update
    :func:`vectorised_synaptic_uncertainty_update` applies to the accumulated
    precision, :math:`\Delta \leftarrow \Delta + c\,\overline{p \otimes h^2} -
    \Delta / N`, with the child-side factor :math:`p` left out and the parent
    side kept as a full matrix rather than its diagonal.

    Two consequences follow. Where the child-side factor is one constant
    :math:`\xi` across units and samples (a continuous clamped layer), every row
    :math:`a` of the accumulated precision is :math:`\Delta_{a,\cdot} = \xi \,
    \operatorname{diag}(G)`, so :math:`G` holds the off-diagonal terms that the
    per-weight precision discards. Where it varies, :math:`\bar{p}_a\, G` is the
    Kronecker-factored approximation of the precision of row :math:`a`.

    Parameters
    ----------
    gram :
        The accumulated Gram matrix, ``(n_parents[+1], n_parents[+1])``.
    increment :
        The weighted batch mean of :math:`h h^\top`, same shape, before the
        increment scale is applied.
    settings :
        The resolved settings of the weight-belief rule.

    Returns
    -------
    jnp.ndarray
        The advanced Gram matrix.
    """
    return gram + settings.increment_scale * increment - gram / settings.window


def create_step_gram(weights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Initialise the running input Gram matrix of one weight matrix at zero.

    Parameters
    ----------
    weights :
        The weight matrix the Gram matrix accompanies, ``(n_children,
        n_parents[+1])``.

    Returns
    -------
    tuple of jnp.ndarray
        Zeros of shape ``(n_parents[+1], n_parents[+1])``, stored as
        ``weights_step_gram``, and a zero scalar, stored as
        ``weights_step_mass``.
    """
    return create_input_gram(weights), jnp.zeros((), dtype=weights.dtype)


def vectorised_step_gram_update(
    gram: jnp.ndarray,
    mass: jnp.ndarray,
    batch_gram: jnp.ndarray,
    settings: SynapticUncertaintySettings,
    rate: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Mix one batch into the running input Gram matrix of one weight matrix.

    .. math::

        G \leftarrow (1 - \alpha) G + \alpha\, \overline{h h^\top},
        \qquad
        m \leftarrow (1 - \alpha) m + \alpha,

    with :math:`\alpha = 1 / M` and :math:`M` the ``natural_memory``, or
    :math:`\alpha` the layer's own ``weights_step_rate`` where one is given.
    Starting from :math:`G = 0` and :math:`m = 0`, :math:`G / m` is the
    weighted average of the batches seen, each batch weighted by the product
    of the :math:`1 - \alpha` factors of the steps after it, so its first value
    is the first batch's Gram matrix whatever :math:`\alpha` is.

    Parameters
    ----------
    gram, mass :
        ``weights_step_gram`` and ``weights_step_mass``.
    batch_gram :
        The weighted batch mean of :math:`h h^\top`, before any increment scale.
    settings :
        The resolved settings of the weight-belief rule.
    rate :
        The mixing rate :math:`\alpha` of this step, a scalar in :math:`(0, 1]`,
        or ``None`` (default) for :math:`1/M`.

    Returns
    -------
    tuple of jnp.ndarray
        The advanced Gram matrix and normaliser.
    """
    alpha = 1.0 / settings.natural_memory if rate is None else rate
    return (1.0 - alpha) * gram + alpha * batch_gram, (1.0 - alpha) * mass + alpha


def vectorised_step_rows_update(
    rows: jnp.ndarray,
    batch_rows: jnp.ndarray,
    settings: SynapticUncertaintySettings,
    rate: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    r"""Mix one batch's row evidence into the running row evidence.

    .. math::

        e_a \leftarrow (1 - \alpha)\, e_a + \alpha\, \sum_t \omega_t\, p_{a,t}
        \lVert h_t \rVert^2,

    with :math:`p_{a,t}` the evidence child unit :math:`a` received from sample
    :math:`t` (the child-side importance factor of
    :func:`learning_weights_vectorised`), :math:`h_t` the coupled parent
    activations, :math:`\omega_t` the normalised sample weights, and :math:`\alpha`
    the rate :func:`vectorised_step_gram_update` uses on the same step. The batch
    term is the row sum of the batch's importance increment :math:`p \otimes h^2`,
    taken over the edges only: in a one-to-one element row :math:`a` has the single
    input :math:`h_a`, and its term is :math:`\sum_t \omega_t\, p_{a,t} h_{a,t}^2`.

    Divided by its mean over rows, :math:`e` is the row scale :math:`s_a` of the
    Kronecker-factored curvature :math:`s_a G` of row :math:`a`, whose input side
    :math:`G` is ``weights_step_gram`` (trace matching: :math:`s_a \operatorname{tr}
    G` equals the row's own curvature trace up to one factor common to all rows).
    Where the evidence is one constant across units, every :math:`s_a` is 1.

    Parameters
    ----------
    rows :
        ``weights_step_rows``, shape ``(n_children,)``.
    batch_rows :
        The batch's row evidence, same shape.
    settings :
        The resolved settings of the weight-belief rule.
    rate :
        The mixing rate of this step, or ``None`` (default) for :math:`1/M`.

    Returns
    -------
    jnp.ndarray
        The advanced row evidence.
    """
    alpha = 1.0 / settings.natural_memory if rate is None else rate
    return (1.0 - alpha) * rows + alpha * batch_rows


def step_gram_surprise(
    gram: jnp.ndarray,
    mass: jnp.ndarray,
    batch_gram: jnp.ndarray,
    ridge: float,
) -> jnp.ndarray:
    r"""How far a batch's input Gram matrix lies outside the running one.

    .. math::

        s = \lambda_{max}\!\left((\bar G + \delta I)^{-1} G_b\right),
        \qquad
        \bar G = G / m,
        \qquad
        \delta = \rho\, \frac{\operatorname{tr} \bar G}{d},

    with :math:`G_b` the batch's Gram matrix and :math:`\bar G` the running
    average before this batch is mixed in. :math:`s` is the largest factor by
    which the batch drives an input direction more strongly than the recent
    batches did: about 1 when the batch resembles them, and large when it
    drives a direction they left near the ridge. It is computed by
    :data:`_POWER_ITERATIONS` steps of power iteration. Before any batch has been mixed in (:math:`m = 0`) there is no running
    average, and :math:`s = 1`.

    Parameters
    ----------
    gram, mass :
        ``weights_step_gram`` and ``weights_step_mass``, before this step's
        batch is mixed in.
    batch_gram :
        The weighted batch mean of :math:`h h^\top`.
    ridge :
        :math:`\rho`, relative to the mean eigenvalue of :math:`\bar G`.

    Returns
    -------
    jnp.ndarray
        The scalar :math:`s`.
    """
    average = _symmetrised(gram / jnp.maximum(mass, _EPS))
    d = average.shape[-1]
    delta = jnp.maximum(ridge * jnp.trace(average) / d, _EPS)
    factor = jax.scipy.linalg.cho_factor(
        average + delta * jnp.eye(d, dtype=average.dtype)
    )
    largest = _largest_relative_eigenvalue(factor, _symmetrised(batch_gram))
    return jnp.where(mass > 0, largest, jnp.ones_like(largest))


def _largest_relative_eigenvalue(factor, batch_gram: jnp.ndarray) -> jnp.ndarray:
    r"""Largest eigenvalue of :math:`A^{-1} G_b`, with ``factor`` the Cholesky of A.

    Found by :data:`_POWER_ITERATIONS` power iterations from the normalised vector of
    ones.
    """
    d = batch_gram.shape[-1]

    def power(_, vector):
        vector = jax.scipy.linalg.cho_solve(factor, batch_gram @ vector)
        return vector / jnp.maximum(jnp.linalg.norm(vector), _EPS)

    start = jnp.ones(d, dtype=batch_gram.dtype) / jnp.sqrt(d)
    vector = jax.lax.fori_loop(0, _POWER_ITERATIONS, power, start)
    return jnp.linalg.norm(jax.scipy.linalg.cho_solve(factor, batch_gram @ vector))


def _symmetrised(gram: jnp.ndarray) -> jnp.ndarray:
    """Average a Gram matrix with its transpose, removing float32 asymmetry."""
    return 0.5 * (gram + jnp.swapaxes(gram, -1, -2))
