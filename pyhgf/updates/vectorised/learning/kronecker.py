# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Kronecker-factored weight belief of the ``"kronecker"`` learning kind.

Each weight matrix carries a belief whose precision is the Kronecker product of an
output and an input factor. The mean steps along a descent gradient preconditioned by
the damped factors, under one trust region shared by every weight matrix. The rule and
its settings are stated in :func:`resolve_kronecker_settings`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from pyhgf.typing.vectorised import LayerStack, VectorisedNetwork
from pyhgf.updates.vectorised.elements import _EPS, _child_view, _sweeps_reach_top
from pyhgf.updates.vectorised.learning.backprop import BACKPROP_KIND
from pyhgf.updates.vectorised.learning.variational import (
    VARIATIONAL_KIND,
    relaxed_effective_precision,
)

#: The learning kind of the Kronecker-factored weight belief
#: (see :func:`resolve_kronecker_settings`).
KRONECKER_KIND: str = "kronecker"

#: The gradients the Kronecker-factored weight belief can descend.
KRONECKER_GRADIENTS: tuple = (VARIATIONAL_KIND, "precision_weighted", BACKPROP_KIND)

#: Settings of the Kronecker-factored weight belief and their defaults, as
#: ``learning_kwargs`` accepts them (see :func:`resolve_kronecker_settings`).
KRONECKER_DEFAULTS: dict = {
    "learning_rate": 0.03,
    "memory": 100.0,
    "damping": 0.1,
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
    gradient: str
    bias_correction: bool
    anchor: float
    anchor_prior: float


def resolve_kronecker_settings(
    learning_kwargs: Optional[dict] = None,
) -> KroneckerSettings:
    r"""Validate the Kronecker-factored weight belief's settings and fill in defaults.

    Each weight matrix :math:`W`, predicting child layer :math:`a` from parent layer
    :math:`b`, carries a belief whose precision, over the weights stacked row by row,
    is the Kronecker product :math:`\Omega \otimes \Gamma` of two factors: the output
    factor :math:`\Omega`, shape ``(n_child, n_child)``, and the input factor
    :math:`\Gamma`, shape ``(n_parent[+1], n_parent[+1])``, the bias included. Both are
    running means, with memory :math:`N`, of what the update sweep leaves on each
    sample:

    .. math::

        \Omega &\leftarrow \Big(1 - \frac{1}{N}\Big) \Omega
            + \frac{1}{N} M_a, \\
        \Gamma &\leftarrow \Big(1 - \frac{1}{N}\Big) \Gamma
            + \frac{1}{N} Q, \qquad
            Q = \bar{g}\, \bar{g}^\top
            + \begin{pmatrix} G \Sigma_b G & 0 \\ 0 & 0 \end{pmatrix},

    with :math:`\bar{g} = (g(\mu_b), 1)` the coupled parent activation and the
    constant input, :math:`G = \operatorname{diag}(g'(\mu_b))`, and :math:`\Sigma_b =
    \operatorname{diag}(1 / \pi_b)` the posterior covariance of the parent layer.
    :math:`M_a = \operatorname{diag}(s_a)` is the precision the observations give the
    prediction of layer :math:`a`, with :math:`s_a` the relaxed effective precision
    (:func:`~pyhgf.updates.vectorised.learning.variational.relaxed_effective_precision`).
    For the observed layer :math:`M_a` is the observation precision.

    The product :math:`\Omega \otimes \Gamma` approximates the expected curvature of
    the log-likelihood with respect to the weights as Eq. 1 of [1]_ does: the mean of
    the products :math:`M_a \otimes Q` is replaced by the product of the two means,
    each kept as an exponentially decaying average (section 5 of [1]_). [1]_ stacks
    the columns of :math:`W`, which writes the same matrix as
    :math:`\Gamma \otimes \Omega`. Its output factor is the second moment, under the
    model's predictive distribution, of the derivative of the log-likelihood with
    respect to the child's prediction. :math:`M_a` equals it at the clamped observed
    layer, and is read from the inference on a hidden layer.

    The factors start at the identity, so the first steps, while the identity still
    dominates the factors, are close to gradient steps at rate
    :math:`\eta / (1 + \lambda)^2`. With ``bias_correction``
    they start at zero instead and are kept as decayed sums, :math:`\Omega \leftarrow (1
    - 1/N)\, \Omega + M_a` and likewise for :math:`\Gamma`, beside a mass :math:`m
    \leftarrow (1 - 1/N)\, m + 1`, also started at zero. Each factor is divided by the
    mass before the step, so the step uses the weighted mean of the increments mixed in
    so far from the first one on. With :math:`N = \infty` the sums do not decay, and
    that mean is the plain mean of every increment since the first.

    The mean takes one step against the descent gradient :math:`\nabla` of the
    chosen ``gradient``, preconditioned by the damped factors:

    .. math::

        \tilde{\Omega} &= \Omega + \lambda\, \omega\, I,
        \qquad
        \tilde{\Gamma} = \Gamma + \lambda\, \nu\, I,
        \qquad
        \omega = \frac{\operatorname{tr} \Omega}{\dim \Omega}, \quad
        \nu = \frac{\operatorname{tr} \Gamma}{\dim \Gamma}, \\
        D &= \tilde{\Omega}^{-1} \nabla \tilde{\Gamma}^{-1}, \\
        W &\leftarrow W - c\, \eta\, D, \qquad
        c = \min\Big(1, \sqrt{\kappa / (\eta^2 q)}\Big), \quad
        q = \sum \langle D, \tilde{\Omega} D \tilde{\Gamma} \rangle,

    the sum running over every weight matrix of the network, so a single scale
    :math:`c` bounds the step :math:`\Delta w` of all of them together to
    :math:`\Delta w^\top (\tilde{\Omega} \otimes \tilde{\Gamma}) \Delta w \leq
    \kappa`.

    The two shifts are the factored Tikhonov damping of Eq. 7 of [1]_, with the trace
    choice of the scale between the two factors (section 6.3 of [1]_), at the damping
    strength :math:`\lambda^2 \omega \nu`. The damping therefore scales with the
    factors, so it vanishes when they do. The output factor of a categorical observed
    layer vanishes as its predictions become confident, because
    :math:`\operatorname{diag}(p) - p p^\top \to 0` as :math:`p` approaches a one-hot
    vector. The descent gradient vanishes at the same rate, so the step on the logits
    keeps a constant size however well the observations are already predicted.

    With an ``anchor`` :math:`\alpha > 0`, the belief also keeps a slow mean
    :math:`\bar{W}`, the mean of a Gaussian belief about the weights whose precision
    approximates the decayed sum of every increment, rather than their mean, by the
    damped factors times their mass, :math:`m\, (\tilde{\Omega} \otimes
    \tilde{\Gamma})`. Each step combines that belief with the batch by the Gaussian
    update, the batch's gradient carried from the weights :math:`W` it was read at to
    :math:`\bar{W}` through the batch's own curvature :math:`M_a \otimes Q`:

    .. math::

        \bar{W} \leftarrow \bar{W} + \frac{1}{m + m_0}\, \tilde{\Omega}^{-1}
            \big( M_a (W - \bar{W}) Q - \nabla \big) \tilde{\Gamma}^{-1},

    the update and projection steps of the online Laplace approximation (Eqs. 6 and 7
    of [2]_) taken at every batch instead of at the end of every task (see
    :func:`kronecker_slow_mean_update`). :math:`m_0` (``anchor_prior``) counts the
    initial weights, where the slow mean starts, as that many batches of evidence:
    with :math:`m_0 = 0` the first batch sets the slow mean to a full Newton step from
    its own curvature alone. The weights then take their step and are pulled toward
    the slow mean, :math:`W \leftarrow W' - \alpha\, (W' - \bar{W})` with :math:`W'`
    the stepped weights. The pull is the preconditioned step of a Gaussian prior on the
    weights with mean :math:`\bar{W}` and precision :math:`\beta\, (\Omega \otimes
    \Gamma)`: under the metric :math:`(1 + \beta)(\Omega \otimes \Gamma)` its Newton
    step is :math:`\alpha\, (W - \bar{W})` with :math:`\alpha = \beta / (1 + \beta)`.
    That prior is the quadratic penalty of the Kronecker-factored online Laplace
    approximation (Eqs. 5 and 9 of [2]_), centred on the slow mean instead of on the
    mode reached at the end of the last task, and with one Kronecker product as its
    precision instead of a sum of one per task (Eq. 7 of [2]_). For a network of one
    linear layer, with :math:`\lambda \to 0` and :math:`m_0 = 0`, the slow mean is the
    least-squares solution of every batch since the first (recursive least squares),
    whichever path the weights took, since the weights cancel from the bracket.
    :math:`\alpha = 0` (default) keeps no slow mean and :math:`\alpha = 1` sets the
    weights to it after every step.

    For a network with one weight matrix, between the clamped predictors and the
    observed layer, :math:`M_a \otimes Q` is the Fisher information of one sample.
    The step with :math:`\eta = 1/N`, :math:`\lambda = 0` and no trust region is then
    the online natural gradient of Definition 1 of [3]_, with the Kronecker-factored
    approximation in place of the Fisher matrix: its learning rate equals the decay
    rate of the curvature, and the curvature takes in the sample before the step, as
    the correspondence of [3]_ requires. By Proposition 3 of [3]_, it is the mean
    update of an extended Kalman filter on the weights whose memory of past samples
    fades by the factor :math:`1 - 1/N` at every step. The natural-gradient step of
    predictive coding (Eq. 6 of [4]_) preconditions the precision-weighted gradient by
    an exponential moving average of :math:`\bar{g} \bar{g}^\top` alone (Eqs. 5 and 7
    of [4]_), and the fixed precision that weights the gradient cancels from the step:
    the case :math:`\Sigma_b = 0` with :math:`\Omega` equal to that precision.

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
        ``trust_region``
            Bound :math:`\kappa` on the step's squared length under the damped
            factors. Positive.
        ``gradient``
            The descent gradient the step preconditions, one of
            :data:`KRONECKER_GRADIENTS`: ``"variational"`` (default; see
            :func:`~pyhgf.updates.vectorised.learning.variational.vectorised_variational_gradient`),
            ``"precision_weighted"`` (see
            :func:`~pyhgf.updates.vectorised.learning.factors.learning_weights_vectorised`),
            or ``"backprop"``, the gradient, by automatic differentiation, of the
            precision-weighted squared error of the network's forward pass, which
            ignores the inference (see
            :func:`~pyhgf.updates.vectorised.learning.backprop.vectorised_backprop_gradient`).
            ``"variational"`` and ``"backprop"`` are available in online learning
            only.
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
       Conference on Machine Learning, 2408 to 2417. Equation and section numbers
       refer to the extended version, arXiv:1503.05671.
    .. [2] Ritter, H., Botev, A., & Barber, D. (2018). Online structured Laplace
       approximations for overcoming catastrophic forgetting. Advances in Neural
       Information Processing Systems 31, 3738 to 3748. Equation numbers refer to
       arXiv:1805.07810.
    .. [3] Ollivier, Y. (2018). Online natural gradient as a Kalman filter. Electronic
       Journal of Statistics, 12(2), 2930 to 2961. doi:10.1214/18-EJS1468
    .. [4] Beumer, L., de Haan, S., Pokel, N., Vilimelis Aceituno, P., & Grewe, B. F.
       (2026). Natural gradient learning in predictive coding through lateral
       circuits. arXiv:2611.16666
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

    Parameters
    ----------
    mass :
        The scalar mass :math:`m` of the factors, ``weights_factor_mass``.
    memory :
        The memory :math:`N`, in learning steps.
    weight :
        :math:`w`, the weight the increments are mixed in with (see
        :func:`kronecker_factor_update`).

    Returns
    -------
    jnp.ndarray
        The advanced mass.
    """
    return (1.0 - weight / memory) * mass + weight


def kronecker_direction(
    output_factor: jnp.ndarray,
    input_factor: jnp.ndarray,
    gradient: jnp.ndarray,
    damping: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Precondition one weight matrix's gradient by its damped Kronecker factors.

    Returns :math:`D = \tilde{\Omega}^{-1} \nabla \tilde{\Gamma}^{-1}` and
    :math:`\langle D, \tilde{\Omega} D \tilde{\Gamma} \rangle`, with each factor damped
    by :math:`\lambda` times its mean eigenvalue (see
    :func:`resolve_kronecker_settings`). This is the factored Tikhonov damping of Eq. 7
    of [1]_, with the trace choice of the scale between the two factors, at the
    damping strength :math:`\lambda^2 \omega \nu`, with :math:`\omega` and :math:`\nu`
    the mean eigenvalues of :math:`\Omega` and :math:`\Gamma`. Both damped factors are
    symmetric positive definite, so the two solves use their Cholesky factorisations.

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

    Returns
    -------
    tuple
        ``(direction, quadratic)``: :math:`D`, shape of ``gradient``, and the scalar
        :math:`\langle D, \tilde{\Omega} D \tilde{\Gamma} \rangle`.

    References
    ----------
    .. [1] Martens, J., & Grosse, R. (2015). Optimizing neural networks with
       Kronecker-factored approximate curvature. Proceedings of the 32nd International
       Conference on Machine Learning, 2408 to 2417. Equation and section numbers
       refer to the extended version, arXiv:1503.05671.
    """
    output_scale = jnp.maximum(jnp.trace(output_factor) / output_factor.shape[-1], _EPS)
    input_scale = jnp.maximum(jnp.trace(input_factor) / input_factor.shape[-1], _EPS)

    def damped(factor, shift):
        return factor + shift * jnp.eye(factor.shape[-1], dtype=factor.dtype)

    output_damped = damped(output_factor, damping * output_scale)
    input_damped = damped(input_factor, damping * input_scale)
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

    Parameters
    ----------
    quadratic :
        :math:`q`, the scalar sum of the quadratics of every weight matrix.
    settings :
        The resolved settings, which give :math:`\eta` and :math:`\kappa`.
    trust_region_scale :
        :math:`s`, the factor on :math:`\kappa` for this step.

    Returns
    -------
    jnp.ndarray
        The scalar :math:`c`, between zero and one.
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
    :func:`resolve_kronecker_settings`). These are the update and projection steps of
    the online Laplace approximation (Eqs. 6 and 7 of [1]_) applied to every batch:
    one Newton step on the batch's quadratic model stands in for the mode of Eq. 6,
    and the precision of Eq. 7 is decayed by :math:`1 - 1/N` and kept in the
    Kronecker-factored form of Eq. 9 of [1]_. An increment or a step with a
    non-finite entry leaves the slow mean unchanged.

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
    damping :
        :math:`\lambda` of :func:`kronecker_direction`.
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

    References
    ----------
    .. [1] Ritter, H., Botev, A., & Barber, D. (2018). Online structured Laplace
       approximations for overcoming catastrophic forgetting. Advances in Neural
       Information Processing Systems 31, 3738 to 3748. Equation numbers refer to
       arXiv:1805.07810.
    """
    output_increment = 0.5 * (output_increment + output_increment.T)
    input_increment = 0.5 * (input_increment + input_increment.T)
    bracket = output_increment @ (weights - slow_mean) @ input_increment - gradient
    direction, _ = kronecker_direction(output_factor, input_factor, bracket, damping)
    advanced = slow_mean + (weight / jnp.maximum(mass + prior_mass, _EPS)) * direction
    finite = (
        jnp.all(jnp.isfinite(output_increment))
        & jnp.all(jnp.isfinite(input_increment))
        & jnp.all(jnp.isfinite(direction))
        & (mass > 0)
    )
    return jnp.where(finite, advanced, slow_mean)


def kronecker_increment_vectors(network: VectorisedNetwork) -> tuple:
    r"""Return the vectors of one sample's Kronecker increments, per element.

    Must run after the update sweep. For the weights of element :math:`i`, predicting
    child :math:`a` (element :math:`i - 1`) from parent :math:`b` (element :math:`i`),
    returns ``(d, p, h, v)`` such that the output increment :math:`M_a` of
    :func:`resolve_kronecker_settings` is :math:`\operatorname{diag}(d) - p p^\top` and
    the input increment is :math:`h h^\top + \operatorname{diag}(v)`:

    - a volatile child contributes its relaxed effective precision
      (:func:`~pyhgf.updates.vectorised.learning.variational.relaxed_effective_precision`),
      which is the observation precision at the clamped observed layer, and ``p`` is
      ``None``;
    - a clamped categorical child contributes the Fisher information of its
      likelihood with respect to the logits, :math:`\operatorname{diag}(\mathbf{p}) -
      \mathbf{p}\mathbf{p}^\top`, with :math:`\mathbf{p}` its predicted probabilities:
      the expectation under the model's predictive distribution that section 5 of
      [1]_ estimates by sampling;
    - a clamped binary child contributes :math:`\hat{p}(1 - \hat{p})` and ``p`` is
      ``None``.

    :math:`h = g(\mu_b)` is the coupled parent activation at the posterior means and
    :math:`v = g'(\mu_b)^2 / \pi_b` the posterior variance it carries, both padded
    with the constant input (variance zero) when the weights carry a bias. The top
    element is clamped to the predictors, so its variance is zero unless the sweeps
    update it (``update_input_layer`` with a volatile top layer). Non-finite entries
    are set to zero.

    Parameters
    ----------
    network :
        The network after the update sweep on one sample.

    Returns
    -------
    tuple
        One ``(d, p, h, v)`` per element, ``None`` for the bottom element. Stacked
        over a batch, they are the ``vectors`` of :func:`vectorised_kronecker_update`.

    Raises
    ------
    NotImplementedError
        If an element is a ``LayerStack``.

    References
    ----------
    .. [1] Martens, J., & Grosse, R. (2015). Optimizing neural networks with
       Kronecker-factored approximate curvature. Proceedings of the 32nd International
       Conference on Machine Learning, 2408 to 2417. Equation and section numbers
       refer to the extended version, arXiv:1503.05671.
    """
    elements = network.layers
    if any(isinstance(element, LayerStack) for element in elements):
        raise NotImplementedError(
            "learning_kind='kronecker' does not support layer stacks; add the layers "
            "one by one with add_layer."
        )
    top = len(elements) - 1
    top_known = not _sweeps_reach_top(network)

    def finite(a):
        return None if a is None else jnp.where(jnp.isfinite(a), a, 0.0)

    vectors: list = [None]
    for i in range(1, len(elements)):
        parent = elements[i]
        child_state, child_kind, child_is_input_layer = _child_view(elements[i - 1])
        if child_kind == "categorical":
            d = p = child_state.expected_mean
        elif child_kind == "binary":
            prob = child_state.expected_mean
            d, p = prob * (1.0 - prob), None
        else:
            d = relaxed_effective_precision(
                child_state, child_is_input_layer, network.mean_field_updates
            )
            p = None
        mean = parent.state.mean
        h = parent.coupling_fn(mean)
        if i == top and top_known:
            v = jnp.zeros_like(h)
        else:
            slope = jax.vmap(jax.grad(parent.coupling_fn))(mean)
            v = slope**2 / jnp.maximum(parent.state.precision, _EPS)
        if parent.add_constant_input:
            h = jnp.append(h, 1.0)
            v = jnp.append(v, 0.0)
        vectors.append((finite(d), finite(p), finite(h), finite(v)))
    return tuple(vectors)


def vectorised_kronecker_update(
    network: VectorisedNetwork,
    grads: tuple,
    settings: KroneckerSettings,
    vectors: tuple,
    sample_weights: jnp.ndarray,
    weight_reuse: float = 1.0,
    trust_region_scale=1.0,
) -> VectorisedNetwork:
    r"""Advance every Kronecker-factored weight belief of a network one step.

    For each element holding weights, mixes the batch's increments into
    ``weights_output_factor`` and ``weights_input_factor``
    (:func:`kronecker_factor_update`), preconditions the element's gradient by the
    damped factors (:func:`kronecker_direction`), and moves every ``weights_mean`` by
    its preconditioned gradient under one trust region shared by all elements
    (:func:`kronecker_step_scale`), as :func:`resolve_kronecker_settings` states. The
    increments are the weighted sums over the batch

    .. math::

        M_a = k \sum_t \omega_t \big( \operatorname{diag}(d_t) - p_t p_t^\top \big),
        \qquad
        Q = \sum_t \omega_t \big( h_t h_t^\top + \operatorname{diag}(v_t) \big),

    with :math:`\omega` the ``sample_weights`` and :math:`k` the ``weight_reuse``,
    formed without any per-sample matrix. They are mixed in with weight one, or with
    weight zero when every sample weight is zero, which leaves the factors and their
    mass unchanged.

    With bias-corrected factors (an element carrying ``weights_factor_mass``) the
    stored factors are decayed sums, and the step divides them by the advanced mass
    first (:func:`kronecker_mass_update`). With an ``anchor`` above zero, each
    element's ``weights_anchor_mean`` is combined with the batch
    (:func:`kronecker_slow_mean_update`) and the stepped weights are pulled toward it
    by the fraction ``anchor``. The rule needs no optimiser.

    Parameters
    ----------
    network :
        The network after the update sweep.
    grads :
        One descent gradient per element, ``None`` for the bottom element.
    settings :
        The resolved settings of the rule.
    vectors :
        One ``(d, p, h, v)`` per element, as :func:`kronecker_increment_vectors`
        returns them, each with a leading batch axis. A single sample is a batch of
        one.
    sample_weights :
        :math:`\omega`, shape ``(batch,)``, summing to one, or all zero for a batch
        that carries no data.
    weight_reuse :
        :math:`k`, how many times each weight matrix is applied per sample. It
        multiplies the output increment, as it multiplies the gradient. Default
        ``1.0``.
    trust_region_scale :
        Multiplies the trust region :math:`\kappa` for this step (see
        :func:`kronecker_step_scale`). Default ``1.0``.

    Returns
    -------
    VectorisedNetwork
        The network with updated weights and factors.

    Raises
    ------
    ValueError
        If an element holding weights carries no factors, or no slow mean under an
        ``anchor`` above zero. :meth:`~pyhgf.model.DeepNetwork.install_kronecker_belief`
        installs both.
    NotImplementedError
        If an element holding weights is one-to-one (``fully_connected=False``).
    """
    output_weights = sample_weights * weight_reuse
    mix = (jnp.sum(sample_weights) > 0).astype(sample_weights.dtype)
    elements = list(network.layers)
    advanced = {}
    quadratic = jnp.zeros(())
    for i, elem in enumerate(elements):
        if elem.weights_mean is None or grads[i] is None:
            continue
        if elem.weights_output_factor is None or elem.weights_input_factor is None:
            raise ValueError(
                f"layers[{i}] holds weights but no Kronecker factors. Install them "
                "with install_kronecker_belief(), or let learning_kind='kronecker' "
                "install them."
            )
        if not elem.fully_connected:
            raise NotImplementedError(
                "learning_kind='kronecker' requires fully connected weights."
            )
        d, p, h, v = vectors[i]
        output_increment = jnp.diag(output_weights @ d)
        if p is not None:
            output_increment = output_increment - jnp.einsum(
                "b,bi,bj->ij", output_weights, p, p
            )
        input_increment = jnp.einsum("b,bi,bj->ij", sample_weights, h, h) + jnp.diag(
            sample_weights @ v
        )
        mass = elem.weights_factor_mass
        # Bias-corrected factors are decayed sums, divided by their mass below.
        output_factor = kronecker_factor_update(
            elem.weights_output_factor,
            output_increment,
            settings.memory,
            mix,
            summed=mass is not None,
        )
        input_factor = kronecker_factor_update(
            elem.weights_input_factor,
            input_increment,
            settings.memory,
            mix,
            summed=mass is not None,
        )
        if mass is None:
            output_mean, input_mean = output_factor, input_factor
        else:
            mass = kronecker_mass_update(mass, settings.memory, mix)
            scale = 1.0 / jnp.maximum(mass, _EPS)
            output_mean = jnp.where(
                mass > 0, output_factor * scale, jnp.eye(output_factor.shape[0])
            )
            input_mean = jnp.where(
                mass > 0, input_factor * scale, jnp.eye(input_factor.shape[0])
            )
        direction, matrix_quadratic = kronecker_direction(
            output_mean, input_mean, grads[i], settings.damping
        )
        slow = None
        if settings.anchor > 0:
            if elem.weights_anchor_mean is None or mass is None:
                raise ValueError(
                    f"layers[{i}] has no slow mean for anchor={settings.anchor}. "
                    "Install it with install_kronecker_belief(bias_correction=True, "
                    "anchor=True), or let learning_kind='kronecker' install it."
                )
            slow = kronecker_slow_mean_update(
                elem.weights_anchor_mean,
                elem.weights_mean,
                grads[i],
                output_increment,
                input_increment,
                output_mean,
                input_mean,
                mass,
                settings.damping,
                mix,
                settings.anchor_prior,
            )
        advanced[i] = (output_factor, input_factor, mass, direction, slow)
        quadratic = quadratic + matrix_quadratic

    step = (
        kronecker_step_scale(quadratic, settings, trust_region_scale)
        * settings.learning_rate
    )
    for i, (output_factor, input_factor, mass, direction, slow) in advanced.items():
        direction = jnp.where(jnp.isfinite(direction), direction, 0.0)
        fields = {} if mass is None else {"weights_factor_mass": mass}
        weights = elements[i].weights_mean - step * direction
        if slow is not None:
            weights = weights - settings.anchor * (weights - slow)
            fields["weights_anchor_mean"] = slow
        elements[i] = dataclasses.replace(
            elements[i],
            weights_mean=weights,
            weights_output_factor=output_factor,
            weights_input_factor=input_factor,
            **fields,
        )
    return dataclasses.replace(network, layers=tuple(elements))
