# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Mixed pipelines: declaring models that mix PyHGF networks and fixed calculations.

A pipeline is a tree of **parts**. Each part either learns — a
:class:`DeepNetworkAdapter` wrapping a PyHGF :class:`~pyhgf.model.DeepNetwork`, which
learns locally through PyHGF's belief updates — or is frozen — an
:class:`EquinoxAdapter` wrapping a fixed calculation (an activation function, a
normalisation) that never learns and only translates errors with a hand-derived formula.
:class:`PCSequential` chains parts; :class:`Residual` declares the "add back the input"
shortcut.

The classes here *declare* the model: which slots learn, what configuration each
learning part uses, which formulas the frozen parts apply. Execution is the job of
:class:`~pyhgf.model.fused.FusedPipeline`, which stages the whole tree — forward walk,
error at the output, every part's local learning step — into one compiled program per
training step. Walked forward, the tree predicts; walked backward, each part learns from
the error at its output and hands the error at its *input* to the part behind it.
Nothing resembling backpropagation runs anywhere: no global computation graph, no
automatic differentiation.

Error convention: the arrays passed between parts are *descent* errors — the gradient of
the loss with respect to that signal, the same object a backprop library would hand
around. PyHGF's internal prediction errors follow the opposite, observed-minus-predicted
convention; the executor converts between the two at the learning part's boundary, in
exactly one place.

All parts operate on batches: arrays of shape ``(n_samples, n_features)``, one row per
sample (e.g. one row per token position, flattened across sequences).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from pyhgf.model.deep_network import DeepNetwork
from pyhgf.model.error_types import DescentError, ObservedMinusPredicted

__all__ = [
    "PCModule",
    "EquinoxAdapter",
    "DeepNetworkAdapter",
    "PCSequential",
    "Residual",
    "gelu_adapter",
    "layer_norm_adapter",
    "linear_adapter",
]


class PCModule:
    """Base class for all parts in a mixed pipeline.

    A part contributes two core responsibilities to a mixed training pipeline:

    1. **Declaration:** The part object (this class and its subclasses) declares
       which computations learn and which are frozen, along with their
       configuration (optimizer, layer sizes, activation functions, etc.).

    2. **State management:** Each part declares its state structure via
       :meth:`init_state`, which returns the state pytree this part holds
       (network beliefs, optimizer moments, etc.). Execution is delegated to
       :class:`~pyhgf.model.fused.FusedPipeline`.

    All subclasses **must** implement:

    - ``__init__(...)``: Store part configuration (layer sizes, optimizer, etc.)
    - ``init_state()``: Return this part's state pytree

    The executor (:class:`~pyhgf.model.fused.FusedPipeline`) calls ``init_state()``
    to initialize the part's state, threads it through forward/backward walks in
    one compiled JAX program, and writes it back via a ``merge()`` closure.

    Notes
    -----
    Subclasses that do not define both ``__init__`` and ``init_state`` raise a
    :class:`TypeError` at class definition time. See :class:`DeepNetworkAdapter`,
    :class:`EquinoxAdapter`, :class:`PCSequential`, and :class:`Residual` for examples.
    """

    def __init_subclass__(cls, **kwargs):
        """Require concrete parts to define ``__init__`` and ``init_state``.

        Checked at class definition time so an incomplete part fails early with a
        clear message rather than cryptically during training.

        Raises
        ------
        TypeError
            If the subclass defines neither ``__init__`` nor ``init_state``.
        """
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must define __init__() to declare part configuration."
            )
        # init_state must be overridden somewhere below PCModule's stub.
        if not any(
            "init_state" in base.__dict__
            for base in cls.__mro__
            if base is not PCModule
        ):
            raise TypeError(
                f"{cls.__name__} must define init_state() returning its state pytree."
            )

    def init_state(self) -> Any:
        """Initialize and return this part's state pytree.

        Called by :class:`~pyhgf.model.fused.FusedPipeline` during setup to
        build the full state pytree that is threaded through every training step:

        - **Frozen parts** (:class:`EquinoxAdapter`): the empty tuple ``()``.
        - **Learning parts** (:class:`DeepNetworkAdapter`): ``(network, opt_state)``.
        - **Composite parts** (:class:`PCSequential`, :class:`Residual`, …): the
          nested tuple of their children's states.

        Raises
        ------
        NotImplementedError
            If called on a subclass that does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.init_state() must return its state pytree. "
            f"See {PCModule.__name__} docstring for the per-part conventions."
        )


class EquinoxAdapter(PCModule):
    """A frozen part: a fixed calculation that routes errors but never learns.

    Declares a forward function and its hand-derived backward companion — no
    automatic differentiation is involved at any point:

    - ``forward_fn(x) -> (y, cache)`` computes the output for a batch and
      returns whatever the backward formula needs;
    - ``backward_fn(cache, error) -> error_in`` translates the error at the
      output into the error at the input, using only the cache.

    Error Convention
    ----------------
    Both forward_fn and backward_fn use the **descent-error convention**
    (see :mod:`pyhgf.model.error_types`):

    - forward_fn receives arrays in the pipeline's usual format
    - backward_fn receives :class:`~pyhgf.model.error_types.DescentError`
      (positive = signal too high) and returns the same convention

    The hand-derived backward formula must respect this convention:
    if the function is ``y = f(x)`` and loss is ``L(y)``, then
    ``backward_fn`` should return ``∂L/∂x = (∂L/∂y) @ (∂y/∂x)^T``.

    Use the ready-made constructors :func:`gelu_adapter` and
    :func:`layer_norm_adapter` for the standard Transformer pieces.
    """

    def __init__(
        self,
        forward_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, tuple]],
        backward_fn: Callable[[tuple, DescentError], DescentError],
    ):
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def init_state(self) -> tuple:
        """Return the empty state pytree (frozen parts have no state)."""
        return ()


# Constant of the tanh approximation of GELU, sqrt(2 / pi). The formulas
# below must match ``jax.nn.gelu`` with ``approximate=True`` (its default).
_GELU_C = 0.7978845608028654
_GELU_A = 0.044715


def _gelu_forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
    return jax.nn.gelu(x), (x,)


def _gelu_backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
    # Derivative of the tanh approximation
    #   gelu(x) = 0.5 x (1 + tanh(c (x + a x^3))),
    #   gelu'(x) = 0.5 (1 + t) + 0.5 x (1 - t^2) c (1 + 3 a x^2),  t = tanh(...).
    (x,) = cache
    t = jnp.tanh(_GELU_C * (x + _GELU_A * x**3))
    slope = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * _GELU_C * (
        1.0 + 3.0 * _GELU_A * x**2
    )
    return error * slope


def gelu_adapter() -> EquinoxAdapter:
    """Build a frozen GELU: the error is multiplied by the slope at the cached input."""
    return EquinoxAdapter(_gelu_forward, _gelu_backward)


def layer_norm_adapter(layer_norm: eqx.nn.LayerNorm) -> EquinoxAdapter:
    """Build a frozen LayerNorm around an ``eqx.nn.LayerNorm``'s parameters.

    Forward, each row is centred, divided by its spread, then scaled and shifted by the
    (frozen) ``weight``/``bias``. Backward, the error is scaled by ``weight``, then the
    centring and rescaling are undone: the error's own row-average and its overlap with
    the normalised input are subtracted before dividing by the spread — the two
    subtractions account for the fact that shifting or stretching a whole row leaves its
    normalisation unchanged.
    """
    gamma = layer_norm.weight if layer_norm.weight is not None else 1.0
    beta = layer_norm.bias if layer_norm.bias is not None else 0.0
    eps = layer_norm.eps

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        mean = x.mean(axis=-1, keepdims=True)
        std = jnp.sqrt(x.var(axis=-1, keepdims=True) + eps)
        normed = (x - mean) / std
        return gamma * normed + beta, (normed, std)

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        normed, std = cache
        scaled = error * gamma
        return (
            scaled
            - scaled.mean(axis=-1, keepdims=True)
            - normed * (scaled * normed).mean(axis=-1, keepdims=True)
        ) / std

    return EquinoxAdapter(forward, backward)


def linear_adapter(linear: eqx.nn.Linear) -> EquinoxAdapter:
    """Build a frozen linear map around an ``eqx.nn.Linear``'s parameters.

    Forward, ``x @ weight.T (+ bias)`` on the last axis. Backward, the error is
    multiplied back through the weight matrix (``error @ weight``) — the exact input
    error of a linear map, no derivation needed. The bias, being added, passes the error
    through untouched.
    """
    weight = jnp.asarray(linear.weight)
    bias = None if linear.bias is None else jnp.asarray(linear.bias)

    def forward(x: jnp.ndarray) -> tuple[jnp.ndarray, tuple]:
        y = x @ weight.T
        if bias is not None:
            y = y + bias
        return y, ()

    def backward(cache: tuple, error: jnp.ndarray) -> jnp.ndarray:
        return error @ weight

    return EquinoxAdapter(forward, backward)


class DeepNetworkAdapter(PCModule):
    """A learning part: a PyHGF :class:`~pyhgf.model.DeepNetwork` in the pipeline.

    Declares the wrapped network and how it learns; the executor runs one
    batch-synchronous local learning step per training step (the same
    computation as :meth:`~pyhgf.model.DeepNetwork.batch_update`, staged
    in-trace) and threads the error at the network's input onward.

    Error Convention Bridge
    -----------------------
    This part is where the pipeline's **descent-error convention** meets
    **PyHGF's observed-minus-predicted convention** (see :mod:`pyhgf.model.error_types`).
    The executor performs the conversion at this single boundary:

    **Forward (pipeline → PyHGF):**
        Input (DescentError) is unchanged; used as usual.

    **Backward (PyHGF → pipeline):**
        1. Pipeline passes :class:`~pyhgf.model.error_types.DescentError` (positive = too high)
        2. This part converts to :class:`~pyhgf.model.error_types.ObservedMinusPredicted`:
           ``observation = output - descent_error``
        3. PyHGF's ``batch_update`` treats this as the target, computing
           ``prediction_error = observation - output = -descent_error``
        4. Learning uses the prediction error natively
        5. Input error (also prediction-error convention) is negated back to
           descent-error convention before passing to the previous part

    This two-step conversion (descent↔observed-minus-predicted) happens
    **in exactly one place**: inside the executor's backward pass for this
    adapter. No other part needs to know about the convention flip.

    Parameters
    ----------
    net :
        The wrapped network. Its top (input) layer width is the part's input
        size; its bottom (output) layer width is the part's output size.
    optimizer :
        Optax optimiser for the local weight step. ``None`` freezes the
        weights (the beliefs still update).
    learning_kind :
        Weight-gradient mode, as in :meth:`~pyhgf.model.DeepNetwork.fit`.
    update_confidences :
        Whether the confidence state adapts across batches (see
        :meth:`~pyhgf.model.DeepNetwork.batch_update`). Defaults to False —
        the setting used for exact comparisons against backpropagation.
    time_step :
        Inference time step — scales the confidence leak per batch (one batch
        counts as one observation of duration ``time_step``).
    """

    def __init__(
        self,
        net: DeepNetwork,
        optimizer: Optional[optax.GradientTransformation] = None,
        learning_kind: str = "precision_weighted",
        update_confidences: bool = False,
        time_step: float = 1.0,
    ):
        self.net = net
        self.optimizer = optimizer
        self.learning_kind = learning_kind
        self.update_confidences = update_confidences
        self.time_step = time_step

    def init_state(self) -> tuple:
        """Return the ``(network, opt_state)`` state pytree.

        ``opt_state`` is None when ``optimizer`` is None (weights frozen, beliefs
        still update); otherwise it is initialized here if not already set.

        Raises
        ------
        ValueError
            If the network has no layers yet (call ``net.add_layer(...)`` first).
        """
        if self.net.state is None:
            raise ValueError(
                "Network has no layers yet. Call add_layer(...) before init_state()."
            )

        opt_state = self.net.opt_state
        if self.optimizer is not None and opt_state is None:
            opt_state = self.optimizer.init(self.net.state.weights_tuple())

        return (self.net.state, opt_state)


class PCSequential(PCModule):
    """A chain of parts: forward in order to predict, in reverse to update."""

    def __init__(self, parts: list):
        self.parts = list(parts)

    def init_state(self) -> tuple:
        """Return a tuple of each child part's state, in ``self.parts`` order."""
        return tuple(part.init_state() for part in self.parts)


class Residual(PCModule):
    """The shortcut junction: ``output = input + branch(input)``.

    Backward, the error is copied into both routes — unchanged along the
    shortcut, translated through the branch — and the two messages add.
    """

    def __init__(self, branch: PCModule):
        self.branch = branch

    def init_state(self):
        """Return the branch's state (the shortcut itself carries no state)."""
        return self.branch.init_state()
