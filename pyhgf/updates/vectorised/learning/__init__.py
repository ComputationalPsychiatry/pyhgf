# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Weight learning for deep predictive coding networks.

Each ``learning_kind`` of :class:`pyhgf.model.DeepNetwork` is implemented in one
submodule, which holds the rule's equations and the step that applies them to every
weight matrix of a network:

- :mod:`~pyhgf.updates.vectorised.learning.factors`: the rank-one gradient and
  importance factors of ``"standard"`` and ``"precision_weighted"``, which
  ``"synaptic_uncertainty"`` also reads, and the evidence walk that feeds the importance
  factor.
- :mod:`~pyhgf.updates.vectorised.learning.variational`: the ``"variational"``
  gradient of ``"kronecker"``.
- :mod:`~pyhgf.updates.vectorised.learning.backprop`: the ``"backprop"`` gradient of
  ``"kronecker"``.
- :mod:`~pyhgf.updates.vectorised.learning.synaptic_uncertainty`: the weight-belief rule
  of ``"synaptic_uncertainty"``.
- :mod:`~pyhgf.updates.vectorised.learning.kronecker`: the Kronecker-factored weight
  belief of ``"kronecker"``.
- :mod:`~pyhgf.updates.vectorised.learning.optimiser`: the optax step of the two
  gradient kinds.

The gradient kinds (``"standard"`` and ``"precision_weighted"``) hand a descent gradient
to an optax optimiser. The weight-belief kinds (``"synaptic_uncertainty"`` and
``"kronecker"``) carry a precision for the weights and take their own step along a
descent gradient; ``"kronecker"`` can also descend the ``"variational"`` or the
``"backprop"`` gradient. Every public name of the submodules is importable from this
package.
"""

from pyhgf.updates.vectorised.learning.backprop import (
    BACKPROP_KIND,
    vectorised_backprop_gradient,
)
from pyhgf.updates.vectorised.learning.factors import (
    SEPARABLE_KINDS,
    clamped_layer_evidence,
    evidence_pullback,
    learning_weights_vectorised,
)
from pyhgf.updates.vectorised.learning.kronecker import (
    KRONECKER_DEFAULTS,
    KRONECKER_GRADIENTS,
    KRONECKER_KIND,
    KroneckerSettings,
    kronecker_direction,
    kronecker_factor_update,
    kronecker_increment_vectors,
    kronecker_mass_update,
    kronecker_slow_mean_update,
    kronecker_step_scale,
    resolve_kronecker_settings,
    vectorised_kronecker_update,
)
from pyhgf.updates.vectorised.learning.synaptic_uncertainty import (
    SYNAPTIC_UNCERTAINTY_DEFAULTS,
    SynapticUncertaintySettings,
    resolve_synaptic_uncertainty_settings,
    vectorised_synaptic_uncertainty_update,
)
from pyhgf.updates.vectorised.learning.variational import (
    VARIATIONAL_KIND,
    relaxed_effective_precision,
    vectorised_variational_gradient,
)

__all__ = [
    "BACKPROP_KIND",
    "KRONECKER_DEFAULTS",
    "KRONECKER_GRADIENTS",
    "KRONECKER_KIND",
    "KroneckerSettings",
    "SEPARABLE_KINDS",
    "SYNAPTIC_UNCERTAINTY_DEFAULTS",
    "SynapticUncertaintySettings",
    "VARIATIONAL_KIND",
    "clamped_layer_evidence",
    "evidence_pullback",
    "kronecker_direction",
    "kronecker_factor_update",
    "kronecker_increment_vectors",
    "kronecker_mass_update",
    "kronecker_slow_mean_update",
    "kronecker_step_scale",
    "learning_weights_vectorised",
    "relaxed_effective_precision",
    "resolve_kronecker_settings",
    "resolve_synaptic_uncertainty_settings",
    "vectorised_backprop_gradient",
    "vectorised_kronecker_update",
    "vectorised_synaptic_uncertainty_update",
    "vectorised_variational_gradient",
]
