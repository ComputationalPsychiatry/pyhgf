# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

r"""The ``"backprop"`` gradient of the Kronecker-factored weight belief.

Its descent gradient is the gradient, by automatic differentiation, of

.. math::

    \tfrac12 \sum_k \bar{\pi}_{0,k} \big( y_k - \hat{y}_k \big)^2,

with :math:`y` the posterior mean of the observed (bottom) layer, the observation it is
clamped to, :math:`\bar{\pi}_0` its conditional expected precision, and :math:`\hat{y}`
the forward pass of the network: from the mean of the top layer down, each layer set to
the value its parents predict through their coupling functions. The hidden beliefs the
update sweep inferred are not read, so this is the gradient backpropagation computes
for the same weights. Under the relaxed updates (not ``mean_field_updates``), as the
conditional expected precisions of the hidden layers grow with the top layer's variance
negligible against theirs, the precision-weighted and the variational gradients tend to
it (see
:func:`~pyhgf.updates.vectorised.learning.variational.vectorised_variational_gradient`).

:func:`vectorised_backprop_gradient` computes it after the update sweep, when the
``gradient`` setting of ``learning_kind="kronecker"`` names it. It is available with a
continuous observed layer, and for online learning only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pyhgf.typing.vectorised import VectorisedNetwork

#: The ``gradient`` setting of ``"kronecker"`` that ignores the inference: the gradient
#: of the precision-weighted squared error of the network's forward pass, by automatic
#: differentiation.
BACKPROP_KIND: str = "backprop"


def vectorised_backprop_gradient(network: VectorisedNetwork) -> tuple:
    r"""Gradient of the forward pass's precision-weighted squared error.

    Differentiates :math:`\tfrac12 \sum_k \bar{\pi}_{0,k} ( y_k - \hat{y}_k )^2` (see
    :mod:`~pyhgf.updates.vectorised.learning.backprop`) with respect to every weight
    matrix, by automatic differentiation. :math:`y` and :math:`\bar{\pi}_0` are read
    from the observed layer, and the forward pass starts from the mean of the top
    layer, so ``network`` is read after the update sweep. The hidden layers' beliefs
    are not read.

    Parameters
    ----------
    network :
        The network after the update sweep, with its layers added one by one.

    Returns
    -------
    tuple
        One gradient per element, shape of its ``weights_mean``, ``None`` for the
        observed (bottom) layer.

    Raises
    ------
    NotImplementedError
        If the observed layer is not continuous.
    """
    elements = network.layers
    if elements[0].kind not in ("volatile", "continuous"):
        raise NotImplementedError(
            "The 'backprop' gradient needs a continuous observed layer, got "
            f"{elements[0].kind!r}."
        )
    top = len(elements) - 1
    target = elements[0].state.mean
    precision = elements[0].state.conditional_expected_precision

    def loss(weights):
        value = elements[top].state.mean
        for i in range(top, 0, -1):
            parent = elements[i]
            matrix = weights[i - 1]
            n_parents = parent.state.mean.shape[-1]
            value = matrix[:, :n_parents] @ parent.coupling_fn(value)
            if parent.add_constant_input:
                value = value + matrix[:, -1]
        return 0.5 * jnp.sum(precision * (target - value) ** 2)

    weights = tuple(element.weights_mean for element in elements[1:])
    return (None,) + tuple(jax.grad(loss)(weights))
