# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from jax import jit

from jax.nn import sigmoid
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_state_node_prediction(
    attributes: dict, edges: Edges, node_idx: int, **args
) -> dict:
    r"""Get the new expected mean and precision of a binary state node.

    The predictions of a binary state node :math:`b` at time :math:`k` depends on the
    prediction of its value parent :math:`a`, such as:

    .. math::

        \hat{\mu}_b^{(k)} = \frac{1}{1 + e^{-\hat{\mu}_a^{(k)}}}

    and

    .. math::

        \hat{\pi}_b^{(k)} = \hat{\mu}^{(k)}(1-\hat{\mu}^{(k)})

    which corresponds to the uncertainty at the first level (i.e. inverse of the
    precision).

    .. warning::

        However, we keep the same name internally (i.e. `"precision"`) so this value can
        be used by the posterior update at the second level without differentiating
        between binary and continuous state nodes.

    .. note::

        Here we use the inverse (i.e. uncertainty) so this value can be use as such in
        the posterior update of the value parent (eq. 81, Weber et al., v2) without
        requiering a different update step for binary vs. continuous nodes to compensate
        for this, the value prediction error is divided by the expected_precision in the
        prediction error step.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the binary state node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    # List the (unique) value parent of the binary state node
    expected_mean = 0.0
    for value_parent_idx in edges[node_idx].value_parents:  # type: ignore
        expected_mean += attributes[value_parent_idx]["expected_mean"]

    # Estimate the new expected mean using the sigmoid transform
    # eq. 80 in Weber et al., v2
    expected_mean = sigmoid(expected_mean)
    attributes[node_idx]["expected_mean"] = expected_mean

    # Estimate the new expected precision from the new expected mean
    # note that here we use the inverse (i.e. uncertainty) so this value can be use
    # as such in the posterior update of the value parent (eq. 81, Weber et al., v2)
    # without requiering a different update step for binary vs. continuous nodes
    # to compensate for this, the value prediction error is divided by the
    # expected_precision in the prediction error step
    attributes[node_idx]["expected_precision"] = expected_mean * (1 - expected_mean)

    return attributes
