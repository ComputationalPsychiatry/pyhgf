# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

from jax import jit

from pyhgf.typing import Attributes


@partial(jit, static_argnames=("node_idx"))
def set_observation(
    attributes: Attributes,
    node_idx: int,
    values: float,
    observed: int,
) -> Attributes:
    r"""Add observations to the target node by setting the posterior to a given value.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network.
    node_idx :
        Pointer to the input node.
    values :
        The new observed value.
    observed :
        Whether value was observed or not.

    Returns
    -------
    attributes :
        The attributes of the probabilistic network.

    """
    attributes[node_idx]["mean"] = values
    attributes[node_idx]["observed"] = observed

    return attributes


@partial(jit, static_argnames=("node_idx"))
def set_predictors(
    attributes: Attributes,
    node_idx: int,
    values: float,
) -> Attributes:
    r"""Add observations to the predictor layer of a deep network.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network.
    node_idx :
        Pointer to the input node.
    values :
        The new observed value.

    Returns
    -------
    attributes :
        The attributes of the probabilistic network.

    """
    attributes[node_idx]["expected_mean"] = values

    return attributes
