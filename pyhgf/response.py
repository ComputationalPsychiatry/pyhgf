# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.typing import ArrayLike

from pyhgf.math import binary_surprise, gaussian_surprise

if TYPE_CHECKING:
    from pyhgf.model import HGF


def first_level_gaussian_surprise(
    hgf: "HGF", response_function_inputs=None, response_function_parameters=None
) -> float:
    """Gaussian surprise at the first level of a probabilistic network.

    .. note::
      The Gaussian surprise at the first level is the default method to compute surprise
      for continuous models. The function returns `jnp.inf` if the model could not fit
      at a given time point.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_inputs :
        The inputs to the response functions, not required here.
    response_function_parameters :
        Additionnal parameters for the response function, not required here.

    Returns
    -------
    surprise :
        The model's surprise given the input data.

    """
    # compute the sum of Gaussian surprise at the first level
    # the input value at time t is compared to the Gaussian prediction at t-1
    surprise = gaussian_surprise(
        x=hgf.node_trajectories[0]["mean"],
        expected_mean=hgf.node_trajectories[1]["expected_mean"],
        expected_precision=hgf.node_trajectories[1]["expected_precision"],
    )

    # Return an infinite surprise if the model could not fit at any point
    return jnp.where(jnp.isnan(surprise), jnp.inf, surprise)


def total_gaussian_surprise(
    hgf: "HGF", response_function_inputs=None, response_function_parameters=None
) -> float:
    """Sum of the Gaussian surprise across the probabilistic network.

    .. note::
      The function returns `jnp.inf` if the model could not fit at a given time point.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_inputs :
        The inputs to the response functions, not required here.
    response_function_parameters :
        Additionnal parameters for the response function, not required here.

    Returns
    -------
    surprise :
        The model's surprise given the input data.

    """
    # compute the sum of Gaussian surprise across every node
    surprise = 0.0

    # then we do the same for every node that is not an input node
    # and not the parent of an input node
    for i in range(len(hgf.edges)):
        if hgf.edges[i].node_type == 2:
            surprise += gaussian_surprise(
                x=hgf.node_trajectories[i]["mean"],
                expected_mean=hgf.node_trajectories[i]["expected_mean"],
                expected_precision=hgf.node_trajectories[i]["expected_precision"],
            )

    # Return an infinite surprise if the model could not fit at any point
    return jnp.where(
        jnp.any(jnp.isnan(hgf.node_trajectories[1]["mean"])), jnp.inf, surprise
    )


def first_level_binary_surprise(
    hgf: "HGF", response_function_inputs=None, response_function_parameters=None
) -> float:
    """Time series of binary surprises for all binary state nodes.

    .. note::
      The binary surprise is the default method to compute surprise when
      `model_type=="binary"`, therefore this method will only return the valid time
      points, and `jnp.inf` if the model could not fit.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_inputs :
        The inputs to the response functions, not required here.
    response_function_parameters :
        Additionnal parameters for the response function, not required here.

    Returns
    -------
    surprise :
        The model's surprise given the input data.

    """
    surprise = jnp.zeros(len(hgf.node_trajectories[-1]["time_step"]))
    for binary_input_idx in hgf.input_idxs:
        surprise += binary_surprise(
            expected_mean=hgf.node_trajectories[binary_input_idx]["expected_mean"],
            x=hgf.node_trajectories[binary_input_idx]["mean"],
        )

    # Return an infinite surprise if the model cannot fit
    surprise = jnp.where(
        jnp.isnan(surprise),
        jnp.inf,
        surprise,
    )

    return surprise


def binary_softmax(
    hgf: "HGF",
    response_function_inputs=ArrayLike,
    response_function_parameters=ArrayLike,
) -> float:
    """Surprise under the binary sofmax model.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_inputs :
        The inputs to the response functions, here containing the decision from the
        paraticipant at time *k* [0 or 1].
    response_function_parameters :
        Additionnal parameters for the response function, not required here.

    Returns
    -------
    surprise :
        The surprise under the binary sofmax model.

    """
    # the expected values at the first level of the HGF
    beliefs = hgf.node_trajectories[0]["expected_mean"]

    # the binary surprises
    surprise = binary_surprise(x=response_function_inputs, expected_mean=beliefs)

    # ensure that inf is returned if the model cannot fit
    surprise = jnp.where(jnp.isnan(surprise), jnp.inf, surprise)

    return surprise


def binary_softmax_inverse_temperature(
    hgf: "HGF",
    response_function_inputs=ArrayLike,
    response_function_parameters=ArrayLike,
) -> float:
    r"""Surprise from a binary sofmax parametrized by the inverse temperature.

    The probability of chosing A is given by:

    .. math::

       P(A|\hat{\mu}^{(k)_{1}, t) = \frac{1}{1+e^{-t\hat{\mu}^{(k)_{1}}}

    Where :math:`\hat{mu}^{(k)_{1}` is the expected probability of A at the firt level,
    and :math:`t` is the temperature parameter.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_inputs :
        The inputs to the response functions, here containing the decision from the
        paraticipant at time *k* [0 or 1].
    response_function_parameters :
        Additionnal parameters for the response function (optional). Here, the inverse
        temperature  is provided.

    Returns
    -------
    surprise :
        The surprise under the binary sofmax model.

    """
    # the expected values at the first level of the HGF
    beliefs = (
        hgf.node_trajectories[0]["expected_mean"] ** response_function_parameters
    ) / (
        hgf.node_trajectories[0]["expected_mean"] ** response_function_parameters
        + (1 - hgf.node_trajectories[0]["expected_mean"])
        ** response_function_parameters
    )

    # the binary surprises
    surprise = binary_surprise(x=response_function_inputs, expected_mean=beliefs)

    # ensure that inf is returned if the model cannot fit
    surprise = jnp.where(jnp.isnan(surprise), jnp.inf, surprise)

    return surprise
