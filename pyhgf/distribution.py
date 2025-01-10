# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Callable, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
from jax import Array, grad, jit, vmap
from jax.tree_util import Partial
from jax.typing import ArrayLike
from pytensor.graph import Apply, Op
from pytensor.tensor.variable import TensorVariable

from pyhgf.model import HGF


@partial(jit, static_argnames=("hgf", "response_function"))
def logp(
    mean_1: float,
    mean_2: float,
    mean_3: float,
    precision_1: float,
    precision_2: float,
    precision_3: float,
    tonic_volatility_1: float,
    tonic_volatility_2: float,
    tonic_volatility_3: float,
    tonic_drift_1: float,
    tonic_drift_2: float,
    tonic_drift_3: float,
    volatility_coupling_1: float,
    volatility_coupling_2: float,
    input_precision: float,
    response_function_parameters: ArrayLike,
    input_data: ArrayLike,
    time_steps: ArrayLike,
    response_function_inputs: Optional[ArrayLike],
    response_function: Optional[Callable],
    hgf: HGF,
) -> float:
    """Compute the log-probability of a decision model under belief trajectories.

    This function returns the evidence of a single Hierarchical Gaussian Filter given
    network parameters, input data and behaviours under a decision model.

    Parameters
    ----------
    mean_1 :
        The mean at the first level of the HGF. For the continuous HGF, this is the mean
        of the first value parent (x_1). For the binary HGF this is the mean of the
        binary state node (x_0).
    mean_2 :
        The mean at the second level of the HGF. For the continuous HGF, this is the
        mean of the first volatility parent (x_2). For the binary HGF this is the mean
        of the first continuous state node (x_1).
    mean_3 :
        The mean at the third level of the HGF. The value of this parameter will be
        ignored when using a two-level HGF (`n_levels=2`). For the continuous HGF, this
        is the mean of the second volatility parent (x_3). For the binary HGF this is
        the mean of the first volatility parent (x_2).
    precision_1 :
        The precision at the first level of the HGF. For the continuous HGF, this is the
        precision of the first value parent (x_1). For the binary HGF this is the
        precision of the binary state node (x_0).
    precision_2 :
        The precision at the second level of the HGF. For the continuous HGF, this is
        the precision of the first volatility parent (x_2). For the binary HGF this is
        the precision of the first continuous state node (x_1).
    precision_3 :
        The precision at the third level of the HGF. The value of this parameter will
        be ignored when using a two-level HGF (`n_levels=2`). For the continuous HGF,
        this is the precision of the second volatility parent (x_3). For the binary HGF
        this is the precision of the first volatility parent (x_2).
    tonic_volatility_1 :
        The tonic volatility at the first level (x_1 for the continuous HGF, x_2 for the
        binary HGF). This parameter represents the tonic part of the variance (the part
        that is not inherited from parent nodes).
    tonic_volatility_2 :
        The tonic volatility at the second level (x_2 for the continuous HGF, x_3 for
        the binary HGF). This parameter represents the tonic part of the variance (the
        part that is not inherited from parent nodes).
    tonic_volatility_3 :
        The tonic volatility at the third level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parent
        nodes). This parameter is only used for a three-level continuous HGF.
    tonic_drift_1 :
        The tonic drift at the first level of the HGF (x_1 for the continuous HGF,
        x_2 for the binary HGF). This parameter represents the drift of the random walk.
    tonic_drift_2 :
        The tonic drift at the second level of the HGF (x_2 for the continuous HGF,
        x_3 for the binary HGF). This parameter represents the drift of the random walk.
    tonic_drift_3 :
        The tonic drift at the third level of the HGF. This parameter represents the
        drift of the random walk. This parameter is only used for a three-level
        continuous HGF.
    volatility_coupling_1 :
        The volatility coupling between the first and second levels of the HGF (between
        x_1 and x_2 for a continuous HGF, and between x_2 and x_3 for a binary HGF).
        This represents the phasic part of the variance (the part affected by the
        parent nodes). Defaults to `1.0` (full connectivity).
    volatility_coupling_2 :
        The volatility coupling between the second and third levels of the HGF (x_2 and
        x_2 for a continuous HGF, not applicable to a binary HGF). This represents the
        phasic part of the variance (the part affected by the parent nodes). Defaults
        to `1.0`  (full connectivity). The value of this parameter will be ignored when
        using a two-level HGF (`n_levels=2`).
    input_precision :
        The expected precision associated with the continuous input.
    response_function_parameters :
        An array of additional parameters that will be passed to the response function
        to compute the surprise. This can include values over which inference is
        performed in a PyMC model (e.g. the inverse temperature of a binary softmax).
    input_data :
        An array of input time series. The first dimension is the number of time steps
        and the second dimension is the number of features. The number of features is
        the number of input nodes time the input dimensions.
    time_steps :
        An array of input time steps where the first dimension is the number of models
        to fit in parallel.
    response_function_inputs :
        An array of behavioural inputs passed to the response function where the first
        dimension is the number of models to fit in parallel.
    response_function :
        The response function that is used by the decision model.
    hgf :
        An instance of a two or three-level Hierarchical Gaussian Filter.

    Returns
    -------
    logp :
        The log-probability (negative surprise).

    """
    if hgf.model_type == "continuous":

        # update this network's attributes
        hgf.attributes[0]["precision"] = input_precision

        hgf.attributes[1]["mean"] = mean_1
        hgf.attributes[2]["mean"] = mean_2

        hgf.attributes[1]["precision"] = precision_1
        hgf.attributes[2]["precision"] = precision_2

        hgf.attributes[1]["tonic_volatility"] = tonic_volatility_1
        hgf.attributes[2]["tonic_volatility"] = tonic_volatility_2

        hgf.attributes[1]["tonic_drift"] = tonic_drift_1
        hgf.attributes[2]["tonic_drift"] = tonic_drift_2

        hgf.attributes[1]["volatility_coupling_parents"] = (volatility_coupling_1,)
        hgf.attributes[2]["volatility_coupling_children"] = (volatility_coupling_1,)

        if hgf.n_levels == 3:
            hgf.attributes[3]["mean"] = mean_3
            hgf.attributes[3]["precision"] = precision_3
            hgf.attributes[3]["tonic_volatility"] = tonic_volatility_3
            hgf.attributes[3]["tonic_drift"] = tonic_drift_3
            hgf.attributes[2]["volatility_coupling_parents"] = (volatility_coupling_2,)
            hgf.attributes[3]["volatility_coupling_children"] = (volatility_coupling_2,)

    elif hgf.model_type == "binary":

        # update this network's attributes
        hgf.attributes[0]["mean"] = mean_1
        hgf.attributes[1]["mean"] = mean_2

        hgf.attributes[0]["precision"] = precision_1
        hgf.attributes[1]["precision"] = precision_2

        hgf.attributes[1]["tonic_volatility"] = tonic_volatility_2

        hgf.attributes[1]["tonic_drift"] = tonic_drift_1

        if hgf.n_levels == 3:
            hgf.attributes[2]["mean"] = mean_3
            hgf.attributes[2]["precision"] = precision_3
            hgf.attributes[2]["tonic_volatility"] = tonic_volatility_3
            hgf.attributes[2]["tonic_drift"] = tonic_drift_3
            hgf.attributes[1]["volatility_coupling_parents"] = (volatility_coupling_2,)
            hgf.attributes[2]["volatility_coupling_children"] = (volatility_coupling_2,)

    surprise = hgf.input_data(input_data=input_data, time_steps=time_steps).surprise(
        response_function=response_function,
        response_function_inputs=response_function_inputs,
        response_function_parameters=response_function_parameters,
    )

    return -surprise


def hgf_logp(
    mean_1: ArrayLike = 0.0,
    mean_2: ArrayLike = 0.0,
    mean_3: ArrayLike = 0.0,
    precision_1: ArrayLike = 1.0,
    precision_2: ArrayLike = 1.0,
    precision_3: ArrayLike = 1.0,
    tonic_volatility_1: ArrayLike = -3.0,
    tonic_volatility_2: ArrayLike = -3.0,
    tonic_volatility_3: ArrayLike = -3.0,
    tonic_drift_1: ArrayLike = 0.0,
    tonic_drift_2: ArrayLike = 0.0,
    tonic_drift_3: ArrayLike = 0.0,
    volatility_coupling_1: ArrayLike = 1.0,
    volatility_coupling_2: ArrayLike = 1.0,
    input_precision: ArrayLike = np.inf,
    response_function_parameters: ArrayLike = np.ones(1),
    vectorized_logp: Callable = logp,
    input_data: ArrayLike = np.nan,
    response_function_inputs: ArrayLike = np.nan,
    time_steps: ArrayLike = np.nan,
) -> Array:
    """Compute log-probabilities of a batch of Hierarchical Gaussian Filters.

    .. hint::

       This function supports broadcasting along the first axis, which means that it can
       fit multiple models when input data are provided. When a network parameter is a
       float, this value will be used on all models. When a network parameter is an
       array, the size should match the number of input data, and different values will
       be used accordingly.

    Parameters
    ----------
    mean_1 :
        The mean at the first level of the HGF.
    mean_2 :
        The mean at the second level of the HGF.
    mean_3 :
        The mean at the third level of the HGF. The value of this parameter will be
        ignored when using a two-level HGF (`n_levels=2`).
    precision_1 :
        The precision at the first level of the HGF.
    precision_2 :
        The precision at the second level of the HGF.
    precision_3 :
        The precision at the third level of the HGF. The value of this parameter will
        be ignored when using a two-level HGF (`n_levels=2`).
    tonic_volatility_1 :
        The tonic volatility at the first level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parent
        nodes).
    tonic_volatility_2 :
        The tonic volatility at the second level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parent
        nodes).
    tonic_volatility_3 :
        The tonic volatility at the third level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parent
        nodes). The value of this parameter will be ignored when using a two-level HGF
        (`n_levels=2`).
    tonic_drift_1 :
        The tonic drift at the first level of the HGF. This parameter represents the
        drift of the random walk.
    tonic_drift_2 :
        The tonic drift at the second level of the HGF. This parameter represents the
        drift of the random walk.
    tonic_drift_3 :
        The tonic drift at the first level of the HGF. This parameter represents the
        drift of the random walk. The value of this parameter will be ignored when
        using a two-level HGF (`n_levels=2`).
    volatility_coupling_1 :
        The volatility coupling between the first and second levels of the HGF. This
        represents the phasic part of the variance (the part affected by the
        parent nodes). Defaults to `1.0`.
    volatility_coupling_2 :
        The volatility coupling between the second and third levels of the HGF. This
        represents the phasic part of the variance (the part affected by the
        parent nodes). Defaults to `1.0`. The value of this parameter will be
        ignored when using a two-level HGF (`n_levels=2`).
    input_precision :
        The expected precision associated with the continuous or binary input, depending
        on the model type. The default is `np.inf`.
    response_function_parameters :
        An array list of additional parameters that will be passed to the response
        function. This can include values over which inference is performed in a PyMC
        model (e.g. the inverse temperature of a binary softmax).
    vectorized_logp :
        A vectorized log probability function for a two or three-layered HGF.
    input_data :
        An array of input time series where the first dimension is the number of models
        to fit in parallel.
    response_function_inputs :
        An array of behavioural input passed to the response function where the first
        dimension is the number of models to fit in parallel.
    time_steps :
        An array of input time steps where the first dimension is the number of models
        to fit in parallel.

    Returns
    -------
    log_prob :
        The sum of the log probabilities (negative surprise).

    """
    # number of models
    n = input_data.shape[0]

    # ensure that the response parameters have n_model as first dimension
    response_function_parameters = jnp.broadcast_to(
        response_function_parameters, (n,) + response_function_parameters.shape[1:]
    )

    # Broadcast inputs to an array with length n>=1
    (
        _tonic_volatility_1,
        _tonic_volatility_2,
        _tonic_volatility_3,
        _input_precision,
        _tonic_drift_1,
        _tonic_drift_2,
        _tonic_drift_3,
        _precision_1,
        _precision_2,
        _precision_3,
        _mean_1,
        _mean_2,
        _mean_3,
        _volatility_coupling_1,
        _volatility_coupling_2,
        _,
    ) = jnp.broadcast_arrays(
        tonic_volatility_1,
        tonic_volatility_2,
        tonic_volatility_3,
        input_precision,
        tonic_drift_1,
        tonic_drift_2,
        tonic_drift_3,
        precision_1,
        precision_2,
        precision_3,
        mean_1,
        mean_2,
        mean_3,
        volatility_coupling_1,
        volatility_coupling_2,
        jnp.zeros(n),
    )

    logp = vectorized_logp(
        input_data=input_data,
        response_function_inputs=response_function_inputs,
        response_function_parameters=response_function_parameters,
        time_steps=time_steps,
        mean_1=_mean_1,
        mean_2=_mean_2,
        mean_3=_mean_3,
        precision_1=_precision_1,
        precision_2=_precision_2,
        precision_3=_precision_3,
        tonic_volatility_1=_tonic_volatility_1,
        tonic_volatility_2=_tonic_volatility_2,
        tonic_volatility_3=_tonic_volatility_3,
        tonic_drift_1=_tonic_drift_1,
        tonic_drift_2=_tonic_drift_2,
        tonic_drift_3=_tonic_drift_3,
        volatility_coupling_1=_volatility_coupling_1,
        volatility_coupling_2=_volatility_coupling_2,
        input_precision=_input_precision,
    )

    # Return the log probabilities (negative surprise)
    return logp.sum(), logp


class HGFLogpGradOp(Op):
    """Gradient Op for the HGF distribution."""

    def __init__(
        self,
        input_data: Union[ArrayLike] = np.nan,
        time_steps: Optional[ArrayLike] = None,
        model_type: str = "continous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_inputs: Optional[ArrayLike] = None,
    ):
        """Initialize function.

        Parameters
        ----------
        input_data :
            An array of input time series where the first dimension is the number of
            models to fit in parallel. By default, the associated time_steps vector is
            the unit vector. A different time vector can be passed to the `time_steps`
            argument.
        time_steps :
            An array with shapes equal to input_data containing the time_steps vectors.
            If one of the list items is `None`, or if `None` is provided instead, the
            time_steps vector will default to an integer vector starting at 0.
        model_type :
            The model type to use (can be "continuous" or "binary").
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterwards
            using `add_nodes()`.
        response_function :
            The response function to use to compute the model surprise.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.

        """
        if time_steps is None:
            time_steps = np.ones(shape=input_data.shape)

        # create the default HGF template to be use by the logp function
        self.hgf = HGF(n_levels=n_levels, model_type=model_type)

        # create a vectorized version of the logp function
        vectorized_logp = vmap(
            Partial(
                logp,
                hgf=self.hgf,
                response_function=response_function,
            )
        )

        # define the gradient of the vectorized function
        self.grad_logp = jit(
            grad(
                Partial(
                    hgf_logp,
                    vectorized_logp=vectorized_logp,
                    input_data=input_data,
                    time_steps=time_steps,
                    response_function_inputs=response_function_inputs,
                ),
                argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                has_aux=True,
            )
        )

    def make_node(
        self,
        mean_1: ArrayLike = np.array(0.0),
        mean_2: ArrayLike = np.array(0.0),
        mean_3: ArrayLike = np.array(0.0),
        precision_1: ArrayLike = np.array(1.0),
        precision_2: ArrayLike = np.array(1.0),
        precision_3: ArrayLike = np.array(1.0),
        tonic_volatility_1: ArrayLike = np.array(-3.0),
        tonic_volatility_2: ArrayLike = np.array(-3.0),
        tonic_volatility_3: ArrayLike = np.array(-3.0),
        tonic_drift_1: ArrayLike = np.array(0.0),
        tonic_drift_2: ArrayLike = np.array(0.0),
        tonic_drift_3: ArrayLike = np.array(0.0),
        volatility_coupling_1: ArrayLike = np.array(1.0),
        volatility_coupling_2: ArrayLike = np.array(1.0),
        input_precision: ArrayLike = np.array(1e4),
        response_function_parameters: ArrayLike = np.array(1.0),
    ):
        """Initialize node structure."""
        # Convert our inputs to symbolic variables
        inputs = [
            pt.as_tensor_variable(mean_1, ndim=1),
            pt.as_tensor_variable(mean_2, ndim=1),
            pt.as_tensor_variable(mean_3, ndim=1),
            pt.as_tensor_variable(precision_1, ndim=1),
            pt.as_tensor_variable(precision_2, ndim=1),
            pt.as_tensor_variable(precision_3, ndim=1),
            pt.as_tensor_variable(tonic_volatility_1, ndim=1),
            pt.as_tensor_variable(tonic_volatility_2, ndim=1),
            pt.as_tensor_variable(tonic_volatility_3, ndim=1),
            pt.as_tensor_variable(tonic_drift_1, ndim=1),
            pt.as_tensor_variable(tonic_drift_2, ndim=1),
            pt.as_tensor_variable(tonic_drift_3, ndim=1),
            pt.as_tensor_variable(volatility_coupling_1, ndim=1),
            pt.as_tensor_variable(volatility_coupling_2, ndim=1),
            pt.as_tensor_variable(input_precision, ndim=1),
            pt.as_tensor_variable(response_function_parameters),
        ]
        # This `Op` will return one gradient per input. For simplicity, we assume
        # each output is of the same type as the input. In practice, you should use
        # the exact dtype to avoid overhead when saving the results of the computation
        # in `perform`
        outputs = [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(
        self, node, inputs: List[TensorVariable], outputs: List[TensorVariable]
    ):
        """Perform node operations."""
        (
            grad_mean_1,
            grad_mean_2,
            grad_mean_3,
            grad_precision_1,
            grad_precision_2,
            grad_precision_3,
            grad_tonic_volatility_1,
            grad_tonic_volatility_2,
            grad_tonic_volatility_3,
            grad_tonic_drift_1,
            grad_tonic_drift_2,
            grad_tonic_drift_3,
            grad_volatility_coupling_1,
            grad_volatility_coupling_2,
            grad_input_precision,
            grad_response_function_parameters,
        ), _ = self.grad_logp(*inputs)

        outputs[0][0] = np.asarray(grad_mean_1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_mean_2, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(grad_mean_3, dtype=node.outputs[2].dtype)
        outputs[3][0] = np.asarray(grad_precision_1, dtype=node.outputs[3].dtype)
        outputs[4][0] = np.asarray(grad_precision_2, dtype=node.outputs[4].dtype)
        outputs[5][0] = np.asarray(grad_precision_3, dtype=node.outputs[5].dtype)
        outputs[6][0] = np.asarray(grad_tonic_volatility_1, dtype=node.outputs[6].dtype)
        outputs[7][0] = np.asarray(grad_tonic_volatility_2, dtype=node.outputs[7].dtype)
        outputs[8][0] = np.asarray(grad_tonic_volatility_3, dtype=node.outputs[8].dtype)
        outputs[9][0] = np.asarray(grad_tonic_drift_1, dtype=node.outputs[9].dtype)
        outputs[10][0] = np.asarray(grad_tonic_drift_2, dtype=node.outputs[10].dtype)
        outputs[11][0] = np.asarray(grad_tonic_drift_3, dtype=node.outputs[11].dtype)
        outputs[12][0] = np.asarray(
            grad_volatility_coupling_1, dtype=node.outputs[12].dtype
        )
        outputs[13][0] = np.asarray(
            grad_volatility_coupling_2, dtype=node.outputs[13].dtype
        )
        outputs[14][0] = np.asarray(grad_input_precision, dtype=node.outputs[14].dtype)
        outputs[15][0] = np.asarray(
            grad_response_function_parameters, dtype=node.outputs[15].dtype
        )


class HGFDistribution(Op):
    r"""The HGF distribution PyMC >= 5.0 compatible.

    Examples
    --------
    This class should be used in the context of a PyMC probabilistic model to estimate
    one or many parameter-s probability density with MCMC sampling.

    .. code-block:: python

        import arviz as az
        import numpy as np
        import pymc as pm
        from math import log
        from pyhgf import load_data
        from pyhgf.distribution import HGFDistribution
        from pyhgf.response import total_gaussian_surprise

    Create the data (value and time vectors)

    .. code-block:: python

        timeserie = load_data("continuous")

    We create the PyMC distribution here, specifying the type of model and response
    function we want to use (i.e. simple Gaussian surprise).

    .. code-block:: python

        hgf_logp_op = HGFDistribution(
            n_levels=2,
            input_data=timeserie[np.newaxis],
            response_function=total_gaussian_surprise,
        )

    Create the PyMC model. Here we are fixing all the parameters to arbitrary values and
    sampling the posterior probability density of the tonic volatility at the second
    level, using a normal distribution as prior:

    .. math::

        \omega_2 \sim \mathcal{N}(-11.0, 2.0)

    .. code-block:: python

        with pm.Model() as model:

            # prior over tonic volatility
            ω_2 = pm.Normal("ω_2", -11.0, 2)

            pm.Potential(
                "hhgf_loglike",
                hgf_logp_op(
                    tonic_volatility_2=ω_2,
                    precision_1=1.0,
                    precision_2=1.0,
                    mean_1=1.0,
                ),
            )

    Sample the model - Using 2 chains, 1 cores and 1000 warmups.

    .. code-block:: python

        with model:
            idata = pm.sample(chains=2, cores=1, tune=1000)

    Print a summary of the results using Arviz

    >>> az.summary(idata, var_names="ω_2")

    ===  =======  =====  =======  =======  =========  =======  ========  ========  =====
    ..      mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
    ===  =======  =====  =======  =======  =========  =======  ========  ========  =====
    ω_2  -14.776  1.065  -16.766  -12.944      0.038    0.027       806       938      1
    ===  =======  =====  =======  =======  =========  =======  ========  ========  =====

    """

    def __init__(
        self,
        input_data: ArrayLike = jnp.nan,
        time_steps: Optional[ArrayLike] = None,
        model_type: str = "continuous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_inputs: Optional[ArrayLike] = None,
    ):
        """Distribution initialization.

        Parameters
        ----------
        input_data :
            List of input data. When `n` models should be fitted, the list contains `n`
            1d Numpy arrays. By default, the associated time vector is the unit
            vector starting at `0`. A different time_steps vector can be passed to
            the `time_steps` argument.
        time_steps :
            List of 1d Numpy arrays containing the time_steps vectors for each input
            time series. If one of the list items is `None`, or if `None` is provided
            instead, the time vector will default to an integers vector starting at 0.
        model_type :
            The model type to use (can be "continuous" or "binary").
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterwards
            using `add_nodes()`.
        response_function :
            The response function to use to compute the model surprise.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.

        """
        if time_steps is None:
            time_steps = np.ones(shape=input_data.shape)

        assert time_steps.shape == input_data.shape

        # create the default HGF template to be use by the logp function
        self.hgf = HGF(n_levels=n_levels, model_type=model_type)

        # create a vectorized version of the logp function
        vectorized_logp = vmap(
            Partial(
                logp,
                hgf=self.hgf,
                response_function=response_function,
            )
        )

        # The value function
        self.hgf_logp = jit(
            Partial(
                hgf_logp,
                vectorized_logp=vectorized_logp,
                input_data=input_data,
                time_steps=time_steps,
                response_function_inputs=response_function_inputs,
            ),
        )

        # The gradient function
        self.hgf_logp_grad_op = HGFLogpGradOp(
            input_data=input_data,
            time_steps=time_steps,
            model_type=model_type,
            n_levels=n_levels,
            response_function=response_function,
            response_function_inputs=response_function_inputs,
        )

    def make_node(
        self,
        mean_1: ArrayLike = np.array(0.0),
        mean_2: ArrayLike = np.array(0.0),
        mean_3: ArrayLike = np.array(0.0),
        precision_1: ArrayLike = np.array(1.0),
        precision_2: ArrayLike = np.array(1.0),
        precision_3: ArrayLike = np.array(1.0),
        tonic_volatility_1: ArrayLike = np.array(-3.0),
        tonic_volatility_2: ArrayLike = np.array(-3.0),
        tonic_volatility_3: ArrayLike = np.array(-3.0),
        tonic_drift_1: ArrayLike = np.array(0.0),
        tonic_drift_2: ArrayLike = np.array(0.0),
        tonic_drift_3: ArrayLike = np.array(0.0),
        volatility_coupling_1: ArrayLike = np.array(1.0),
        volatility_coupling_2: ArrayLike = np.array(1.0),
        input_precision: ArrayLike = np.array(1e4),
        response_function_parameters: ArrayLike = np.array(1.0),
    ):
        """Convert inputs to symbolic variables."""
        inputs = [
            pt.as_tensor_variable(mean_1, ndim=1),
            pt.as_tensor_variable(mean_2, ndim=1),
            pt.as_tensor_variable(mean_3, ndim=1),
            pt.as_tensor_variable(precision_1, ndim=1),
            pt.as_tensor_variable(precision_2, ndim=1),
            pt.as_tensor_variable(precision_3, ndim=1),
            pt.as_tensor_variable(tonic_volatility_1, ndim=1),
            pt.as_tensor_variable(tonic_volatility_2, ndim=1),
            pt.as_tensor_variable(tonic_volatility_3, ndim=1),
            pt.as_tensor_variable(tonic_drift_1, ndim=1),
            pt.as_tensor_variable(tonic_drift_2, ndim=1),
            pt.as_tensor_variable(tonic_drift_3, ndim=1),
            pt.as_tensor_variable(volatility_coupling_1, ndim=1),
            pt.as_tensor_variable(volatility_coupling_2, ndim=1),
            pt.as_tensor_variable(input_precision, ndim=1),
            pt.as_tensor_variable(response_function_parameters),
        ]
        # Define the type of output returned by the wrapped JAX function
        outputs = [pt.scalar(dtype=float)]
        return Apply(self, inputs, outputs)

    def perform(
        self, node, inputs: List[TensorVariable], outputs: List[TensorVariable]
    ):
        """Run the function forward."""
        log_likelihood, _ = self.hgf_logp(*inputs)
        outputs[0][0] = np.asarray(log_likelihood, dtype=float)

    def grad(
        self, inputs: List[TensorVariable], output_gradients: List[TensorVariable]
    ) -> List[TensorVariable]:
        """Gradient of the function."""
        (
            grad_mean_1,
            grad_mean_2,
            grad_mean_3,
            grad_precision_1,
            grad_precision_2,
            grad_precision_3,
            grad_tonic_volatility_1,
            grad_tonic_volatility_2,
            grad_tonic_volatility_3,
            grad_tonic_drift_1,
            grad_tonic_drift_2,
            grad_tonic_drift_3,
            grad_volatility_coupling_1,
            grad_volatility_coupling_2,
            grad_input_precision,
            grad_response_function_parameters,
        ) = self.hgf_logp_grad_op(*inputs)

        # If there are inputs for which the gradients will never be needed or cannot
        # be computed, `pytensor.gradient.grad_not_implemented` should  be used as the
        # output gradient for that input.
        output_gradient = output_gradients[0]
        return [
            output_gradient * grad_mean_1,
            output_gradient * grad_mean_2,
            output_gradient * grad_mean_3,
            output_gradient * grad_precision_1,
            output_gradient * grad_precision_2,
            output_gradient * grad_precision_3,
            output_gradient * grad_tonic_volatility_1,
            output_gradient * grad_tonic_volatility_2,
            output_gradient * grad_tonic_volatility_3,
            output_gradient * grad_tonic_drift_1,
            output_gradient * grad_tonic_drift_2,
            output_gradient * grad_tonic_drift_3,
            output_gradient * grad_volatility_coupling_1,
            output_gradient * grad_volatility_coupling_2,
            output_gradient * grad_input_precision,
            output_gradient * grad_response_function_parameters,
        ]


class HGFPointwise(Op):
    """The HGF distribution returning pointwise log probability.

    This class should be used in the context of model comparison where the pointwise log
    probabilities are required for cross-validation.

    """

    def __init__(
        self,
        input_data: ArrayLike = jnp.nan,
        time_steps: Optional[ArrayLike] = None,
        model_type: str = "continuous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_inputs: Optional[ArrayLike] = None,
    ):
        """Distribution initialization.

        Parameters
        ----------
        input_data :
            List of input data. When `n` models should be fitted, the list contains `n`
            1d Numpy arrays. By default, the associated time vector is the unit
            vector starting at `0`. A different time_steps vector can be passed to
            the `time_steps` argument.
        time_steps :
            List of 1d Numpy arrays containing the time_steps vectors for each input
            time series. If one of the list items is `None`, or if `None` is provided
            instead, the time vector will default to an integers vector starting at 0.
        model_type :
            The model type to use (can be "continuous" or "binary").
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterwards
            using `add_nodes()`.
        response_function :
            The response function to use to compute the model surprise.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.

        """
        if time_steps is None:
            time_steps = np.ones(shape=input_data.shape)

        # create the default HGF template to be use by the logp function
        self.hgf = HGF(n_levels=n_levels, model_type=model_type)

        # create a vectorized version of the logp function
        vectorized_logp = vmap(
            Partial(
                logp,
                hgf=self.hgf,
                response_function=response_function,
            )
        )

        # The value function
        self.hgf_logp = jit(
            Partial(
                hgf_logp,
                vectorized_logp=vectorized_logp,
                input_data=input_data,
                time_steps=time_steps,
                response_function_inputs=response_function_inputs,
            ),
        )

    def make_node(
        self,
        mean_1: ArrayLike = np.array(0.0),
        mean_2: ArrayLike = np.array(0.0),
        mean_3: ArrayLike = np.array(0.0),
        precision_1: ArrayLike = np.array(1.0),
        precision_2: ArrayLike = np.array(1.0),
        precision_3: ArrayLike = np.array(1.0),
        tonic_volatility_1: ArrayLike = np.array(-3.0),
        tonic_volatility_2: ArrayLike = np.array(-3.0),
        tonic_volatility_3: ArrayLike = np.array(-3.0),
        tonic_drift_1: ArrayLike = np.array(0.0),
        tonic_drift_2: ArrayLike = np.array(0.0),
        tonic_drift_3: ArrayLike = np.array(0.0),
        volatility_coupling_1: ArrayLike = np.array(1.0),
        volatility_coupling_2: ArrayLike = np.array(1.0),
        input_precision: ArrayLike = np.array(1e4),
        response_function_parameters: ArrayLike = np.array(1.0),
    ):
        """Convert inputs to symbolic variables."""
        inputs = [
            pt.as_tensor_variable(mean_1),
            pt.as_tensor_variable(mean_2),
            pt.as_tensor_variable(mean_3),
            pt.as_tensor_variable(precision_1),
            pt.as_tensor_variable(precision_2),
            pt.as_tensor_variable(precision_3),
            pt.as_tensor_variable(tonic_volatility_1),
            pt.as_tensor_variable(tonic_volatility_2),
            pt.as_tensor_variable(tonic_volatility_3),
            pt.as_tensor_variable(tonic_drift_1),
            pt.as_tensor_variable(tonic_drift_2),
            pt.as_tensor_variable(tonic_drift_3),
            pt.as_tensor_variable(volatility_coupling_1),
            pt.as_tensor_variable(volatility_coupling_2),
            pt.as_tensor_variable(input_precision),
            pt.as_tensor_variable(response_function_parameters),
        ]
        # Define the type of output returned by the wrapped JAX function
        outputs = [pt.matrix(dtype=float)]
        return Apply(self, inputs, outputs)

    def perform(
        self, node, inputs: List[TensorVariable], outputs: List[TensorVariable]
    ):
        """Run the function forward."""
        _, pointwise_log_likelihood = self.hgf_logp(*inputs)
        outputs[0][0] = np.asarray(pointwise_log_likelihood, dtype=float)
