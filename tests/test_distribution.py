# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import arviz as az
import jax.numpy as jnp
import numpy as np
import pymc as pm
from jax import grad, jit, vmap
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.distribution import (
    HGFDistribution,
    HGFLogpGradOp,
    HGFPointwise,
    hgf_logp,
    logp,
)
from pyhgf.model import HGF
from pyhgf.response import (
    binary_softmax_inverse_temperature,
    first_level_gaussian_surprise,
)


def test_logp():
    """Test the log-probability function for single model fit."""
    timeseries = load_data("continuous")
    hgf = HGF(n_levels=2, model_type="continuous")

    log_likelihood = logp(
        tonic_volatility_1=-3.0,
        tonic_volatility_2=-3.0,
        tonic_volatility_3=jnp.nan,
        input_precision=np.array(1e4),
        tonic_drift_1=0.0,
        tonic_drift_2=0.0,
        tonic_drift_3=jnp.nan,
        precision_1=1e4,
        precision_2=1e1,
        precision_3=jnp.nan,
        mean_1=1.0,
        mean_2=0.0,
        mean_3=jnp.nan,
        volatility_coupling_1=1.0,
        volatility_coupling_2=jnp.nan,
        response_function_parameters=[jnp.nan],
        input_data=timeseries,
        time_steps=np.ones(shape=timeseries.shape),
        response_function_inputs=jnp.nan,
        response_function=first_level_gaussian_surprise,
        hgf=hgf,
    )

    assert jnp.isclose(log_likelihood.sum(), 1141.0585)


def test_vectorized_continuous_logp(benchmark):
    """Test the vectorized version of the log-probability function."""
    timeseries = load_data("continuous")
    hgf = HGF(n_levels=2, model_type="continuous")

    # generate data with 2 parameters set
    input_data = np.array([timeseries] * 2)
    time_steps = np.ones(shape=input_data.shape)

    tonic_volatility_1 = -3.0
    tonic_volatility_2 = -3.0
    tonic_volatility_3 = jnp.nan
    input_precision = np.array(1e4)
    tonic_drift_1 = 0.0
    tonic_drift_2 = 0.0
    tonic_drift_3 = jnp.nan
    precision_1 = 1e4
    precision_2 = 1e1
    precision_3 = jnp.nan
    mean_1 = 1.0
    mean_2 = 0.0
    mean_3 = jnp.nan
    volatility_coupling_1 = 1.0
    volatility_coupling_2 = jnp.nan
    response_function_parameters = jnp.ones(2)

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
        jnp.zeros(2),
    )

    # create the vectorized version of the function
    vectorized_logp_two_levels = vmap(
        Partial(
            logp,
            hgf=hgf,
            response_function=first_level_gaussian_surprise,
        )
    )

    # model fit
    def get_likelihood():
        return vectorized_logp_two_levels(
            input_data=input_data,
            response_function_inputs=jnp.ones(2),
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

    # warmup
    _ = get_likelihood()

    # benchmark
    log_likelihoods = benchmark(get_likelihood)

    assert jnp.isclose(log_likelihoods.sum(), 2282.1165).all()


def test_vectorized_binary_logp(benchmark):
    """Test the vectorized version of the log-probability function."""
    u, y = load_data("binary")
    hgf = HGF(n_levels=2, model_type="binary")

    # generate data with 2 parameters set
    input_data = np.array([u] * 2)
    time_steps = np.ones(shape=input_data.shape)

    tonic_volatility_1 = -3.0
    tonic_volatility_2 = -3.0
    tonic_volatility_3 = jnp.nan
    input_precision = jnp.nan
    tonic_drift_1 = 0.0
    tonic_drift_2 = 0.0
    tonic_drift_3 = jnp.nan
    precision_1 = jnp.nan
    precision_2 = 1.0
    precision_3 = jnp.nan
    mean_1 = 0.5
    mean_2 = 0.0
    mean_3 = jnp.nan
    volatility_coupling_1 = 1.0
    volatility_coupling_2 = jnp.nan
    response_function_parameters = jnp.ones(2)

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
        jnp.zeros(2),
    )

    # create the vectorized version of the function
    vectorized_logp_two_levels = vmap(
        Partial(
            logp,
            hgf=hgf,
            response_function=binary_softmax_inverse_temperature,
        )
    )

    # model fit
    def get_likelihood():
        return vectorized_logp_two_levels(
            input_data=input_data,
            response_function_inputs=np.array([y] * 2),
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

    # warmup
    _ = get_likelihood()

    # benchmark
    log_likelihoods = benchmark(get_likelihood)

    assert jnp.isclose(log_likelihoods.sum(), -283.83594).all()


def test_hgf_logp():
    """Test the hgf_logp function used by Distribution Ops on three level models."""
    ##############################
    # Three-level continuous HGF #
    ##############################
    timeseries = load_data("continuous")
    continuous_hgf = HGF(n_levels=3, model_type="continuous")

    # generate data with 2 parameters set
    input_data = np.array([timeseries] * 2)
    time_steps = np.ones(shape=input_data.shape)

    # create the vectorized version of the function
    vectorized_logp_three_levels = vmap(
        Partial(
            logp,
            hgf=continuous_hgf,
            response_function=first_level_gaussian_surprise,
        )
    )

    sum_log_likelihoods, log_likelihoods = hgf_logp(
        tonic_volatility_1=-3.0,
        tonic_volatility_2=-3.0,
        tonic_volatility_3=-3.0,
        input_precision=1e4,
        tonic_drift_1=0.0,
        tonic_drift_2=0.0,
        tonic_drift_3=0.0,
        precision_1=1e4,
        precision_2=1e1,
        precision_3=1.0,
        mean_1=1.0,
        mean_2=0.0,
        mean_3=0.0,
        volatility_coupling_1=1.0,
        volatility_coupling_2=1.0,
        response_function_parameters=np.ones(2),
        response_function_inputs=np.ones(2),
        vectorized_logp=vectorized_logp_three_levels,
        input_data=input_data,
        time_steps=time_steps,
    )
    assert sum_log_likelihoods == log_likelihoods.sum()
    assert jnp.isclose(sum_log_likelihoods, 2269.6929).all()

    # test the gradient
    grad_logp = jit(
        grad(
            Partial(
                hgf_logp,
                vectorized_logp=vectorized_logp_three_levels,
                input_data=input_data,
                time_steps=time_steps,
                response_function_inputs=np.ones(2),
            ),
            argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            has_aux=True,
        ),
    )

    gradients, _ = grad_logp(
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        -3.0,
        -3.0,
        -3.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1e4,
        np.ones(2),
    )
    assert jnp.isclose(gradients[0], 0.09576362)
    assert jnp.isclose(gradients[1], -8.531818)
    assert jnp.isclose(gradients[2], -189.85936)

    ##########################
    # Three-level binary HGF #
    ##########################

    # Create the data (value and time vectors)
    u, y = load_data("binary")
    binary_hgf = HGF(n_levels=3, model_type="binary")

    # generate data with 2 parameters set
    input_data = np.array([u] * 2)
    response_function_inputs = np.array([y] * 2)
    time_steps = np.ones(shape=input_data.shape)

    # create the vectorized version of the function
    vectorized_logp_three_levels = vmap(
        Partial(
            logp,
            hgf=binary_hgf,
            response_function=binary_softmax_inverse_temperature,
        )
    )

    sum_log_likelihoods, log_likelihoods = hgf_logp(
        vectorized_logp=vectorized_logp_three_levels,
        tonic_volatility_1=np.nan,
        tonic_volatility_2=-2.0,
        tonic_volatility_3=-6.0,
        precision_1=np.nan,
        precision_2=1.0,
        precision_3=1.0,
        mean_1=0.5,
        mean_2=0.0,
        mean_3=0.0,
        volatility_coupling_1=1.0,
        volatility_coupling_2=1.0,
        response_function_parameters=np.array([1.0, 1.0]),
        input_data=input_data,
        response_function_inputs=response_function_inputs,
        time_steps=time_steps,
    )

    assert sum_log_likelihoods == log_likelihoods.sum()
    assert jnp.isclose(sum_log_likelihoods, -248.07889)

    # test the gradient
    grad_logp = jit(
        grad(
            Partial(
                hgf_logp,
                vectorized_logp=vectorized_logp_three_levels,
                input_data=input_data,
                time_steps=time_steps,
                response_function_inputs=response_function_inputs,
            ),
            argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            has_aux=True,
        ),
    )

    gradients, _ = grad_logp(
        0.5,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        -3.0,
        -2.0,
        -6.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        np.inf,
        np.ones(2),
    )
    assert jnp.isclose(gradients[0], 0.0)
    assert jnp.isclose(gradients[1], 1.9314771)
    assert jnp.isclose(gradients[2], 30.185408)


def test_pytensor_pointwise_logp():
    """Test the pytensor HGFPointwise op."""
    ##############
    # Binary HGF #
    ##############

    # Create the data (value and time vectors)
    u, y = load_data("binary")

    hgf_logp_op = HGFPointwise(
        input_data=u[np.newaxis, :],
        model_type="binary",
        n_levels=2,
        response_function=binary_softmax_inverse_temperature,
        response_function_inputs=y[np.newaxis, :],
    )

    logp = hgf_logp_op(
        tonic_volatility_1=np.inf,
        tonic_volatility_2=-6.0,
        tonic_volatility_3=np.inf,
        input_precision=np.inf,
        tonic_drift_1=0.0,
        tonic_drift_2=0.0,
        tonic_drift_3=np.inf,
        precision_1=0.0,
        precision_2=1e4,
        precision_3=np.inf,
        mean_1=np.inf,
        mean_2=0.5,
        mean_3=np.inf,
        volatility_coupling_1=1.0,
        volatility_coupling_2=np.inf,
        response_function_parameters=np.array([1.0]),
    ).eval()

    assert jnp.isclose(logp.sum(), -200.2442167699337)


def test_pytensor_logp():
    """Test the pytensor hgf_logp op."""
    ##################
    # Continuous HGF #
    ##################

    # Create the data (value and time vectors)
    timeseries = load_data("continuous")

    hgf_logp_op = HGFDistribution(
        input_data=timeseries[np.newaxis, :],
        model_type="continuous",
        n_levels=2,
        response_function=first_level_gaussian_surprise,
        response_function_inputs=None,
    )

    logp = hgf_logp_op(
        mean_1=np.array(1.0),
        mean_2=np.array(0.0),
        mean_3=np.array(0.0),
        precision_1=np.array(1e4),
        precision_2=np.array(1e1),
        precision_3=np.array(0.0),
        tonic_volatility_1=np.array(-3.0),
        tonic_volatility_2=np.array(-3.0),
        tonic_volatility_3=np.array(0.0),
        tonic_drift_1=np.array(0.0),
        tonic_drift_2=np.array(0.0),
        tonic_drift_3=np.array(0.0),
        volatility_coupling_1=np.array(1.0),
        volatility_coupling_2=np.array(0.0),
        input_precision=np.array(1e4),
        response_function_parameters=np.ones(1),
    ).eval()

    assert jnp.isclose(logp, 1141.05847168)

    ##############
    # Binary HGF #
    ##############

    # Create the data (value and time vectors)
    u, y = load_data("binary")

    hgf_logp_op = HGFDistribution(
        input_data=u[np.newaxis, :],
        model_type="binary",
        n_levels=2,
        response_function=binary_softmax_inverse_temperature,
        response_function_inputs=y[np.newaxis, :],
    )

    logp = hgf_logp_op(
        tonic_volatility_1=np.inf,
        tonic_volatility_2=-6.0,
        tonic_volatility_3=np.inf,
        input_precision=np.inf,
        tonic_drift_1=0.0,
        tonic_drift_2=0.0,
        tonic_drift_3=np.inf,
        precision_1=0.0,
        precision_2=1e4,
        precision_3=np.inf,
        mean_1=np.inf,
        mean_2=0.5,
        mean_3=np.inf,
        volatility_coupling_1=1.0,
        volatility_coupling_2=np.inf,
        response_function_parameters=np.array([1.0]),
    ).eval()

    assert jnp.isclose(logp, -200.2442167699337)


def test_pytensor_grad_logp_continuous(benchmark):
    """Test the pytensor gradient hgf_logp op."""
    ##################
    # Continuous HGF #
    ##################

    # Create the data (value and time vectors)
    timeseries = load_data("continuous")

    hgf_logp_grad_op = HGFLogpGradOp(
        model_type="continuous",
        input_data=timeseries[np.newaxis, :],
        n_levels=2,
        response_function=first_level_gaussian_surprise,
        response_function_inputs=None,
    )

    def get_grad_volatility_1():
        return hgf_logp_grad_op(
            tonic_volatility_1=-3.0,
            tonic_volatility_2=-3.0,
            input_precision=1e4,
            tonic_drift_1=0.0,
            tonic_drift_2=0.0,
            precision_1=1.0,
            precision_2=1.0,
            mean_1=1.0,
            mean_2=0.0,
            volatility_coupling_1=1.0,
        )[6].eval()

    # warmup
    _ = get_grad_volatility_1()

    # benchmark
    tonic_volatility_1 = benchmark(get_grad_volatility_1)

    assert jnp.isclose(tonic_volatility_1, -3.3479857)


def test_pytensor_grad_logp_binary(benchmark):
    """Test the pytensor gradient hgf_logp op."""
    ##############
    # Binary HGF #
    ##############

    # Create the data (value and time vectors)
    u, y = load_data("binary")

    hgf_logp_grad_op = HGFLogpGradOp(
        model_type="binary",
        input_data=u[np.newaxis, :],
        n_levels=2,
        response_function=binary_softmax_inverse_temperature,
        response_function_inputs=y[np.newaxis, :],
    )

    def get_grad_volatility_2():
        return hgf_logp_grad_op(
            tonic_volatility_1=jnp.inf,
            tonic_volatility_2=-6.0,
            input_precision=jnp.inf,
            tonic_drift_1=0.0,
            tonic_drift_2=0.0,
            precision_1=0.0,
            precision_2=1e4,
            mean_1=jnp.inf,
            mean_2=0.5,
            volatility_coupling_1=1.0,
        )[7].eval()

    # warmup
    _ = get_grad_volatility_2()

    # benchmark
    tonic_volatility_2 = benchmark(get_grad_volatility_2)

    assert jnp.isclose(tonic_volatility_2, 10.866466)


def test_pymc_sampling():
    """Test the pytensor hgf_logp op."""
    ##############
    # Continuous #
    ##############

    # Create the data (value and time vectors)
    timeseries = load_data("continuous")

    hgf_logp_op = HGFDistribution(
        n_levels=2,
        input_data=timeseries[np.newaxis, :],
        response_function=first_level_gaussian_surprise,
    )

    with pm.Model() as model:
        tonic_volatility_2 = pm.Uniform("tonic_volatility_2", -10, 0)

        pm.Potential(
            "hhgf_loglike",
            hgf_logp_op(
                tonic_volatility_2=tonic_volatility_2,
                input_precision=np.array(1e4),
            ),
        )

    initial_point = model.initial_point()

    pointslogs = model.point_logps(initial_point)
    assert pointslogs["tonic_volatility_2"] == -1.39
    assert pointslogs["hhgf_loglike"] == 1468.28

    with model:
        idata = pm.sample(chains=2, cores=1, tune=1000)

    assert -9.5 > az.summary(idata)["mean"].values[0] > -10.5
    assert az.summary(idata)["r_hat"].values[0] <= 1.02

    ##########
    # Binary #
    ##########

    # Create the data (value and time vectors)
    u, y = load_data("binary")

    hgf_logp_op = HGFDistribution(
        n_levels=2,
        model_type="binary",
        input_data=u[np.newaxis, :].repeat(2, axis=0),
        response_function=binary_softmax_inverse_temperature,
        response_function_inputs=y[np.newaxis, :].repeat(2, axis=0),
    )

    def logp(value, tonic_volatility_2):
        return hgf_logp_op(tonic_volatility_2=tonic_volatility_2)

    with pm.Model() as model:
        y_data = pm.Data("y_data", y[np.newaxis, :].repeat(2, axis=0))
        tonic_volatility_2 = pm.Normal("tonic_volatility_2", -11.0, 2, shape=(2,))
        pm.CustomDist("likelihood", tonic_volatility_2, logp=logp, observed=y_data)

    initial_point = model.initial_point()

    pointslogs = model.point_logps(initial_point)
    assert pointslogs["tonic_volatility_2"] == -3.22
    assert pointslogs["likelihood"] == -425.18

    with model:
        idata = pm.sample(chains=2, cores=1, tune=1000)

    assert -2 < round(az.summary(idata)["mean"].values[0]) < 0
