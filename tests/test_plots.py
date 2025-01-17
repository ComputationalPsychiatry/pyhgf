# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np

from pyhgf import load_data
from pyhgf.model import HGF, Network


def test_plotting_functions():
    # Read USD-CHF data
    timeserie = load_data("continuous")

    ##############
    # Continuous #
    # ------------

    # Set up standard 2-level HGF for continuous inputs
    two_level_continuous = HGF(
        n_levels=2,
        model_type="continuous",
        initial_mean={"1": 1.04, "2": 1.0},
        initial_precision={"1": 1e4, "2": 1e1},
        tonic_volatility={"1": -13.0, "2": -2.0},
        tonic_drift={"1": 0.0, "2": 0.0},
        volatility_coupling={"1": 1.0},
    ).input_data(input_data=timeserie)

    # plot trajectories
    two_level_continuous.plot_trajectories(show_total_surprise=True)

    # plot correlations
    two_level_continuous.plot_correlations()

    # plot node structures
    two_level_continuous.plot_network()

    # plot nodes
    two_level_continuous.plot_nodes(
        node_idxs=2,
        show_posterior=True,
    )

    # Set up standard 3-level HGF for continuous inputs
    three_level_continuous = HGF(
        n_levels=3,
        model_type="continuous",
        initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
        initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
        tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
        tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": 1.0, "2": 1.0},
    ).input_data(input_data=timeserie)

    # plot trajectories
    three_level_continuous.plot_trajectories(show_total_surprise=True)

    # plot correlations
    three_level_continuous.plot_correlations()

    # plot node structures
    three_level_continuous.plot_network()

    # plot nodes
    three_level_continuous.plot_nodes(
        node_idxs=2,
        show_posterior=True,
    )

    ##########
    # Binary #
    # --------

    # Read binary input
    u, _ = load_data("binary")

    two_level_binary_hgf = HGF(
        n_levels=2,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5},
        initial_precision={"1": 0.0, "2": 1e4},
        tonic_volatility={"1": None, "2": -6.0},
        tonic_drift={"1": None, "2": 0.0},
        volatility_coupling={"1": None},
        eta0=0.0,
        eta1=1.0,
        binary_precision=jnp.inf,
    ).input_data(u)

    # plot trajectories
    two_level_binary_hgf.plot_trajectories(show_total_surprise=True)

    # plot correlations
    two_level_binary_hgf.plot_correlations()

    # plot node structures
    two_level_binary_hgf.plot_network()

    # plot node structures
    two_level_binary_hgf.plot_nodes(
        node_idxs=1,
        show_posterior=True,
    )

    three_level_binary_hgf = HGF(
        n_levels=3,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5, "3": 0.0},
        initial_precision={"1": 0.0, "2": 1e4, "3": 1e1},
        tonic_volatility={"1": None, "2": -6.0, "3": -2.0},
        tonic_drift={"1": None, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": None, "2": 1.0},
        eta0=0.0,
        eta1=1.0,
        binary_precision=jnp.inf,
    ).input_data(u)

    # plot trajectories
    three_level_binary_hgf.plot_trajectories(show_total_surprise=True)

    # plot correlations
    three_level_binary_hgf.plot_correlations()

    # plot node structures
    three_level_binary_hgf.plot_network()

    # plot node structures
    three_level_binary_hgf.plot_nodes(
        node_idxs=2,
        show_posterior=True,
    )

    #############
    # Categorical
    # -----------

    # generate some categorical inputs data
    input_data = np.array(
        [np.random.multinomial(n=1, pvals=[0.1, 0.2, 0.7]) for _ in range(10)],
        dtype=float,
    )

    # create the categorical HGF
    categorical_hgf = Network().add_nodes(
        kind="categorical-state",
        node_parameters={
            "n_categories": 3,
            "binary_parameters": {"tonic_volatility_2": -2.0},
        },
    )

    # fitting the model forwards
    categorical_hgf.input_data(input_data=input_data)

    # plot node structures
    categorical_hgf.plot_network()
