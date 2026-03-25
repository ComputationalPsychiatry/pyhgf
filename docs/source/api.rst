.. _api_ref:

.. currentmodule:: pyhgf


.. contents:: Table of Contents
   :depth: 5

API
###

Updates functions
*****************

Update functions are the heart of probabilistic networks as they shape the propagation of beliefs in the neural hierarchy. The library implements the standard variational updates for value and volatility coupling, as described in Weber et al. (2023).

The `updates` module contains the update functions used during the belief propagation. Update functions are available through three sub-modules, organized according to their functional roles. We usually dissociate the first updates, triggered top-down (from the leaves to the roots of the networks), that are prediction steps and recover the current state of inference. The second updates are the prediction error, signalling the divergence between the prediction and the new observation (for input nodes), or state (for state nodes). Interleaved with these steps are posterior update steps, where a node receives prediction errors from the child nodes and estimates new statistics.


Posterior updates
=================

Update the sufficient statistics of a state node after receiving prediction errors from children nodes. The prediction errors from all the children below the node should be computed before calling the posterior update step.


Categorical nodes
-----------------

.. currentmodule:: pyhgf.updates.posterior.categorical

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.categorical

    categorical_state_update

Continuous nodes
----------------

.. currentmodule:: pyhgf.updates.posterior.continuous.continuous_node_posterior_update_ehgf

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.continuous_node_posterior_update_ehgf

    continuous_node_posterior_update_ehgf

.. currentmodule:: pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded

    continuous_node_posterior_update_unbounded
    posterior_update_unbounded

.. currentmodule:: pyhgf.updates.posterior.continuous.posterior_update_mean_continuous_node

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.posterior_update_mean_continuous_node

    posterior_update_mean_continuous_node

.. currentmodule:: pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node

    posterior_update_precision_continuous_node

.. currentmodule:: pyhgf.updates.posterior.continuous.continuous_node_posterior_update

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.continuous_node_posterior_update

    continuous_node_posterior_update

Volatile nodes
--------------

.. currentmodule:: pyhgf.updates.posterior.volatile.posterior_update_value_level

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.posterior_update_value_level

    posterior_update_precision_value_level
    posterior_update_mean_value_level

.. currentmodule:: pyhgf.updates.posterior.volatile.posterior_update_volatility_level

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.posterior_update_volatility_level

    posterior_update_precision_volatility_level
    posterior_update_mean_volatility_level

.. currentmodule:: pyhgf.updates.posterior.volatile.volatile_node_posterior_update

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.volatile_node_posterior_update

    volatile_node_posterior_update

Exponential family
------------------

.. currentmodule:: pyhgf.updates.posterior.exponential

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.exponential

    posterior_update_exponential_family_dynamic

Prediction steps
================

Compute the expectation for future observation given the influence of parent nodes. The prediction step are executed for all nodes, top-down, before any observation.

Binary nodes
------------

.. currentmodule:: pyhgf.updates.prediction.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.binary

    binary_state_node_prediction

Continuous nodes
----------------

.. currentmodule:: pyhgf.updates.prediction.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.continuous

    predict_mean
    predict_precision
    continuous_node_prediction

Dirichlet processes
-------------------

.. currentmodule:: pyhgf.updates.prediction.dirichlet

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.dirichlet

    dirichlet_node_prediction

Prediction error steps
======================

Compute the value and volatility prediction errors of a given node. The prediction error can only be computed after the posterior update (or observation) of a given node.

Binary state nodes
------------------

.. currentmodule:: pyhgf.updates.prediction_error.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.binary

    binary_state_node_prediction_error
    binary_finite_state_node_prediction_error

Categorical state nodes
-----------------------

.. currentmodule:: pyhgf.updates.prediction_error.categorical

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.categorical

    categorical_state_prediction_error

Continuous state nodes
----------------------

.. currentmodule:: pyhgf.updates.prediction_error.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.continuous

    continuous_node_value_prediction_error
    continuous_node_volatility_prediction_error
    continuous_node_prediction_error

Dirichlet state nodes
---------------------

.. currentmodule:: pyhgf.updates.prediction_error.dirichlet

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.dirichlet

    dirichlet_node_prediction_error
    update_cluster
    create_cluster
    get_candidate
    likely_cluster_proposal
    clusters_likelihood

Exponential family
------------------

.. currentmodule:: pyhgf.updates.prediction_error.exponential

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.exponential

    prediction_error_update_exponential_family_fixed
    prediction_error_update_exponential_family_dynamic

Distribution
************

The Hierarchical Gaussian Filter as a PyMC distribution. This distribution can be
embedded in models using PyMC>=5.0.0.

.. currentmodule:: pyhgf.distribution

.. autosummary::
   :toctree: generated/pyhgf.distribution
   :nosignatures:

   logp
   hgf_logp
   HGFLogpGradOp
   HGFDistribution
   HGFPointwise

Model
*****

The main class is used to create a standard Hierarchical Gaussian Filter for binary or
continuous inputs, with two or three levels. This class wraps the previous JAX modules
and creates a standard node structure for these models.

.. currentmodule:: pyhgf.model

.. autosummary::
   :toctree: generated/pyhgf.model
   :nosignatures:

   HGF
   Network
   DeepNetwork
   add_volatile_node
   add_continuous_state
   add_binary_state
   add_ef_state
   add_categorical_state
   add_dp_state
   get_couplings
   update_parameters
   insert_nodes

Plots
*****

Plotting functionalities to visualize parameters trajectories and correlations after
observing new data. We are currently fully supporting Graphviz for network structure and
Matplotlib for time series visualization. NetworkX is also available for some functions. 

Matplotlib
==========

.. currentmodule:: pyhgf.plots.matplotlib

.. autosummary::
   :toctree: generated/pyhgf.plots.matplotlib

   plot_trajectories
   plot_correlations
   plot_nodes
   plot_samples

Graphviz
========

.. currentmodule:: pyhgf.plots.graphviz

.. autosummary::
   :toctree: generated/pyhgf.plots.graphviz

   plot_network
   plot_deep_network

Networkx
========

.. currentmodule:: pyhgf.plots.networkx

.. autosummary::
   :toctree: generated/pyhgf.plots.networkx

   plot_network

Response
********

A collection of response functions. A response function is simply a callable taking at
least the HGF instance as input after observation and returning surprise.

.. currentmodule:: pyhgf.response

.. autosummary::
   :toctree: generated/pyhgf.response

   first_level_gaussian_surprise
   total_gaussian_surprise
   first_level_binary_surprise
   binary_softmax
   binary_softmax_inverse_temperature

Utils
*****

Utilities for manipulating neural networks.

.. currentmodule:: pyhgf.utils

.. autosummary::
   :toctree: generated/pyhgf.utils

   add_edges
   add_parent
   beliefs_propagation
   fill_categorical_state_node
   get_input_idxs
   get_update_sequence
   learning
   list_branches
   remove_node
   sample_node_distribution
   sample
   set_coupling
   to_pandas

Math
****

Math functions and probability densities.

.. currentmodule:: pyhgf.math

.. autosummary::
   :toctree: generated/pyhgf.math

    MultivariateNormal
    Normal
    gaussian_predictive_distribution
    gaussian_density
    binary_surprise
    gaussian_surprise
    dirichlet_kullback_leibler
    binary_surprise_finite_precision
    sigmoid_inverse_temperature
    parametrised_sigmoid
    smoothed_rectangular