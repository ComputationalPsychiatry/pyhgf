.. _api_ref:

.. currentmodule:: pyhgf


.. contents:: Table of Contents
   :depth: 5

API
###

Datasets
********

Helpers to load the example datasets shipped with the package.

.. currentmodule:: pyhgf

.. autosummary::
   :toctree: generated/pyhgf

   load_data

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
    continuous_node_posterior_update_ehgf_mean_field

.. currentmodule:: pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded

    continuous_node_posterior_update_unbounded
    posterior_update_unbounded

.. currentmodule:: pyhgf.updates.posterior.continuous.posterior_update_mean_continuous_node

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.posterior_update_mean_continuous_node

    posterior_update_mean_continuous_node
    posterior_update_mean_continuous_node_mean_field

.. currentmodule:: pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.posterior_update_precision_continuous_node

    posterior_update_precision_continuous_node
    posterior_update_precision_continuous_node_mean_field
    posterior_update_precision_continuous_node_ehgf
    posterior_update_precision_continuous_node_ehgf_mean_field
    precision_update
    precision_update_mean_field
    precision_update_ehgf
    precision_update_ehgf_mean_field
    precision_update_missing_values

.. currentmodule:: pyhgf.updates.posterior.continuous.continuous_node_posterior_update

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous.continuous_node_posterior_update

    continuous_node_posterior_update
    continuous_node_posterior_update_mean_field

Volatile nodes
--------------

.. currentmodule:: pyhgf.updates.posterior.volatile.posterior_update_value_level

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.posterior_update_value_level

    posterior_update_precision_value_level
    posterior_update_mean_value_level
    posterior_update_precision_value_level_mean_field
    posterior_update_mean_value_level_mean_field

.. currentmodule:: pyhgf.updates.posterior.volatile.posterior_update_volatility_level

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.posterior_update_volatility_level

    posterior_update_precision_volatility_level
    posterior_update_mean_volatility_level
    posterior_update_precision_volatility_level_ehgf

.. currentmodule:: pyhgf.updates.posterior.volatile.volatile_node_posterior_update

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.volatile_node_posterior_update

    volatile_node_posterior_update
    volatile_node_posterior_update_mean_field
    volatile_node_volatility_posterior_update_standard

.. currentmodule:: pyhgf.updates.posterior.volatile.volatile_node_posterior_update_ehgf

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.volatile_node_posterior_update_ehgf

    volatile_node_posterior_update_ehgf

.. currentmodule:: pyhgf.updates.posterior.volatile.volatile_node_posterior_update_unbounded

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.volatile.volatile_node_posterior_update_unbounded

    volatile_node_posterior_update_unbounded

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
    predict_precision_mean_field
    continuous_node_prediction
    continuous_node_prediction_mean_field

Volatile nodes
--------------

.. currentmodule:: pyhgf.updates.prediction.volatile

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.volatile

    predict_precision_volatility_level
    predict_mean_value_level
    predict_precision_value_level
    predict_precision_value_level_mean_field
    volatile_node_prediction
    volatile_node_prediction_mean_field

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

Volatile state nodes
--------------------

.. currentmodule:: pyhgf.updates.prediction_error.volatile

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.volatile

    volatile_node_value_prediction_error
    volatile_node_volatility_prediction_error
    volatile_node_prediction_error

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

Learning
========

Weight learning rules applied after the prediction error step.

.. currentmodule:: pyhgf.updates.learning

.. autosummary::
   :toctree: generated/pyhgf.updates.learning

    learning_weights

Observations
============

Functions for setting observations and predictor values on input nodes.

.. currentmodule:: pyhgf.updates.observation

.. autosummary::
   :toctree: generated/pyhgf.updates.observation

    set_observation
    set_predictors

Vectorized updates
==================

Layer-wise vectorized implementations of the HGF update equations for deep predictive
coding networks. These functions operate on entire layers instead of individual nodes
and are used internally by :class:`pyhgf.model.DeepNetwork`.

Volatile layers
---------------

.. currentmodule:: pyhgf.updates.vectorized.volatile

.. autosummary::
   :toctree: generated/pyhgf.updates.vectorized.volatile

    vectorized_layer_prediction
    vectorized_layer_posterior_update
    vectorized_posterior_update_precision_value_level
    vectorized_posterior_update_mean_value_level
    vectorized_layer_value_prediction_error
    vectorized_layer_volatility_prediction_error
    vectorized_layer_volatility_posterior_standard
    vectorized_layer_volatility_posterior_ehgf
    vectorized_layer_volatility_posterior_unbounded
    vectorized_layer_prediction_error

Binary layers
-------------

.. currentmodule:: pyhgf.updates.vectorized.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.vectorized.binary

    vectorized_binary_prediction
    vectorized_binary_prediction_error

Categorical layers
------------------

.. currentmodule:: pyhgf.updates.vectorized.categorical

.. autosummary::
   :toctree: generated/pyhgf.updates.vectorized.categorical

    vectorized_categorical_prediction
    vectorized_categorical_prediction_error

Vectorized learning
-------------------

.. currentmodule:: pyhgf.updates.vectorized.learning

.. autosummary::
   :toctree: generated/pyhgf.updates.vectorized.learning

    vectorized_weight_gradient
    vectorized_weight_gradient_factors

Model
*****

The main classes used to create probabilistic networks for binary or continuous inputs.

.. currentmodule:: pyhgf.model

.. autosummary::
   :toctree: generated/pyhgf.model
   :nosignatures:

   Network
   DeepNetwork
   add_volatile_state
   add_continuous_state
   add_constant_state
   add_binary_state
   add_ef_state
   add_categorical_state
   add_dp_state
   get_couplings
   update_parameters
   insert_nodes

Weight transplant
=================

Converters that build a :class:`~pyhgf.model.DeepNetwork` reproducing the forward pass
of a trained Equinox module, translating its weights into PyHGF's layer layout.

.. currentmodule:: pyhgf.model.transplant

.. autosummary::
   :toctree: generated/pyhgf.model.transplant

   from_linear
   from_feedforward
   from_embedding

Declarative construction
========================

Build a :class:`~pyhgf.model.DeepNetwork` from serializable layer configurations, for
configuration-driven experiments and reproducible hyperparameter sweeps.

.. currentmodule:: pyhgf.model.builder

.. autosummary::
   :toctree: generated/pyhgf.model.builder
   :nosignatures:

   LayerConfig
   resolve_coupling_fn

Mixed pipelines
===============

Declare models as a tree of **parts** mixing learning PyHGF networks with frozen
calculations. Each part translates prediction errors backward through the tree, so
locally-learning networks and fixed operations compose into one model.

.. currentmodule:: pyhgf.model.hybrid

.. autosummary::
   :toctree: generated/pyhgf.model.hybrid
   :nosignatures:

   PCModule
   DeepNetworkAdapter
   EquinoxAdapter
   PCSequential
   Residual
   linear_adapter
   layer_norm_adapter
   gelu_adapter

Transformer
===========

A GPT-style Transformer assembled from mixed-pipeline parts, where any slot can hold a
frozen calculation or a learning PyHGF network.

.. currentmodule:: pyhgf.model.transformer

.. autosummary::
   :toctree: generated/pyhgf.model.transformer
   :nosignatures:

   MultiHeadAttention
   HybridGPT
   hybrid_from_gpt

Fused pipeline execution
========================

Run a part tree as a single compiled program per training step: the forward walk, the
output error, and every local learning step are staged into one graph with all part
state passed explicitly.

.. currentmodule:: pyhgf.model.fused

.. autosummary::
   :toctree: generated/pyhgf.model.fused
   :nosignatures:

   FusedPipeline
   step_report

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
   plot_layers

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
   predict_step
   remove_node
   sample_node_distribution
   sample
   set_coupling
   to_pandas

Weight initialisation
=====================

Initialisation strategies for coupling weight matrices in predictive-coding networks.

.. currentmodule:: pyhgf.utils.weight_initialisation

.. autosummary::
   :toctree: generated/pyhgf.utils.weight_initialisation

   xavier_init
   he_init
   orthogonal_init
   sparse_init

Sampling
========

Low-level helpers used when sampling from a network's generative distribution.

.. currentmodule:: pyhgf.utils.sample

.. autosummary::
   :toctree: generated/pyhgf.utils.sample

   single_sample

Vectorized belief propagation
=============================

Scan-based belief propagation used internally by :class:`pyhgf.model.DeepNetwork`.

.. currentmodule:: pyhgf.utils.vectorized_belief_propagation

.. autosummary::
   :toctree: generated/pyhgf.utils.vectorized_belief_propagation

   propagation_step
   prediction_pass
   run_scan
   prediction_sweep
   update_sweep
   learn_sweep
   input_prediction_error
   sample_step
   batch_step
   apply_confidence_increments
   batched_prediction_pass
   batched_prediction_states

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
    lambert_w0

Typing
******

Named tuples and PyTree containers describing the structure and state of
probabilistic networks.

Network structure
=================

.. currentmodule:: pyhgf.typing.typing

.. autosummary::
   :toctree: generated/pyhgf.typing.typing

   AdjacencyLists
   UpdateSequence
   LearningSequence

Vectorized networks
===================

.. currentmodule:: pyhgf.typing.vectorised

.. autosummary::
   :toctree: generated/pyhgf.typing.vectorised

   LayerParams
   LayerState
   Layer
   LayerStack
   Network
   stack_layers
