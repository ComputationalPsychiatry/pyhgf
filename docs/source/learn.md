```{toctree}
:maxdepth: 2
:hidden:
:caption: Theory

notebooks/0.1-Theory.ipynb
notebooks/0.2-Creating_networks.ipynb
notebooks/0.3-Generalised_filtering.ipynb
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: The Hierarchical Gaussian Filter

notebooks/1.1-Binary_HGF.ipynb
notebooks/1.2-Categorical_HGF.ipynb
notebooks/1.3-Continuous_HGF.ipynb
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Tutorials

notebooks/2-Using_custom_response_functions.ipynb
notebooks/3-Multilevel_HGF.ipynb
notebooks/4-Parameter_recovery.ipynb
notebooks/5-Non_linear_value_coupling
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Use cases

notebooks/Example_1_Heart_rate_variability.ipynb
notebooks/Example_2_Input_node_volatility_coupling.ipynb
notebooks/Example_3_Multi_armed_bandit.ipynb
notebooks/Example_4_Causal.ipynb
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Exercises

notebooks/Exercise_1_Introduction_to_the_generalised_hierarchical_gaussian_filter.ipynb
notebooks/Exercise_2_Bayesian_reinforcement_learning.ipynb
```

# Learn

In this section, you can find tutorial notebooks that describe the internals of pyhgf, the theory behind the Hierarchical Gaussian filter, and step-by-step application and use cases of the model. At the beginning of every tutorial, you will find a badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/0.1-Creating_networks.ipynb) to run the notebook interactively in a Google Colab session.

## Theory

::::{grid} 1 1 2 3
:gutter: 1

:::{grid-item-card}  An introduction to the Hierarchical Gaussian Filter
:link: theory
:link-type: ref
:img-top: ./images/trajectories.png

How the generative model of the Hierarchical Gaussian filter can be turned into update functions that update nodes through value and volatility coupling?
:::

:::{grid-item-card}  Creating and manipulating networks of probabilistic nodes
:link: probabilistic_networks
:link-type: ref
:img-top: ./images/graph_network.svg

How to create and manipulate a network of probabilistic nodes for reinforcement learning? Working at the intersection of graphs, neural networks and probabilistic frameworks.
:::

:::{grid-item-card}  Generalised Bayesian filtering
:link: generalised_filtering
:link-type: ref
:img-top: ./images/multivariate_hgf.gif


Predict, filter and smooth any distribution from the exponential family using generalisations of the Hierarchical Gaussian Filter.
:::
::::


## The Hierarchical Gaussian Filter

::::{grid} 1 1 2 3
:gutter: 1

:::{grid-item-card}  The binary Hierarchical Gaussian Filter
:link: binary_hgf
:link-type: ref
:img-top: ./images/binary.svg

Introducing with example the binary Hierarchical Gaussian filter and its applications to reinforcement learning.
:::

:::{grid-item-card}  The categorical Hierarchical Gaussian Filter
:link: categorical_hgf
:link-type: ref
:img-top: ./images/categorical.svg

The categorical Hierarchical Gaussian Filter is a generalisation of the binary HGF to handle categorical distribution with and without transition probabilities.
:::

:::{grid-item-card}  The continuous Hierarchical Gaussian Filter
:link: continuous_hgf
:link-type: ref
:img-top: ./images/continuous.svg

Introducing with example the continuous Hierarchical Gaussian filter and its applications to signal processing.
:::

::::

## Tutorials

Advanced customisation of predictive coding neural networks and Bayesian modelling for group studies. 

::::{grid} 1 1 2 3
:gutter: 1

:::{grid-item-card} Using custom response functions
:link: custom_response_functions
:link-type: ref
:img-top: ./images/response_models.png

How to adapt any model to specific behaviours and experimental design by using custom response functions.
:::

:::{grid-item-card} Embedding the Hierarchical Gaussian Filter in a Bayesian network for multilevel inference
:link: multilevel_hgf
:link-type: ref
:img-top: ./images/multilevel-hgf.png

How to use any model as a distribution to perform hierarchical inference at the group level.
:::

:::{grid-item-card} Parameter recovery, prior and posterior predictive sampling
:link: parameters_recovery
:link-type: ref
:img-top: ./images/parameter_recovery.png

Recovering parameters from the generative model and using the sampling functionalities to estimate prior and posterior uncertainties.
:::

:::{grid-item-card} Non-linear value coupling
:link: non_linear_coupling
:link-type: ref
:img-top: ./images/non_linear_coupling.png

Recovering parameters from the generative model and using the sampling functionalities to estimate prior and posterior uncertainties.
:::
::::

## Use cases

Examples of possible applications and extensions of the standards Hierarchical Gaussian Filters to more complex experimental designs

::::{grid} 1 1 2 3
:gutter: 1

:::{grid-item-card}  Inferring cardiac beliefs using Bayesian filtering
:link: example_1
:link-type: ref

Application of continuous Bayesian filtering to cardiac physiological recordings to infer interoceptive beliefs and their volatility.

:::

:::{grid-item-card}  Value and volatility coupling with an input node
:link: example_2
:link-type: ref
:img-top: ./images/input_mean_precision.png

Dynamic inference over both the mean and variance of a normal distribution.
:::

:::{grid-item-card}  Multi-armed bandit task with independent reward and punishments
:link: example_3
:link-type: ref
:img-top: ./images/multiarmedbandittask.png

A generalisation of the binary Hierarchical Gaussian Filter to multiarmed bandit where the probabilities of the outcomes are evolving independently.
:::

:::{grid-item-card}  Causal discovery for predictive coding
:link: example_4
:link-type: ref
:img-top: ./images/causal_discovery.png

Deriving a causal discovery prediction error to track the causal coupling strength between two random variable over time.
:::

::::

## Exercises

Hand-on exercises for theComputational Psychiatry Course (Zurich) to build intuition around the generalised Hierarchical Gaussian Filter, how to create and manipulate probabilistic networks, design an agent to perform a reinforcement learning task and use MCMC sampling for parameter inference and model comparison—about 4 hours.


::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} Introduction to the Generalised Hierarchical Gaussian Filter
:link: hgf_exercises_1
:link-type: ref
:img-top: ./images/cpc_tutorial_1.png

Theoretical introduction to the generative model of the generalised Hierarchical Gaussian Filter and presentation of the update functions (i.e. the first inversion of the model). 
:::

:::{grid-item-card} Applying the Hierarchical Gaussian Filter to reinforcement learning
:link: hgf_exercises_2
:link-type: ref
:img-top: ./images/cpc_tutorial_2.png

Practical application of the generalised Hierarchical Gaussian Filter to reinforcement learning problems and estimation of parameters through MCMC sampling (i.e. the second inversion of the model). 
:::

::::
