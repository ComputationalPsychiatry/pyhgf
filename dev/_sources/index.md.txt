<p align="center">
  <img src="https://raw.githubusercontent.com/ComputationalPsychiatry/pyhgf/master/docs/source/images/logo.png" alt="hgf" width="160">
</p>

<h1 align="center">PyHGF: A Neural Network Library for Predictive Coding</h1>

<p align="center">
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a>
  <a href="https://github.com/ComputationalPsychiatry/pyhgf/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="license"></a>
  <a href="https://codecov.io/gh/ComputationalPsychiatry/pyhgf"><img src="https://codecov.io/gh/ComputationalPsychiatry/pyhgf/branch/master/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="http://mypy-lang.org/"><img src="http://www.mypy-lang.org/static/mypy_badge.svg" alt="mypy"></a>
  <a href="https://badge.fury.io/py/pyhgf"><img src="https://badge.fury.io/py/pyhgf.svg" alt="pip"></a>
</p>

PyHGF is a Python library for creating and manipulating dynamic probabilistic networks for predictive coding. These networks approximate Bayesian inference by optimizing beliefs through the diffusion of predictions and precision-weighted prediction errors. The graph structure remains flexible during message-passing steps, allowing for dynamic adjustments. They can be used as a biologically plausible cognitive model in computational neuroscience or as a generalization of Bayesian filtering for designing efficient, modular decision-making agents. With the current version you can:

- Build arbitrarily sized networks with the **generalized Hierarchical Gaussian Filters** ([Weber et al., 2026](https://doi.org/10.7554/eLife.110174.1 ))
- Use **generalised Bayesian filtering** with distributions from the exponential family ([Mathys & Weber, 2020](https://doi.org/10.1007/978-3-030-64919-7_7))
- Define custom **planning** and **action selection** functions throught trajectories sampling (e.g. sophisticated inference, [Friston et al., 2020](https://doi.org/10.1162/neco_a_01351))
- Learn in **deep predictive coding networks** using a fast and scalable rethinking of prospective configuration ([Song et al., 2024](https://doi.org/10.1038/s41593-023-01514-1)) from closed-form updates and volatility learning.

The framework support both a JAX and Rust backend. It is designed to be adaptable to other algorithms. The core functions are differentiable and JIT-compiled where applicable. The library is optimized for modularity and ease of use, allowing seamless integration with other libraries in the ecosystem for Bayesian inference and optimization. You can find the method paper describing the toolbox [here](https://arxiv.org/abs/2410.09206).

* 📖 [API Documentation](https://computationalpsychiatry.github.io/pyhgf/api.html)  
* ✏️ [Tutorials and examples](https://computationalpsychiatry.github.io/pyhgf/learn.html)  

## Getting started

### Installation

The last official release can be downloaded from PIP:

`pip install pyhgf`

The current version under development can be installed from the master branch of the GitHub folder:

`pip install “git+https://github.com/ComputationalPsychiatry/pyhgf.git”`

### How does it work?

Dynamic networks are fully defined by the following variables:

* The attributes (dictionary) that store each node's states and parameters (e.g. value, precision, learning rates, volatility coupling, ...).
* The edges (tuple) that lists, for each node, the indexes of the parents and children.
* A set of update functions. An update function receive a network tuple and returns an updated network tuple.
* An update sequence (tuple) of callables (update functions) and pointers (nodes).

<img src="https://raw.githubusercontent.com/ComputationalPsychiatry/pyhgf/master/docs/source/images/graph_network.svg" align="center" alt="networks" style="width:100%; height:auto;">


You can find a deeper introduction to how to create and manipulate networks under the following link:

* 🎓 [Creating and manipulating networks of probabilistic nodes](https://computationalpsychiatry.github.io/pyhgf/notebooks/0.2-Creating_networks.html)  


### The Generalized Hierarchical Gaussian Filter

Generalized Hierarchical Gaussian Filters (gHGF) are specific instances of dynamic networks where node encodes a Gaussian distribution that can inherit its value (mean) and volatility (variance) from other nodes. The presentation of a new observation at the lowest level of the hierarchy (i.e., the input node) triggers a recursive update of the nodes' belief (i.e., posterior distribution) through top-down predictions and bottom-up precision-weighted prediction errors. The resulting probabilistic network operates as a Bayesian filter, and a response function can parametrize actions/decisions given the current beliefs. By comparing those behaviours with actual outcomes, a surprise function can be optimized over a set of free parameters. The Hierarchical Gaussian Filter for binary and continuous inputs was first described in Mathys et al. (2011, 2014), and later implemented in the Matlab HGF Toolbox (part of [TAPAS](https://translationalneuromodeling.github.io/tapas) (Frässle et al. 2021).

You can find a deeper introduction on how does the gHGF works under the following link:

* 🎓 [Introduction to the Hierarchical Gaussian Filter](https://computationalpsychiatry.github.io/pyhgf/notebooks/0.1-Theory.html#theory)  

#### Model fitting

Here we demonstrate how to fit forwards a two-level binary Hierarchical Gaussian filter. The input time series are binary observations using an associative learning task Iglesias et al. (2013).

<details>
<summary>Creating and fitting a binary HGF</summary>

```python
from pyhgf.model import Network
from pyhgf import load_data

# Load time series example data (observations, decisions)
u, y = load_data("binary")

# Create a two-level binary HGF from scratch
hgf = (
    Network()
    .add_nodes(kind="binary-state")
    .add_nodes(kind="continuous-state", value_children=0)
)

# add new observations
hgf.input_data(input_data=u)

# visualization of the belief trajectories
hgf.plot_trajectories();
```
</details>

![png](https://raw.githubusercontent.com/ComputationalPsychiatry/pyhgf/master/docs/source/images/trajectories.png)

#### Surprise

<details>
<summary>Computing the model's surprise</summary>

```python
from pyhgf.response import binary_softmax_inverse_temperature

# compute the model's surprise (-log(p)) 
# using the binary softmax with inverse temperature as the response model
surprise = hgf.surprise(
    response_function=binary_softmax_inverse_temperature,
    response_function_inputs=y,
    response_function_parameters=4.0
)
print(f"Sum of surprises = {surprise.sum()}")
```
</details>  

`Model's surprise = 138.8992462158203`

### Generalised Bayesian filtering

The framework supports online Bayesian filtering over any distribution in the [exponential family](https://en.wikipedia.org/wiki/Exponential_family) ([Mathys & Weber, 2020](https://doi.org/10.1007/978-3-030-64919-7_7)). Because these distributions share a common mathematical form, the posterior update reduces to a simple rule over the expected sufficient statistics $\xi$ and a pseudo-count $\nu$ that acts as an inverse learning rate. When the data source is non-stationary, $\nu$ can be kept fixed for a constant learning rate, or it can be dynamically adapted through a Hierarchical Gaussian Filter, giving the agent a volatility-sensitive learning rate that speeds up when the environment changes and slows down when it is stable.

<details>
<summary>Tracking a bivariate normal distribution</summary>

```python
from pyhgf.model import Network
import numpy as np

# Create a generalised filter for a 2D normal distribution
bivariate_normal = (
    Network()
    .add_nodes(
        kind="ef-state",
        nus=8.0,
        learning="generalised-filtering",
        distribution="multivariate-normal",
        dimension=2,
    )
    .input_data(input_data=spiral_data)
)
```
</details>

![gif](https://raw.githubusercontent.com/ComputationalPsychiatry/pyhgf/master/docs/source/images/multivariate_normal.gif)

* 🎓 [Generalised Bayesian filtering](https://computationalpsychiatry.github.io/pyhgf/notebooks/0.3-Generalised_filtering.html)

### Learning in deep predictive coding networks

The framework extends predictive coding to deep neural networks through *prospective configuration* ([Song et al., 2024](https://doi.org/10.1038/s41593-023-01514-1)): before updating any weight, the network first infers the most likely activations at every layer by settling prediction errors across the hierarchy, and only then adjusts the coupling strengths (weights). This two-phase infer-then-update cycle avoids the catastrophic interference that plagues standard backpropagation and naturally yields precision-weighted learning, where the balance of uncertainty between inputs and outputs controls the depth at which weights change.

<details>
<summary>Binary classification on a two-moons dataset</summary>

```python
from pyhgf.model import DeepNetwork
import jax.numpy as jnp

# Build a 2 → 16 → 16 → 1 (binary) predictive coding network
clf_net = (
    DeepNetwork(coupling_fn=jnp.tanh)
    .add_layer(size=1, kind="binary")
    .add_layer(size=16, tonic_volatility=-4.0)
    .add_layer(size=16, tonic_volatility=-4.0)
    .add_layer(size=2, add_constant_input=False, coupling_fn=lambda x: x)
    .weight_initialisation("he", seed=0)
)

# Train for 100 epochs using the Adam optimiser
for epoch in range(100):
    clf_net.fit(X_train, y_train, lr=0.1, optimizer="adam")
```
</details>

![gif](https://raw.githubusercontent.com/ComputationalPsychiatry/pyhgf/master/docs/source/images/two_moons_training.gif)

* 🎓 [Deep Bayesian predictive coding](https://computationalpsychiatry.github.io/pyhgf/notebooks/0.5-Learning.html)


## Acknowledgments

This implementation of the Hierarchical Gaussian Filter was inspired by the original [Matlab HGF Toolbox](https://translationalneuromodeling.github.io/tapas). A Julia implementation of the gHGF is also available [here](https://github.com/ComputationalPsychiatry/HGF.jl).

## References

1. Legrand, N., Weber, L., Waade, P. T., Daugaard, A. H. M., Khodadadi, M., Mikuš, N., & Mathys, C. (2024). pyhgf: A neural network library for predictive coding (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2410.09206  
2. Mathys, C. (2011). A Bayesian foundation for individual learning under uncertainty. In Frontiers in Human Neuroscience (Vol. 5). Frontiers Media SA. https://doi.org/10.3389/fnhum.2011.00039  
3. Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the hierarchical Gaussian filter. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00825  
4. Lilian Aline Weber, Peter Thestrup Waade, Nicolas Legrand, Anna Hedvig Møller, Klaas Enno Stephan, & Christoph Mathys. (2026). The generalized Hierarchical Gaussian Filter. eLife 15:RP110174. https://doi.org/10.7554/eLife.110174.1 
5. Frässle, S., Aponte, E. A., Bollmann, S., Brodersen, K. H., Do, C. T., Harrison, O. K., Harrison, S. J., Heinzle, J., Iglesias, S., Kasper, L., Lomakina, E. I., Mathys, C., Müller-Schrader, M., Pereira, I., Petzschner, F. H., Raman, S., Schöbi, D., Toussaint, B., Weber, L. A., … Stephan, K. E. (2021). TAPAS: An Open-Source Software Package for Translational Neuromodeling and Computational Psychiatry. In Frontiers in Psychiatry (Vol. 12). Frontiers Media SA. https://doi.org/10.3389/fpsyt.2021.680811  
6. Iglesias, S., Kasper, L., Harrison, S. J., Manka, R., Mathys, C., & Stephan, K. E. (2021). Cholinergic and dopaminergic effects on prediction error and uncertainty responses during sensory associative learning. In NeuroImage (Vol. 226, p. 117590). Elsevier BV. https://doi.org/10.1016/j.neuroimage.2020.117590 
7. Mathys, C., Weber, L. (2020). Hierarchical Gaussian Filtering of Sufficient Statistic Time Series for Active Inference. In: Verbelen, T., Lanillos, P., Buckley, C.L., De Boom, C. (eds) Active Inference. IWAI 2020. Communications in Computer and Information Science, vol 1326. Springer, Cham. https://doi.org/10.1007/978-3-030-64919-7_7
8. Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021). Sophisticated Inference. Neural Computation, 33(3), 713–763. https://doi.org/10.1162/neco_a_01351 
9. Song, Y., Millidge, B., Salvatori, T. et al. Inferring neural activity before plasticity as a foundation for learning beyond backpropagation. Nat Neurosci 27, 348–358 (2024). https://doi.org/10.1038/s41593-023-01514-1 

```{toctree}
---
hidden:
---
Learn <learn.md>
API <api.rst>
Cite <cite.md>
References <references.md>
```
