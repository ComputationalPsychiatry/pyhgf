# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, random
from jax._src.typing import Array as KeyArray
from jax.lax import cond
from jax.scipy.stats.norm import pdf
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.math import Normal
from pyhgf.typing import Attributes, Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def dirichlet_node_prediction_error(
    edges: Edges,
    attributes: Dict,
    node_idx: int,
    **args,
) -> Attributes:
    """Prediction error and update the child networks of a Dirichlet process node.

    When receiving a new input, this node chose to either:
    1. Allocate the value to a pre-existing cluster.
    2. Create a new cluster.

    The network always contains a temporary branch as the new cluster candidate. This
    branch is parametrized under the new observation to assess its likelihood and the
    previous clusters' likelihood.

    Parameters
    ----------
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the Dirichlet process input node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    values = attributes[node_idx]["mean"]  # the input value
    alpha = attributes[node_idx]["alpha"]  # the concentration parameter
    n_total = attributes[node_idx]["n_total"]  # total number of observations
    n = attributes[node_idx]["n"]  # number of observations per cluster
    sensory_precision = attributes[node_idx][
        "sensory_precision"
    ]  # number of observations per cluster

    # likelihood of the current observation under existing clusters
    # -------------------------------------------------------------
    cluster_ll = clusters_likelihood(
        value=values,
        expected_mean=attributes[node_idx]["expected_means"],
        expected_sigma=attributes[node_idx]["expected_sigmas"],
    )

    # set the likelihood to 0 for inactive clusters
    cluster_ll *= attributes[node_idx]["activated"]

    # likelihood of the current observation under the best candidate cluster
    # ----------------------------------------------------------------------

    # find the best cluster candidate given the new observation
    candidate_mean, candidate_sigma = get_candidate(
        value=values,
        sensory_precision=sensory_precision,
        expected_mean=attributes[node_idx]["expected_means"],
        expected_sigma=attributes[node_idx]["expected_sigmas"],
    )

    # get the likelihood under this candidate
    candidate_ll = clusters_likelihood(
        value=values,
        expected_mean=candidate_mean,
        expected_sigma=candidate_sigma,
    )

    # DP step: compare the likelihood of existing cluster with a new cluster
    # ----------------------------------------------------------------------

    # probability of being assigned to a pre-existing cluster
    cluster_ll *= n / (alpha + n_total)

    # probability to draw a new cluster
    candidate_ll *= alpha / (alpha + n_total)

    best_val = jnp.max(cluster_ll)

    # set all cluster to non-observed by default
    for parent_idx in edges[node_idx].value_parents:  # type:ignore
        attributes[parent_idx]["observed"] = 0

    # get the index of the cluster (!= the node index)
    # depending on whether a new cluster is created or updated
    cluster_idx = jnp.where(
        best_val >= candidate_ll,
        jnp.argmax(cluster_ll),
        attributes[node_idx]["n_active_cluster"],
    )

    update_fn = Partial(
        update_cluster,
        edges=edges,
        node_idx=node_idx,
    )

    create_fn = Partial(
        create_cluster,
        edges=edges,
        node_idx=node_idx,
    )

    # apply either cluster update or cluster creation
    operands = attributes, cluster_idx, values, (candidate_mean, candidate_sigma)

    attributes = cond(best_val >= candidate_ll, update_fn, create_fn, operands)

    attributes[node_idx]["n_total"] += 1

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def update_cluster(operands: Tuple, edges: Edges, node_idx: int) -> Attributes:
    """Update an existing cluster.

    Parameters
    ----------
    operands :
        Non-static parameters.
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    node_idx :
        Pointer to the Dirichlet process input node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    attributes, cluster_idx, value, _ = operands

    # activate the corresponding branch and pass the value
    for i, value_parent_idx in enumerate(edges[node_idx].value_parents):  # type: ignore

        attributes[value_parent_idx]["observed"] = jnp.where(cluster_idx == i, 1.0, 0.0)
        attributes[value_parent_idx]["mean"] = value

    attributes[node_idx]["n"] = (
        attributes[node_idx]["n"]
        .at[cluster_idx]
        .set(attributes[node_idx]["n"][cluster_idx] + 1.0)
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def create_cluster(operands: Tuple, edges: Edges, node_idx: int) -> Attributes:
    """Create a new cluster.

    Parameters
    ----------
    operands :
        Non-static parameters.
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    node_idx :
        Pointer to the Dirichlet process input node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    attributes, cluster_idx, value, (candidate_mean, candidate_sigma) = operands

    # creating a new cluster
    attributes[node_idx]["activated"] = (
        attributes[node_idx]["activated"].at[cluster_idx].set(1)
    )

    for i, value_parent_idx in enumerate(edges[node_idx].value_parents):  # type: ignore

        attributes[value_parent_idx]["observed"] = 0.0
        attributes[value_parent_idx]["mean"] = value

        # initialize the new cluster using candidate values
        attributes[value_parent_idx]["xis"] = jnp.where(
            cluster_idx == i,
            Normal().sufficient_statistics_from_parameters(
                mean=candidate_mean, variance=candidate_sigma**2
            ),
            attributes[value_parent_idx]["xis"],
        )

    attributes[node_idx]["n"] = attributes[node_idx]["n"].at[cluster_idx].set(1.0)
    attributes[node_idx]["n_active_cluster"] += 1

    return attributes


@jit
def get_candidate(
    value: float,
    sensory_precision: float,
    expected_mean: ArrayLike,
    expected_sigma: ArrayLike,
    n_samples: int = 20_000,
) -> Tuple[float, float]:
    """Find the best cluster candidate given previous clusters and an input value.

    Parameters
    ----------
    value :
        The new observation.
    sensory_precision :
        The expected precision of the new observation.
    expected_mean :
        The mean of the existing clusters.
    expected_sigma :
        The standard deviation of the existing clusters.
    n_samples :
        The number of samples that should be simulated.

    Returns
    -------
    mean :
        The mean of the new candidate cluster.
    sigma :
        The standard deviation of the new candidate cluster.

    """
    # sample n likely clusters given the base distribution priors
    mus, sigmas, weights = likely_cluster_proposal(
        mean_mu_G0=0.0,
        sigma_mu_G0=10.0,
        sigma_pi_G0=3.0,
        expected_mean=expected_mean,
        expected_sigma=expected_sigma,
        key=random.key(42),
        n_samples=n_samples,
    )

    # 1 - Likelihood of the new observation under each sampled cluster
    # ----------------------------------------------------------------
    ll_value = pdf(value, mus, sigmas)
    ll_value /= ll_value.sum()  # normalize the weights

    # 2- re-scale the weights using expected precision
    # ------------------------------------------------
    weights *= ll_value**sensory_precision

    # only use the 1000 best candidates for inference
    idxs = jnp.argsort(weights)
    mus, sigmas, weights = (
        mus[idxs][-1000:],
        sigmas[idxs][-1000:],
        weights[idxs][-1000:],
    )

    # 3 - estimate new mean and standard deviation using the weigthed mean
    # --------------------------------------------------------------------
    mean = jnp.average(mus, weights=weights)
    sigma = jnp.average(sigmas, weights=weights)

    return mean, sigma


@partial(jit, static_argnames=("n_samples"))
def likely_cluster_proposal(
    mean_mu_G0: float,
    sigma_mu_G0: float,
    sigma_pi_G0: float,
    expected_mean=ArrayLike,
    expected_sigma=ArrayLike,
    key: KeyArray = random.key(42),
    n_samples: int = 20_000,
) -> Tuple[Array, Array, Array]:
    """Sample likely new belief distributions given pre-existing clusters.

    Parameters
    ----------
    mean_mu_G0 :
        The mean of the mean of the base distribution.
    sigma_mu_G0 :
        The standard deviation of mean of the base distribution.
    sigma_pi_G0 :
        The standard deviation of the standard deviation of the base distribution.
    expected_mean :
        Pre-existing clusters means.
    expected_sigma :
        Pre-existing clusters standard deviation.
    key :
        Random state.
    n_samples :
        The number of samples used during the simulations.

    Returns
    -------
    new_mu :
        A vector of means candidates.
    new_sigma :
        A vector of standard deviation candidates.
    weights :
        Weigths for each cluster candidate under pre-existing cluster (irrespective of
        new observations).

    """
    # sample new candidate for cluster means
    key, use_key = random.split(key)
    new_mu = sigma_mu_G0 * random.normal(use_key, (n_samples,)) + mean_mu_G0

    # sample new candidate for cluster standard deviation
    key, use_key = random.split(key)
    new_sigma = jnp.abs(random.normal(use_key, (n_samples,)) * sigma_pi_G0)

    # 1 - Cluster specificity
    # -----------------------
    # this cluster should explain new dimensions, not explained by other clusters

    # evidence for pre-existing clusters
    pre_existing_likelihood = jnp.zeros(n_samples)
    for mu_i, sigma_i in zip(expected_mean, expected_sigma):
        pre_existing_likelihood += pdf(new_mu, mu_i, sigma_i)

    # evidence for the new cluster proposal
    new_likelihood = pdf(new_mu, new_mu, new_sigma)

    # standardize the measure of cluster specificity (ratio)
    ratio = new_likelihood / (new_likelihood + pre_existing_likelihood)
    ratio -= ratio.min()
    ratio /= ratio.max()
    weights = ratio

    # 2 - Cluster isolation
    # ---------------------
    # this cluster should not try to explain what was already explained

    # (pre-existing cluster) / (pre-existing cluster + new cluster)
    cluster_isolation = jnp.ones(n_samples)
    for mu_i, sigma_i in zip(expected_mean, expected_sigma):
        ratio = pdf(mu_i, mu_i, sigma_i) / (
            pdf(mu_i, mu_i, sigma_i) + pdf(mu_i, new_mu, new_sigma)
        )
        cluster_isolation *= ratio
    cluster_isolation -= cluster_isolation.min()
    cluster_isolation /= cluster_isolation.max()

    weights *= cluster_isolation

    # 3 - Spread of the cluster
    # -------------------------
    # large clusters should be favored over small clusters
    cluster_spread = pdf(1 / (new_sigma**2), 0.0, 5.0)
    cluster_spread -= cluster_spread.min()
    cluster_spread /= cluster_spread.max()
    weights *= cluster_spread

    return new_mu, new_sigma, weights


def clusters_likelihood(
    value: float,
    expected_mean: ArrayLike,
    expected_sigma: ArrayLike,
) -> ArrayLike:
    """Likelihood of a parametrized candidate under the new observation.

    Parameters
    ----------
    value :
        The new observation.
    expected_mean :
        Pre-existing clusters means.
    expected_sigma :
        Pre-existing clusters standard deviation.

    Returns
    -------
    likelihood :
        The probability of observing the value under each cluster.

    """
    return pdf(value, expected_mean, expected_sigma)
