# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

from pyhgf.math import MultivariateNormal, Normal
from pyhgf.typing import AdjacencyLists
from pyhgf.utils import fill_categorical_state_node

if TYPE_CHECKING:
    from pyhgf.model import Network


def add_continuous_state(
    network: Network,
    n_nodes: int,
    value_parents,
    volatility_parents,
    value_children,
    volatility_children,
    node_parameters: Dict,
    additional_parameters: Dict,
    coupling_fn: Tuple[Optional[Callable], ...],
):
    """Add continuous state node(s) to a network."""
    node_type = 2

    default_parameters = {
        "mean": 0.0,
        "expected_mean": 0.0,
        "precision": 1.0,
        "expected_precision": 1.0,
        "volatility_coupling_children": volatility_children[1],
        "volatility_coupling_parents": volatility_parents[1],
        "value_coupling_children": value_children[1],
        "value_coupling_parents": value_parents[1],
        "tonic_volatility": -4.0,
        "tonic_drift": 0.0,
        "autoconnection_strength": 1.0,
        "observed": 1,
        "temp": {
            "effective_precision": 0.0,
            "value_prediction_error": 0.0,
            "volatility_prediction_error": 0.0,
        },
    }

    node_parameters = update_parameters(
        node_parameters, default_parameters, additional_parameters
    )

    network = insert_nodes(
        network=network,
        n_nodes=n_nodes,
        node_type=node_type,
        node_parameters=node_parameters,
        value_parents=value_parents,
        volatility_parents=volatility_parents,
        value_children=value_children,
        volatility_children=volatility_children,
        coupling_fn=coupling_fn,
    )

    return network


def add_binary_state(
    network: Network,
    n_nodes: int,
    value_parents,
    volatility_parents,
    value_children,
    volatility_children,
    node_parameters: Dict,
    additional_parameters: Dict,
):
    """Add binary state node(s) to a network."""
    # define the type of node that is created
    node_type = 1

    default_parameters = {
        "observed": 1,
        "mean": 0,
        "expected_mean": 0.5,
        "precision": 1.0,
        "expected_precision": 1.0,
        "value_coupling_parents": value_parents[1],
        "temp": {
            "value_prediction_error": 0.0,
        },
    }

    node_parameters = update_parameters(
        node_parameters, default_parameters, additional_parameters
    )

    network = insert_nodes(
        network=network,
        n_nodes=n_nodes,
        node_type=node_type,
        node_parameters=node_parameters,
        value_parents=value_parents,
        volatility_parents=volatility_parents,
        value_children=value_children,
        volatility_children=volatility_children,
    )

    return network


def add_ef_state(
    network: Network,
    n_nodes: int,
    node_parameters: dict,
    additional_parameters: dict,
    value_children: tuple = (None, None),
):
    """Add exponential family state node(s) to a network."""
    node_type = 3

    default_parameters = {
        "dimension": 1,
        "distribution": "normal",
        "learning": "generalised-filtering",
        "nus": 3.0,
        "xis": jnp.array([0.0, 1.0]),
        "observed": 1,
    }

    node_parameters = update_parameters(
        node_parameters, default_parameters, additional_parameters
    )

    # the size of the sufficient statistics vector of a multivariate normal
    # distribution is given by d + d(d+1) / 2, where d is the dimension
    d = node_parameters["dimension"]
    n_suff_stats = d + d * (d + 1) // 2
    node_parameters["mean"] = jnp.zeros(d) if d > 1 else 0.0
    node_parameters["observation_ss"] = jnp.zeros(n_suff_stats)
    if node_parameters["distribution"] == "normal":
        node_parameters["xis"] = jnp.array([0.0, 1.0])
    elif node_parameters["distribution"] == "multivariate-normal":
        node_parameters["xis"] = (
            MultivariateNormal.sufficient_statistics_from_parameters(
                mean=jnp.zeros(d), covariance=jnp.identity(d)
            )
        )
    network = insert_nodes(
        network=network,
        n_nodes=n_nodes,
        node_type=node_type,
        node_parameters=node_parameters,
        value_children=value_children,
    )

    # loop over the indexes of nodes created in the previous step
    for node_idx in range(network.n_nodes - 1, network.n_nodes - n_nodes - 1, -1):

        # create the sufficient statistic function and store in the side parameters
        if network.attributes[node_idx]["distribution"] == "normal":
            sufficient_stats_fn = Normal().sufficient_statistics_from_observations
        elif network.attributes[node_idx]["distribution"] == "multivariate-normal":
            sufficient_stats_fn = (
                MultivariateNormal().sufficient_statistics_from_observations
            )
        else:
            raise ValueError(
                "The distribution should be either 'normal' or 'multivariate-normal'."
            )

        # add the sufficient statistics function in the side parameters
        network.additional_parameters.setdefault(node_idx, {})[
            "sufficient_stats_fn"
        ] = sufficient_stats_fn

        if "hgf" in network.attributes[node_idx]["learning"]:

            # create a collection of continuous state nodes
            # to track the sufficient statistics of the implied distribution
            for i in range(n_suff_stats):
                network.add_nodes(value_children=node_idx)
                network.add_nodes(value_children=network.n_nodes - 1)
                if (
                    "-2" in network.attributes[node_idx]["learning"]
                    or "-3" in network.attributes[node_idx]["learning"]
                ):
                    network.add_nodes(volatility_children=network.n_nodes - 1)
                if "-3" in network.attributes[node_idx]["learning"]:
                    network.add_nodes(volatility_children=network.n_nodes - 1)

        network.attributes[node_idx].pop("distribution")
        network.attributes[node_idx].pop("learning")

    return network


def add_categorical_state(
    network: Network, n_nodes: int, node_parameters: Dict, additional_parameters: Dict
) -> Network:
    """Add categorical state node(s) to a network."""
    node_type = 5

    if "n_categories" in node_parameters:
        n_categories = node_parameters["n_categories"]
    elif "n_categories" in additional_parameters:
        n_categories = additional_parameters["n_categories"]
    else:
        n_categories = 4
    binary_parameters = {
        "n_categories": n_categories,
        "precision_1": 1.0,
        "precision_2": 1.0,
        "precision_3": 1.0,
        "mean_1": 1 / n_categories,
        "mean_2": -jnp.log(n_categories - 1),
        "mean_3": 0.0,
        "tonic_volatility_2": -4.0,
        "tonic_volatility_3": -4.0,
    }
    binary_idxs: List[int] = [1 + i + len(network.edges) for i in range(n_categories)]
    default_parameters = {
        "binary_idxs": binary_idxs,  # type: ignore
        "n_categories": n_categories,
        "surprise": 0.0,
        "kl_divergence": 0.0,
        "alpha": jnp.ones(n_categories),
        "observed": 1,
        "mean": jnp.array([1.0 / n_categories] * n_categories),
        "binary_parameters": binary_parameters,
    }

    node_parameters = update_parameters(
        node_parameters, default_parameters, additional_parameters
    )

    network = insert_nodes(
        network=network,
        n_nodes=n_nodes,
        node_type=node_type,
        node_parameters=node_parameters,
    )

    # create the implied binary network(s) here
    for node_idx in range(network.n_nodes - 1, network.n_nodes - n_nodes - 1, -1):
        network = fill_categorical_state_node(
            network,
            node_idx=node_idx,
            binary_states_idxs=node_parameters["binary_idxs"],  # type: ignore
            binary_parameters=binary_parameters,
        )

    return network


def add_dp_state(
    network: Network, n_nodes: int, node_parameters: Dict, additional_parameters: Dict
):
    """Add a Dirichlet Process node to a network."""
    node_type = 4

    if "batch_size" in additional_parameters.keys():
        batch_size = additional_parameters["batch_size"]
    elif "batch_size" in node_parameters.keys():
        batch_size = node_parameters["batch_size"]
    else:
        batch_size = 10

    default_parameters = {
        "observed": 1,
        "batch_size": batch_size,  # number of branches available in the network
        "n": jnp.zeros(batch_size),  # number of observation in each cluster
        "n_total": 0,  # the total number of observations in the node
        "alpha": 1.0,  # concentration parameter for the implied Dirichlet dist.
        "expected_means": jnp.zeros(batch_size),
        "expected_sigmas": jnp.ones(batch_size),
        "sensory_precision": 1.0,
        "activated": jnp.zeros(batch_size),
        "value_coupling_children": (1.0,),
        "mean": 0.0,
        "n_active_cluster": 0,
    }

    node_parameters = update_parameters(
        node_parameters, default_parameters, additional_parameters
    )

    network = insert_nodes(
        network=network,
        n_nodes=n_nodes,
        node_type=node_type,
        node_parameters=node_parameters,
    )

    return network


def get_couplings(
    value_parents: Optional[Union[Tuple, List, int]],
    volatility_parents: Optional[Union[Tuple, List, int]],
    value_children: Optional[Union[Tuple, List, int]],
    volatility_children: Optional[Union[Tuple, List, int]],
) -> Tuple[Tuple, ...]:
    """Transform coupling parameter into tuple of indexes and strenghts."""
    couplings = []
    for indexes in [
        value_parents,
        volatility_parents,
        value_children,
        volatility_children,
    ]:
        if indexes is not None:
            if isinstance(indexes, int):
                coupling_idxs = tuple([indexes])
                coupling_strengths = tuple([1.0])
            elif isinstance(indexes, list):
                coupling_idxs = tuple(indexes)
                coupling_strengths = tuple([1.0] * len(coupling_idxs))
            elif isinstance(indexes, tuple):
                coupling_idxs = tuple(indexes[0])
                coupling_strengths = tuple(indexes[1])
        else:
            coupling_idxs, coupling_strengths = None, None
        couplings.append((coupling_idxs, coupling_strengths))

    return tuple(couplings)


def update_parameters(
    node_parameters: Dict, default_parameters: Dict, additional_parameters: Dict
) -> Dict:
    """Update the default node parameters using keywords args and dictonary."""
    if bool(additional_parameters):
        # ensure that all passed values are valid keys
        invalid_keys = [
            key
            for key in additional_parameters.keys()
            if key not in default_parameters.keys()
        ]

        if invalid_keys:
            raise ValueError(
                (
                    "Some parameter(s) passed as keyword arguments were not found "
                    f"in the default key list for this node (i.e. {invalid_keys})."
                    " If you want to create a new key in the node attributes, "
                    "please use the node_parameters argument instead."
                )
            )

        # if keyword parameters were provided, update the default_parameters dict
        default_parameters.update(additional_parameters)

    # update the defaults using the dict parameters
    default_parameters.update(node_parameters)

    return default_parameters


def insert_nodes(
    network: Network,
    n_nodes: int,
    node_type: int,
    node_parameters: Dict,
    value_parents: Tuple = (None, None),
    volatility_parents: Tuple = (None, None),
    value_children: Tuple = (None, None),
    volatility_children: Tuple = (None, None),
    coupling_fn: Tuple[Optional[Callable], ...] = (None,),
) -> Network:
    """Insert a set of parametrised node in a network."""
    # ensure that the set of coupling functions match with the number of child nodes
    if value_children[0] is not None:
        if value_children[0] is None:
            children_number = 0
        elif isinstance(value_children[0], int):
            children_number = 1
        elif isinstance(value_children[0], tuple):
            children_number = len(value_children[0])

        # for mutiple value children, set a default tuple with corresponding length

        if children_number != len(coupling_fn):
            if coupling_fn == (None,):
                coupling_fn = children_number * coupling_fn
            else:
                raise ValueError(
                    "The number of coupling fn and value children do not match"
                )

    for _ in range(n_nodes):
        # convert the structure to a list to modify it
        edges_as_list: List = list(network.edges)

        node_idx = network.n_nodes  # the index of the new node

        # add a new edge
        edges_as_list.append(
            AdjacencyLists(node_type, None, None, None, None, coupling_fn=coupling_fn)
        )

        # convert the list back to a tuple
        network.edges = tuple(edges_as_list)

        # update the node structure
        network.attributes[node_idx] = deepcopy(node_parameters)

        # Update the edges of the parents and children accordingly
        # --------------------------------------------------------
        if value_parents[0] is not None:
            network.add_edges(
                kind="value",
                parent_idxs=value_parents[0],
                children_idxs=node_idx,
                coupling_strengths=value_parents[1],  # type: ignore
            )
        if value_children[0] is not None:
            network.add_edges(
                kind="value",
                parent_idxs=node_idx,
                children_idxs=value_children[0],
                coupling_strengths=value_children[1],  # type: ignore
                coupling_fn=coupling_fn,
            )
        if volatility_children[0] is not None:
            network.add_edges(
                kind="volatility",
                parent_idxs=node_idx,
                children_idxs=volatility_children[0],
                coupling_strengths=volatility_children[1],  # type: ignore
            )
        if volatility_parents[0] is not None:
            network.add_edges(
                kind="volatility",
                parent_idxs=volatility_parents[0],
                children_idxs=node_idx,
                coupling_strengths=volatility_parents[1],  # type: ignore
            )

        network.n_nodes += 1

    return network
