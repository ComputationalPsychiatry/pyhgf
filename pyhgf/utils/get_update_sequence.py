# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, List, Tuple

from jax.tree_util import Partial

from pyhgf.typing import Sequence
from pyhgf.updates.posterior.categorical import categorical_state_update
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
)
from pyhgf.updates.posterior.exponential import (
    posterior_update_exponential_family_dynamic,
)
from pyhgf.updates.prediction.binary import binary_state_node_prediction
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction.dirichlet import dirichlet_node_prediction
from pyhgf.updates.prediction_error.binary import binary_state_node_prediction_error
from pyhgf.updates.prediction_error.categorical import (
    categorical_state_prediction_error,
)
from pyhgf.updates.prediction_error.continuous import continuous_node_prediction_error
from pyhgf.updates.prediction_error.dirichlet import dirichlet_node_prediction_error
from pyhgf.updates.prediction_error.exponential import (
    prediction_error_update_exponential_family_dynamic,
    prediction_error_update_exponential_family_fixed,
)

if TYPE_CHECKING:
    from pyhgf.model import Network


def get_update_sequence(
    network: "Network", update_type: str
) -> Tuple[Sequence, Sequence]:
    """Generate an update sequence from the network's structure.

    This function return an optimized update sequence considering the edges of the
    network. The function ensures that the following principles apply:
    1. all children have computed prediction errors before the parent is updated.
    2. all children have been updated before the parent compute the prediction errors.

    Parameters
    ----------
    network :
        A neural network, instance of :py:class:`pyhgf.model.network.Network`.
    update_type :
        The type of update to perform for volatility coupling. Can be `"eHGF"`
        (defaults) or `"standard"`. The eHGF update step was proposed as an
        alternative to the original definition in that it starts by updating the
        mean and then the precision of the parent node, which generally reduces the
        errors associated with impossible parameter space and improves sampling.

    Returns
    -------
    prediction_sequence :
        The sequence of prediction update.
    update_sequence :
        The sequence of prediction error and posterior updates.

    """
    # initialize the update and prediction sequences
    update_sequence: List = []
    prediction_sequence: List = []

    n_nodes = len(network.edges)

    # list all nodes that are not triggering prediction errors or posterior updates
    # do not call posterior updates for nodes without children (input nodes)
    nodes_without_prediction_error = [i for i in range(n_nodes)]
    nodes_without_prediction = [i for i in range(n_nodes)]
    nodes_without_posterior_update = [
        i
        for i in range(n_nodes)
        if not (
            (network.edges[i].value_children is None)
            & (network.edges[i].volatility_children is None)
        )
    ]

    # do not update continuous nodes that are parents of an ef state node
    for i in nodes_without_posterior_update:
        for child_idx in network.edges[i].value_children or []:
            if network.edges[child_idx].node_type == 3:
                nodes_without_posterior_update.remove(i)

    # prediction updates ---------------------------------------------------------------
    while True:
        no_update = True

        # for all nodes that should apply prediction update ----------------------------
        # verify that all children have computed the prediction error
        for idx in nodes_without_prediction:
            all_parents = [
                i
                for idx in [
                    network.edges[idx].value_parents,
                    network.edges[idx].volatility_parents,
                ]
                if idx is not None
                for i in idx
            ]

            # there is no parent waiting for a prediction update
            if not any([i in nodes_without_prediction for i in all_parents]):
                no_update = False
                nodes_without_prediction.remove(idx)
                if network.edges[idx].node_type == 1:
                    prediction_sequence.append((idx, binary_state_node_prediction))
                elif network.edges[idx].node_type == 2:
                    prediction_sequence.append((idx, continuous_node_prediction))
                elif network.edges[idx].node_type == 4:
                    prediction_sequence.append((idx, dirichlet_node_prediction))

        if not nodes_without_prediction:
            break

        if no_update:
            raise Warning(
                "The structure of the network cannot be updated consistently."
            )

    # prediction errors and posterior updates
    # will fail if the structure of the network does not allow a consistent update order
    # ----------------------------------------------------------------------------------
    while True:
        no_update = True

        # for all nodes that should apply posterior update -----------------------------
        # verify that all children have computed the prediction error
        update_fn = None
        for idx in nodes_without_posterior_update:
            all_children = [
                i
                for idx in [
                    network.edges[idx].value_children,
                    network.edges[idx].volatility_children,
                ]
                if idx is not None
                for i in idx
            ]

            # all the children have computed prediction errors
            if all([i not in nodes_without_prediction_error for i in all_children]):
                no_update = False
                if network.edges[idx].node_type == 2:
                    if update_type == "eHGF":
                        if network.edges[idx].volatility_children is not None:
                            update_fn = continuous_node_posterior_update_ehgf
                        else:
                            update_fn = continuous_node_posterior_update
                    elif update_type == "standard":
                        update_fn = continuous_node_posterior_update

                elif network.edges[idx].node_type == 4:

                    update_fn = None

                update_sequence.append((idx, update_fn))
                nodes_without_posterior_update.remove(idx)

        # for all nodes that should apply prediction error------------------------------
        # verify that all children have been updated
        update_fn = None
        for idx in nodes_without_prediction_error:

            all_parents = [
                i
                for idx in [
                    network.edges[idx].value_parents,
                    network.edges[idx].volatility_parents,
                ]
                if idx is not None
                for i in idx
            ]

            # if this node has no parent, no need to compute prediction errors
            # unless this is an exponential family state node with fixed learning rate
            if len(all_parents) == 0:
                if network.edges[idx].node_type == 3:

                    # retrieve the desired sufficient statistics function
                    # from the side parameter dictionary
                    sufficient_stats_fn = network.additional_parameters[idx][
                        "sufficient_stats_fn"
                    ]
                    network.additional_parameters[idx].pop("sufficient_stats_fn")

                    # create the sufficient statistic function
                    # for the exponential family node
                    ef_update = Partial(
                        prediction_error_update_exponential_family_fixed,
                        sufficient_stats_fn=sufficient_stats_fn,
                    )
                    update_fn = ef_update
                    no_update = False
                    update_sequence.append((idx, update_fn))

                nodes_without_prediction_error.remove(idx)

            else:
                # if this node has been updated
                if idx not in nodes_without_posterior_update:

                    if network.edges[idx].node_type == 0:
                        pass
                    elif network.edges[idx].node_type == 1:
                        update_fn = binary_state_node_prediction_error
                    elif network.edges[idx].node_type == 2:
                        update_fn = continuous_node_prediction_error
                    elif network.edges[idx].node_type == 3:
                        # retrieve the desired sufficient statistics function
                        # from the side parameter dictionary
                        sufficient_stats_fn = network.additional_parameters[idx][
                            "sufficient_stats_fn"
                        ]
                        network.additional_parameters[idx].pop("sufficient_stats_fn")
                        # create the sufficient statistic function
                        # for the exponential family node
                        update_fn = Partial(
                            prediction_error_update_exponential_family_dynamic,
                            sufficient_stats_fn=sufficient_stats_fn,
                        )

                        # add the posterior update here
                        # this will be moved at the end of the sequence later
                        update_sequence.append(
                            (idx, posterior_update_exponential_family_dynamic)
                        )
                    elif network.edges[idx].node_type == 4:
                        update_fn = dirichlet_node_prediction_error
                    elif network.edges[idx].node_type == 5:
                        update_fn = categorical_state_prediction_error

                        # add the update here, this will move at the end of the sequence
                        update_sequence.append((idx, categorical_state_update))
                    else:
                        raise ValueError(f"Invalid node type encountered at node {idx}")

                    no_update = False
                    update_sequence.append((idx, update_fn))
                    nodes_without_prediction_error.remove(idx)

        if (not nodes_without_prediction_error) and (
            not nodes_without_posterior_update
        ):
            break

        if no_update:
            raise Warning(
                "The structure of the network cannot be updated consistently."
            )

    # remove None steps and return the update sequence
    prediction_sequence = [
        update for update in prediction_sequence if update[1] is not None
    ]
    update_sequence = [update for update in update_sequence if update[1] is not None]

    # move all categorical steps at the end of the sequence
    for step in update_sequence:
        if not isinstance(step[1], Partial):
            if step[1].__name__ in [
                "posterior_update_exponential_family_dynamic",
                "categorical_state_update",
            ]:
                update_sequence.remove(step)
                update_sequence.append(step)

    return tuple(prediction_sequence), tuple(update_sequence)
