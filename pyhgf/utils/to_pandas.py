# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd

from pyhgf.math import binary_surprise, gaussian_surprise

if TYPE_CHECKING:
    from pyhgf.model import Network


def to_pandas(network: "Network") -> pd.DataFrame:
    """Export the nodes trajectories and surprise as a Pandas data frame.

    Returns
    -------
    trajectories_df :
        Pandas data frame with the time series of sufficient statistics and the
        surprise of each node in the structure.

    """
    n_nodes = len(network.edges)
    # get time and time steps from the first input node
    trajectories_df = pd.DataFrame(
        {
            "time_steps": network.node_trajectories[-1]["time_step"],
            "time": jnp.cumsum(network.node_trajectories[-1]["time_step"]),
        }
    )

    # loop over continuous and binary state nodes and store sufficient statistics
    # ---------------------------------------------------------------------------
    states_indexes = [i for i in range(n_nodes) if network.edges[i].node_type in [1, 2]]
    df = pd.DataFrame(
        dict(
            [
                (f"x_{i}_{var}", network.node_trajectories[i][var])
                for i in states_indexes
                for var in network.node_trajectories[i].keys()
                if (("mean" in var) or ("precision" in var))
            ]
        )
    )
    trajectories_df = pd.concat([trajectories_df, df], axis=1)

    # loop over exponential family state nodes and store sufficient statistics
    # ------------------------------------------------------------------------
    ef_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 3]
    for i in ef_indexes:
        for var in ["nus", "xis", "mean"]:
            if network.node_trajectories[i][var].ndim == 1:
                trajectories_df = pd.concat(
                    [
                        trajectories_df,
                        pd.DataFrame(
                            dict([(f"x_{i}_{var}", network.node_trajectories[i][var])])
                        ),
                    ],
                    axis=1,
                )
            else:
                for ii in range(network.node_trajectories[i][var].shape[1]):
                    trajectories_df = pd.concat(
                        [
                            trajectories_df,
                            pd.DataFrame(
                                dict(
                                    [
                                        (
                                            f"x_{i}_{var}_{ii}",
                                            network.node_trajectories[i][var][:, ii],
                                        )
                                    ]
                                )
                            ),
                        ],
                        axis=1,
                    )

    # add surprise from binary state nodes
    binary_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 1]
    for bin_idx in binary_indexes:
        surprise = binary_surprise(
            x=network.node_trajectories[bin_idx]["mean"],
            expected_mean=network.node_trajectories[bin_idx]["expected_mean"],
        )
        trajectories_df[f"x_{bin_idx}_surprise"] = surprise

    # add surprise from continuous state nodes
    continuous_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 2]
    for con_idx in continuous_indexes:
        surprise = gaussian_surprise(
            x=network.node_trajectories[con_idx]["mean"],
            expected_mean=network.node_trajectories[con_idx]["expected_mean"],
            expected_precision=network.node_trajectories[con_idx]["expected_precision"],
        )
        trajectories_df[f"x_{con_idx}_surprise"] = surprise

    # compute the global surprise over all node
    trajectories_df["total_surprise"] = trajectories_df.iloc[
        :, trajectories_df.columns.str.contains("_surprise")
    ].sum(axis=1, min_count=1)

    return trajectories_df
