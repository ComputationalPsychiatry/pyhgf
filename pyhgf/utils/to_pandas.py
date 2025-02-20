# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd

from pyhgf.math import binary_surprise, gaussian_surprise

if TYPE_CHECKING:
    from pyhgf.model import Network


def flatten_with_trajectory(x):
    """Flatten an array while preserving trajectory information.

    Parameters
    ----------
    x : array-like
        Input array, which can have one or more dimensions.

    Returns
    -------
    tuple

    """
    x_arr = jnp.asarray(x)
    if x_arr.ndim >= 2:
        n_traj, n_steps = x_arr.shape[:2]
        flat = x_arr.reshape(
            n_traj * n_steps, *x_arr.shape[2:]
        )  # Preserve additional dimensions
        # Create the trajectory vector
        trajectory = np.repeat(np.arange(n_traj), n_steps)
        return flat, trajectory
    else:
        return x_arr, None


def to_pandas(network: "Network") -> pd.DataFrame:
    """Export the nodes trajectories and surprise as a Pandas data frame.

    Returns
    -------
    trajectories_df :
        Pandas data frame with the time series of sufficient statistics and the
        surprise of each node in the structure.

    """
    n_nodes = len(network.edges)

    # --- Process time steps and time ---
    # Assume time_step is stored in the last input node.
    ts = network.node_trajectories[-1]["time_step"]
    ts_arr = jnp.asarray(ts)
    if ts_arr.ndim < 2:
        # If there is no trajectory dimension, assume a single trajectory.
        time_steps_flat = np.array(ts_arr)
        time_flat = np.cumsum(time_steps_flat)
        trajectory_ids = np.zeros_like(time_flat, dtype=int)
    else:
        # Compute cumulative sum separately for each trajectory
        n_traj, n_steps = ts_arr.shape
        time_list = [jnp.cumsum(ts_arr[i]) for i in range(n_traj)]
        time_arr = jnp.concatenate(time_list, axis=0)
        time_steps_flat = ts_arr.reshape(-1)
        trajectory_ids = np.repeat(np.arange(n_traj), n_steps)
        time_flat = np.array(time_arr)

    # Create initial DataFrame with "trajectory", "time_steps", and "time" columns
    trajectories_df = pd.DataFrame(
        {
            "trajectory": trajectory_ids,
            "time_steps": np.array(time_steps_flat),
            "time": time_flat,
        }
    )

    # --- Process state node statistics (binary and continuous) ---
    states_indexes = [i for i in range(n_nodes) if network.edges[i].node_type in [1, 2]]
    stats_dict = {}
    for i in states_indexes:
        for var, data in network.node_trajectories[i].items():
            if ("mean" in var) or ("precision" in var):
                data_arr = jnp.asarray(data)
                if data_arr.ndim >= 2:
                    flat, _ = flatten_with_trajectory(data_arr)
                    col_name = f"x_{i}_{var}"
                    stats_dict[col_name] = np.array(flat)
                else:
                    col_name = f"x_{i}_{var}"
                    stats_dict[col_name] = np.array(data_arr)
    df_stats = pd.DataFrame(stats_dict)
    trajectories_df = pd.concat([trajectories_df, df_stats], axis=1)

    # --- Process exponential family nodes (node_type == 3) ---
    ef_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 3]
    for i in ef_indexes:
        for var in ["nus", "xis", "mean"]:
            data = network.node_trajectories[i][var]
            data_arr = jnp.asarray(data)
            if data_arr.ndim >= 2:
                flat, _ = flatten_with_trajectory(data_arr)
                col_name = f"x_{i}_{var}"
                trajectories_df[col_name] = np.array(flat)
            else:
                trajectories_df[f"x_{i}_{var}"] = np.array(data_arr)

    # --- Compute and add surprise ---
    # Compute surprise for binary nodes (node_type == 1)
    binary_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 1]
    for bin_idx in binary_indexes:
        surprise = binary_surprise(
            x=network.node_trajectories[bin_idx]["mean"],
            expected_mean=network.node_trajectories[bin_idx]["expected_mean"],
        )
        surp_arr = jnp.asarray(surprise)
        if surp_arr.ndim >= 2:
            flat, _ = flatten_with_trajectory(surp_arr)
            trajectories_df[f"x_{bin_idx}_surprise"] = np.array(flat)
        else:
            trajectories_df[f"x_{bin_idx}_surprise"] = np.array(surp_arr)

    # Compute surprise for continuous nodes (node_type == 2)
    continuous_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 2]
    for con_idx in continuous_indexes:
        surprise = gaussian_surprise(
            x=network.node_trajectories[con_idx]["mean"],
            expected_mean=network.node_trajectories[con_idx]["expected_mean"],
            expected_precision=network.node_trajectories[con_idx]["expected_precision"],
        )
        surp_arr = jnp.asarray(surprise)
        if surp_arr.ndim >= 2:
            flat, _ = flatten_with_trajectory(surp_arr)
            trajectories_df[f"x_{con_idx}_surprise"] = np.array(flat)
        else:
            trajectories_df[f"x_{con_idx}_surprise"] = np.array(surp_arr)

    # Compute total surprise by summing all "_surprise" columns
    surprise_cols = [col for col in trajectories_df.columns if "_surprise" in col]
    trajectories_df["total_surprise"] = trajectories_df[surprise_cols].sum(
        axis=1, min_count=1
    )

    return trajectories_df
