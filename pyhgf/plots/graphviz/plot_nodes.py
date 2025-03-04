# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from pyhgf.model import Network


def plot_nodes(
    network: "Network",
    node_idxs: Union[int, List[int]],
    ci: bool = True,
    show_surprise: bool = True,
    show_posterior: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    color: Optional[Union[Tuple, str]] = None,
    axs: Optional[Union[List, Axes]] = None,
):
    """Plot the trajectory.

    Parameters
    ----------
    network : Network
        The network containing nodes whose trajectories will be plotted.
    node_idxs : int or list of int
        Index or list of indices of nodes to plot.
    ci : bool, optional (default=False)
        Whether to display confidence intervals around the expected mean.
    show_surprise : bool, optional (default=True)
        Whether to show surprise.
    show_posterior : bool, optional (default=False)
        Whether to display posterior mean estimates.
    figsize : tuple of int, optional (default=(12, 5))
        Size of the figure for plotting.
    color : tuple or str, optional (default=None)
        Color for the plotted lines.
    axs : list or plt.Axes, optional (default=None)
        Predefined matplotlib axes for plotting. If None, new axes are created.

    """
    if not isinstance(node_idxs, list):
        node_idxs = [node_idxs]
    trajectories_df = network.to_pandas()

    if axs is None:
        _, axs = plt.subplots(nrows=len(node_idxs), figsize=figsize, sharex=True)

    if isinstance(node_idxs, int) or len(node_idxs) == 1:
        axs = [axs]

    # Iterate over each node index to plot
    for i, node_idx in enumerate(node_idxs):
        ax = axs[i]

        # Dictionary to track unique legend labels and avoid duplicates
        handles_labels = {}

        # Loop through each trajectory and plot the corresponding data
        for traj, df_traj in trajectories_df.groupby("trajectory"):
            if node_idx in network.input_idxs:
                # Plot expected mean for input nodes
                (line,) = ax.plot(
                    df_traj.time,
                    df_traj[f"x_{node_idx}_expected_mean"],
                    label="Expected mean",
                    color=color,
                    linewidth=1,
                    zorder=2,
                )
                ax.set_ylabel(rf"$\mu_{{{node_idx}}}$")
                handles_labels["Expected mean"] = line

                # binary state nodes (e.g., Bernoulli/Categorical)
                if network.edges[node_idx].node_type == 1:
                    scatter = ax.scatter(
                        x=df_traj.time,
                        y=df_traj[f"x_{node_idx}_mean"],
                        s=3,
                        label="Observed",
                        color="#2a2a2a",
                        marker="o",
                        zorder=2,
                        alpha=0.4,
                    )
                    handles_labels["Observed"] = scatter

                # continuous state nodes
                if network.edges[node_idx].node_type == 2:
                    scatter = ax.scatter(
                        x=df_traj.time,
                        y=df_traj[f"x_{node_idx}_mean"],
                        s=3,
                        label="Input",
                        color="#2a2a2a",
                        zorder=2,
                    )
                    handles_labels["Input"] = scatter
                    if ci:
                        precision = df_traj[f"x_{node_idx}_expected_precision"]
                        sd = np.sqrt(1 / precision)
                        y1 = df_traj[f"x_{node_idx}_expected_mean"] - sd
                        y2 = df_traj[f"x_{node_idx}_expected_mean"] + sd
                        fill = ax.fill_between(
                            df_traj["time"], y1, y2, alpha=0.2, color=color, zorder=2
                        )
                        handles_labels["Confidence Interval"] = fill

                ax.set_title(f"Input Node {node_idx}", loc="left")

            else:
                # Plot state nodes (non-input)
                (line,) = ax.plot(
                    df_traj.time,
                    df_traj[f"x_{node_idx}_expected_mean"],
                    label="Expected mean",
                    color=color,
                    linewidth=1,
                    zorder=2,
                )
                handles_labels["Expected mean"] = line
                ax.set_ylabel(rf"$\mu_{{{node_idx}}}$")

                if ci:
                    precision = df_traj[f"x_{node_idx}_expected_precision"]
                    sd = np.sqrt(1 / precision)
                    y1 = df_traj[f"x_{node_idx}_expected_mean"] - sd
                    y2 = df_traj[f"x_{node_idx}_expected_mean"] + sd
                    fill = ax.fill_between(
                        df_traj["time"], y1, y2, alpha=0.2, color=color, zorder=2
                    )
                    handles_labels["Confidence Interval"] = fill

                if show_posterior:

                    scatter = ax.scatter(
                        x=df_traj.time,
                        y=df_traj[f"x_{node_idx}_mean"],
                        s=3,
                        label="Posterior",
                        color="#2a2a2a",
                        zorder=2,
                        alpha=0.5,
                    )
                    handles_labels["Posterior"] = scatter

                ax.set_title(f"State Node {node_idx}", loc="left")

        ax.set_xlabel("Time")

        # Plot surprise if enabled
        if show_surprise:
            # Extract surprise values for this node from the complete dataframe
            node_surprise = trajectories_df[f"x_{node_idx}_surprise"].to_numpy()

            if not np.isnan(node_surprise).all():
                surprise_ax = ax.twinx()

                sp = node_surprise.sum()
                surprise_ax.set_title(f"Surprise: {sp:.2f}", loc="right")

                surprise_ax.fill_between(
                    trajectories_df["time"],
                    y1=node_surprise,
                    y2=node_surprise.min(),
                    where=network.node_trajectories[node_idx]["observed"],
                    color="#7f7f7f",
                    alpha=0.1,
                    zorder=-1,
                    label="Surprise",
                )

                # Hide unobserved surprise values
                node_surprise[network.node_trajectories[node_idx]["observed"] == 0] = (
                    np.nan
                )

                surprise_ax.plot(
                    trajectories_df["time"],
                    node_surprise,
                    color="#2a2a2a",
                    linewidth=0.5,
                    zorder=-1,
                )

                surprise_ax.set_ylabel("Surprise")
                surprise_ax.legend(loc="upper right")

        # Ensure the legend appears only once per subplot
        unique_handles = list(handles_labels.values())
        unique_labels = list(handles_labels.keys())
        ax.legend(unique_handles, unique_labels, loc="upper left")

    return axs
