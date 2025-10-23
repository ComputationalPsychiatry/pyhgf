# Author: Sylvain Estebe
# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from pyhgf.model import Network


def plot_samples(
    network: "Network",
    show_total_surprise: bool = False,
    figsize: tuple[int, int] = (18, 9),
    axs: Optional[Union[list, Axes]] = None,
) -> Axes:
    """Plot simulation trajectories for nodes.

    After drawing the nodes via `plot_nodes`, this function extracts sample trajectories
    and overlays them on the plot. The figures should be passed to the `axs` argument.

    Parameters
    ----------
    network : Network
        Main Network instance.
    show_total_surprise :
        If `True`, plot the sum of surprises across all nodes in the bottom panel.
        Defaults to `False`.
    figsize : tuple of int, default (18, 9)
        Figure size.
    axs : list or Axes, optional
        Axes on which to plot. A new figure is created if None.

    Returns
    -------
    Axes :
        Matplotlib axes used for plotting.

    """
    # Number of nodes based on the network's structure
    n_nodes = len(network.edges)

    # Create or get the axes
    if axs is None:
        n_rows = n_nodes + 1 if show_total_surprise else n_nodes
        _, axs = plt.subplots(nrows=n_rows, figsize=figsize, sharex=True)
        if n_nodes == 1:
            axs = [axs]

    # get the current time step, accounting for previous node trajectories
    if network.node_trajectories is None:
        current_time = 0.0
    else:
        current_time = network.node_trajectories[-1]["time_step"].cumsum()[-1]
    shifted_time_steps = current_time + network.samples[-1]["time_step"][0].cumsum()
    n_predictions = network.samples[-1]["time_step"].shape[0]

    for node in range(network.n_nodes):
        ax_idx = n_nodes - node - 1

        # plot individual generation of samples
        for i in range(n_predictions):
            axs[ax_idx].plot(
                shifted_time_steps,
                network.samples[node]["expected_mean"][i],
                ls="-",
                color="#2a2a2a",
                linewidth=0.25,
                alpha=0.2,
                zorder=1,
            )

        # average over samples
        axs[ax_idx].plot(
            shifted_time_steps,
            network.samples[node]["expected_mean"].mean(0),
            ls="-",
            color="#2a2a2a",
            linewidth=1.0,
            zorder=1,
            label="Expected future wbeliefs",
        )

        axs[ax_idx].legend(loc="lower right")

    return axs
