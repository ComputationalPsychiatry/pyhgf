# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from pyhgf.model import Network


def plot_correlations(network: "Network") -> Axes:
    """Plot the heatmap correlation of the sufficient statistics trajectories.

    Parameters
    ----------
    network :
        An instance of the HGF model.

    Returns
    -------
    axs :
        The Matplotlib axe instance containing the heatmap of parameters trajectories
        correlation.

    """
    trajectories_df = network.to_pandas()
    trajectories_df = pd.concat(
        [
            trajectories_df[["time"]],
            trajectories_df[
                [
                    f"x_{i}_mean"
                    for i in range(len(network.edges))
                    if i in network.input_idxs
                ]
            ],
            trajectories_df.filter(regex="expected"),
            trajectories_df.filter(regex="surprise"),
        ],
        axis=1,
    )

    correlation_mat = trajectories_df.corr()
    ax = sns.heatmap(
        correlation_mat,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        linewidths=2,
        square=True,
    )
    ax.set_title("Correlations between the model trajectories")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)

    return ax
