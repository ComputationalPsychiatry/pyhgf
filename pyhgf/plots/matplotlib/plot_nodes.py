# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from pyhgf.model import Network


def plot_nodes(
    network: "Network",
    node_idxs: Union[int, list[int]],
    ci: bool = True,
    show_surprise: bool = True,
    show_posterior: bool = False,
    figsize: tuple[int, int] = (12, 5),
    color: Optional[Union[tuple, str]] = None,
    axs: Optional[Union[list, Axes]] = None,
):
    r"""Plot the trajectory of expected sufficient statistics of a set of nodes.

    This function will plot the expected mean and precision (converted into standard
    deviation) before observation, and the Gaussian surprise after observation. If
    `children_inputs` is `True`, will also plot the children input (mean for value
    coupling and precision for volatility coupling).

    Parameters
    ----------
    network :
        An instance of main Network class.
    node_idxs :
        The index(es) of the probabilistic node(s) that should be plotted. If multiple
        indexes are provided, multiple rows will be appended to the figure, one for
        each node.
    ci :
        Whether to show the uncertainty around the values estimates (using the standard
        deviation :math:`\sqrt{\frac{1}{\hat{\pi}}}`).
    show_surprise :
        If `True` the surprise, defined as the negative log probability of the
        observation given the expectation, is plotted in the backgroud of the figure
        as grey shadded area.
    show_posterior :
        If `True`, plot the posterior mean and precision on the top of expected mean and
        precision. Defaults to `False`.
    figsize :
        The width and height of the figure. Defaults to `(18, 9)` for a two-level model,
        or to `(18, 12)` for a three-level model.
    color :
        The color of the main curve showing the beliefs trajectory.
    axs :
        A list of Matplotlib axes instances where to draw the trajectories. This should
        correspond to the number of nodes in the structure. The default is `None`
        (create a new figure).

    Returns
    -------
    axs :
        The Matplotlib axes instances where to plot the trajectories.

    Examples
    --------
    Visualization of nodes' trajectories from a three-level continuous HGF model.

    .. plot::

        from pyhgf import load_data
        from pyhgf.model import HGF

        # Set up standard 3-level HGF for continuous inputs
        hgf = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
            tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
            tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
            volatility_coupling={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = load_data("continuous")

        # Feed input
        hgf.input_data(input_data=timeserie)

        # Plot
        hgf.plot_nodes(node_idxs=1)

    """
    if not isinstance(node_idxs, list):
        node_idxs = [node_idxs]
    trajectories_df = network.to_pandas()

    if axs is None:
        _, axs = plt.subplots(nrows=len(node_idxs), figsize=figsize, sharex=True)

    if isinstance(node_idxs, int) | len(node_idxs) == 1:
        axs = [axs]

    # plotting an input node
    # ----------------------
    for i, node_idx in enumerate(node_idxs):
        if node_idx in network.input_idxs:
            # plotting mean
            axs[i].plot(
                trajectories_df.time,
                trajectories_df[f"x_{node_idx}_expected_mean"],
                label="Expected mean",
                color=color,
                linewidth=1,
                zorder=2,
            )
            axs[i].set_ylabel(rf"$\mu_{{{node_idx}}}$")

            # continuous state node ----------------------------------------------------
            if network.edges[node_idx].node_type == 2:
                input_label = "Continuous"
                axs[i].scatter(
                    x=trajectories_df.time[
                        trajectories_df[f"x_{node_idx}_observed"] == 1
                    ],
                    y=trajectories_df[f"x_{node_idx}_mean"][
                        trajectories_df[f"x_{node_idx}_observed"] == 1
                    ],
                    s=3,
                    label="Input",
                    color="#2a2a2a",
                    zorder=2,
                )
                # plotting standard deviation
                if ci is True:
                    precision = trajectories_df[f"x_{node_idx}_expected_precision"]
                    sd = np.sqrt(1 / precision)
                    y1 = trajectories_df[f"x_{node_idx}_expected_mean"] - sd
                    y2 = trajectories_df[f"x_{node_idx}_expected_mean"] + sd

            # binary state node --------------------------------------------------------
            elif network.edges[node_idx].node_type == 1:
                input_label = "Binary"
                axs[i].scatter(
                    x=trajectories_df.time[
                        trajectories_df[f"x_{node_idx}_observed"] == 1
                    ],
                    y=trajectories_df[f"x_{node_idx}_mean"][
                        trajectories_df[f"x_{node_idx}_observed"] == 1
                    ],
                    label="Input",
                    color="#2a2a2a",
                    zorder=2,
                    alpha=0.4,
                )

                # plotting standard deviation - in the case of a binary input node, the
                # CI should be read from the value parent using the sigmoid transform
                if ci is True:
                    # get parent nodes and sum predictions
                    mean_parent, precision_parent = 0.0, 0.0
                    for idx in network.edges[node_idx].value_parents:  # type: ignore
                        # compute  mu +/- sd at time t-1
                        # and use the sigmoid transform before plotting
                        mean_parent += trajectories_df[f"x_{idx}_expected_mean"]
                        precision_parent += trajectories_df[
                            f"x_{idx}_expected_precision"
                        ]
                    sd = np.sqrt(1 / precision_parent)
                    y1 = 1 / (1 + np.exp(-mean_parent + sd))
                    y2 = 1 / (1 + np.exp(-mean_parent - sd))

            if ci is True:
                axs[i].fill_between(
                    x=trajectories_df["time"],
                    y1=y1,
                    y2=y2,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )

            axs[i].set_title(f"{input_label} Input Node {node_idx}", loc="left")
            axs[i].legend(loc="upper left")

        # plotting state nodes
        # --------------------
        else:
            axs[i].set_title(
                f"State Node {node_idx}",
                loc="left",
            )

            # show the expected states
            # ------------------------
            # extract sufficient statistics from the data frame
            mean = trajectories_df[f"x_{node_idx}_expected_mean"]

            # plotting mean
            axs[i].plot(
                trajectories_df.time,
                mean,
                label="Expected mean",
                color=color,
                linewidth=1,
                zorder=2,
            )
            axs[i].set_ylabel(rf"$\mu_{{{node_idx}}}$")

            # plotting standard deviation
            if ci is True:
                precision = trajectories_df[f"x_{node_idx}_expected_precision"]
                sd = 1.0 / np.sqrt(precision)
                y1 = mean - sd
                y2 = mean + sd

                axs[i].fill_between(
                    x=trajectories_df.time,
                    y1=y1,
                    y2=y2,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )

            axs[i].legend(loc="upper left")

            # show the current states
            # -----------------------
            if show_posterior:
                # extract sufficient statistics from the data frame
                mean = trajectories_df[f"x_{node_idx}_mean"]

                axs[i].scatter(
                    x=trajectories_df.time,
                    y=mean,
                    s=3,
                    label="Posterior",
                    color="#2a2a2a",
                    zorder=2,
                    alpha=0.5,
                )
                axs[i].legend(loc="lower left")

        # plotting surprise
        # -----------------
        if show_surprise:
            node_surprise = trajectories_df[f"x_{node_idx}_surprise"].to_numpy()

            if not np.isnan(node_surprise).all():
                surprise_ax = axs[i].twinx()

                sp = node_surprise.sum()
                surprise_ax.set_title(
                    f"Surprise: {sp:.2f}",
                    loc="right",
                )
                surprise_ax.fill_between(
                    x=trajectories_df.time,
                    y1=node_surprise,
                    y2=node_surprise.min(),
                    where=network.node_trajectories[node_idx]["observed"],
                    color="#7f7f7f",
                    alpha=0.1,
                    zorder=-1,
                    label="Surprise",
                )

                # hide surprise if the input was not observed
                node_surprise[network.node_trajectories[node_idx]["observed"] == 0] = (
                    np.nan
                )
                surprise_ax.plot(
                    trajectories_df.time,
                    node_surprise,
                    color="#2a2a2a",
                    linewidth=0.5,
                    zorder=-1,
                )
                surprise_ax.set_ylabel("Surprise")
                surprise_ax.legend(loc="upper right")
    return axs
