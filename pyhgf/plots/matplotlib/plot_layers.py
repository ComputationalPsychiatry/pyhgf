# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from pyhgf.typing import LayerState

if TYPE_CHECKING:
    from pyhgf.model import DeepNetwork

# Virtual variables that are computed from ``LayerState`` fields rather than
# read directly. Each entry maps a name accepted in ``variables`` to a
# function ``(LayerState) -> ndarray`` of shape (T, n_nodes).
_DERIVED_VARIABLES = {
    "PWPE": lambda layer: (
        (np.asarray(layer.mean) - np.asarray(layer.expected_mean))
        * np.asarray(layer.expected_precision)
    ),
}


def plot_layers(
    network: "DeepNetwork",
    layers: Optional[Sequence[int]] = None,
    variables: Union[str, Sequence[str]] = ("expected_mean",),
    mode: str = "all",
    figsize: Optional[tuple] = None,
    axes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Plot layer-wise parameter trajectories of a :class:`DeepNetwork`.

    Each row of the resulting figure corresponds to a *variable* (a field of
    :class:`pyhgf.typing.LayerState`) and each column to a *layer*. In ``"all"``
    mode every node trajectory is drawn as its own line; in ``"mean_ci"`` mode
    the across-node mean and a 95% confidence interval are drawn with
    ``seaborn.lineplot``.

    Parameters
    ----------
    network :
        A :class:`pyhgf.model.DeepNetwork` instance whose ``trajectories``
        attribute has been populated (call ``net.fit(..., record_trajectories=True)``
        first).
    layers :
        Indices of the layers to plot. ``None`` (default) plots every layer.
    variables :
        Name (or sequence of names) of :class:`pyhgf.typing.LayerState` fields
        to plot — for example ``"expected_mean"``, ``"precision"``,
        ``"value_prediction_error"``, ``"mean_vol"``. The derived name
        ``"PWPE"`` is also accepted: it plots the precision-weighted
        prediction error ``(mean - expected_mean) * expected_precision``.
        A single string is accepted as shorthand for a one-element list.
    mode :
        ``"all"`` to draw one line per node, ``"mean_ci"`` to draw the
        across-node mean with a 95% confidence interval (Seaborn's
        ``lineplot``).
    figsize :
        Figure size in inches. Defaults to ``(3.5 * n_cols, 2.5 * n_rows)``.
    axes :
        Pre-existing 2D array of Matplotlib axes (rows = variables, cols =
        layers). When ``None`` (default), a new figure is created.

    Returns
    -------
    axes :
        2D ``ndarray`` of Matplotlib axes, shape ``(len(variables), len(layers))``.

    Raises
    ------
    ValueError
        If ``network.trajectories`` is ``None``, if a variable name is not a
        ``LayerState`` field, if a layer index is out of range, or if *mode*
        is not one of ``"all"``/``"mean_ci"``.
    """
    if network.trajectories is None:
        raise ValueError(
            "Network has no trajectories. Call `fit(..., "
            "record_trajectories=True)` before plotting."
        )

    # Normalise inputs ---------------------------------------------------
    if isinstance(variables, str):
        variables = (variables,)
    variables = tuple(variables)

    if layers is None:
        layers = list(range(network.n_layers))
    layers = list(layers)

    valid_fields = set(LayerState._fields)
    valid_names = valid_fields | set(_DERIVED_VARIABLES)
    invalid_vars = [v for v in variables if v not in valid_names]
    if invalid_vars:
        raise ValueError(
            f"Unknown variable(s) {invalid_vars}. "
            f"Valid names: {sorted(valid_names)} "
            f"(LayerState fields plus derived: {sorted(_DERIVED_VARIABLES)})."
        )

    invalid_layers = [i for i in layers if not 0 <= i < network.n_layers]
    if invalid_layers:
        raise ValueError(
            f"Layer index/indices {invalid_layers} out of range "
            f"[0, {network.n_layers})."
        )

    if mode not in ("all", "mean_ci"):
        raise ValueError(f"mode must be 'all' or 'mean_ci', got {mode!r}.")

    n_rows = len(variables)
    n_cols = len(layers)

    # Set up the axes grid ----------------------------------------------
    if axes is None:
        if figsize is None:
            figsize = (3.5 * n_cols, 2.5 * n_rows)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True
        )
    else:
        axes = np.atleast_2d(axes)
        if axes.shape != (n_rows, n_cols):
            raise ValueError(
                f"`axes` shape {axes.shape} does not match "
                f"(n_variables, n_layers) = ({n_rows}, {n_cols})."
            )

    # Plot ---------------------------------------------------------------
    for r, var in enumerate(variables):
        for c, layer_idx in enumerate(layers):
            ax = axes[r, c]
            layer = network.trajectories.layers[layer_idx]
            if var in _DERIVED_VARIABLES:
                data = np.asarray(_DERIVED_VARIABLES[var](layer))
            else:
                data = np.asarray(getattr(layer, var))
            n_steps, n_nodes = data.shape

            if mode == "all":
                time = np.arange(n_steps)
                for n in range(n_nodes):
                    ax.plot(time, data[:, n], lw=1, alpha=0.8)
            else:  # "mean_ci"
                long_df = pd.DataFrame({
                    "time_step": np.tile(np.arange(n_steps), n_nodes),
                    var: data.T.ravel(),
                    "node": np.repeat(np.arange(n_nodes), n_steps),
                })
                sns.lineplot(
                    data=long_df,
                    x="time_step",
                    y=var,
                    ax=ax,
                    errorbar=("ci", 95),
                )

            if r == 0:
                ax.set_title(
                    f"layer {layer_idx} (size={network.layer_sizes[layer_idx]})"
                )
            if c == 0:
                ax.set_ylabel(var)
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("time step")
            else:
                ax.set_xlabel("")

    return axes
