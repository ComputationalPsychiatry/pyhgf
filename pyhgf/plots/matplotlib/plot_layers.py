# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pyhgf.typing import LayerState

if TYPE_CHECKING:
    from pyhgf.model import DeepNetwork

# 95% normal-approximation z-score used for the ``"mean_ci"`` band.
_Z_95 = 1.959963984540054

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
    the across-node mean and a 95% confidence interval are drawn as a
    Matplotlib line + shaded band.

    Both modes are vectorised so the cost scales linearly with ``T * n_nodes``
    — the ``"all"`` path uses a single :class:`matplotlib.collections.LineCollection`
    instead of a per-node ``ax.plot`` call, and the ``"mean_ci"`` path
    aggregates over nodes with NumPy and renders one ``plot`` + ``fill_between``
    pair per axis (no long-format DataFrame, no bootstrap).

    Parameters
    ----------
    network :
        A :class:`pyhgf.model.DeepNetwork` instance whose ``trajectories``
        attribute has been populated (call ``net.fit(...,
        record_trajectories=True)`` first).
    layers :
        Indices of the layers to plot. ``None`` (default) plots every layer.
    variables :
        Name (or sequence of names) of :class:`pyhgf.typing.LayerState` fields to plot
        — for example ``"expected_mean"``, ``"precision"``,
        ``"value_prediction_error"``, ``"mean_vol"``. The derived name ``"PWPE"`` is
        also accepted: it plots the precision-weighted prediction error ``(mean -
        expected_mean) * expected_precision``. A single string is accepted as shorthand
        for a one-element list.
    mode :
        ``"all"`` to draw one line per node, ``"mean_ci"`` to draw the across-node mean
        with a 95% normal-approximation confidence band.
    figsize :
        Figure size in inches. Defaults to ``(3.5 * n_cols, 2.5 * n_rows)``.
    axes :
        Pre-existing 2D array of Matplotlib axes (rows = variables, cols = layers). When
        ``None`` (default), a new figure is created.

    Returns
    -------
    axes :
        2D ``ndarray`` of Matplotlib axes, shape ``(len(variables),
        len(layers))``.

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
            time = np.arange(n_steps)

            if mode == "all":
                # One LineCollection per axis is dramatically faster than
                # n_nodes separate ax.plot calls when n_nodes is large.
                # segments has shape (n_nodes, T, 2): per-node (x, y) pairs.
                x = np.broadcast_to(time, (n_nodes, n_steps))
                segments = np.stack([x, data.T], axis=-1)
                lc = LineCollection(segments, linewidths=1.0, alpha=0.6)
                ax.add_collection(lc)
                ax.set_xlim(time[0], time[-1] if n_steps > 1 else time[0] + 1)
                finite = data[np.isfinite(data)]
                if finite.size:
                    ymin, ymax = float(finite.min()), float(finite.max())
                    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
                    ax.set_ylim(ymin - pad, ymax + pad)
            else:  # "mean_ci"
                # Aggregate across nodes with NumPy (O(T * n_nodes) once),
                # then draw a single line + shaded band.  This avoids both
                # the (T * n_nodes)-row long DataFrame and Seaborn's
                # bootstrap, which dominate runtime for large inputs.
                mean = np.nanmean(data, axis=1)
                ax.plot(time, mean, lw=1.5)
                if n_nodes > 1:
                    sem = np.nanstd(data, axis=1, ddof=1) / np.sqrt(n_nodes)
                    half = _Z_95 * sem
                    ax.fill_between(
                        time, mean - half, mean + half, alpha=0.25, linewidth=0
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
