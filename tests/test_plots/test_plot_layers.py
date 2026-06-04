# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest
from matplotlib.collections import LineCollection, PolyCollection

from pyhgf.model import DeepNetwork


def test_plot_layers_default(trained_deep_network):
    """Default call returns a (1, n_layers) array of axes and creates a figure."""
    axs = trained_deep_network.plot_layers()
    assert axs.shape == (1, trained_deep_network.n_layers)
    plt.close("all")


def test_plot_layers_layers_int(trained_deep_network):
    """A single int is accepted as shorthand for a one-element layer list."""
    axs_int = trained_deep_network.plot_layers(layers=1)
    assert axs_int.shape == (1, 1)

    # numpy integer types are also accepted
    axs_np = trained_deep_network.plot_layers(layers=np.int64(2))
    assert axs_np.shape == (1, 1)
    plt.close("all")


def test_plot_layers_layers_list(trained_deep_network):
    """A list of ints selects the right columns in order."""
    axs = trained_deep_network.plot_layers(layers=[0, 2])
    assert axs.shape == (1, 2)
    # Layer 0 has size=2, layer 2 has size=3 in the fixture.
    assert "size=2" in axs[0, 0].get_title()
    assert "size=3" in axs[0, 1].get_title()
    plt.close("all")


def test_plot_layers_variables_string_and_sequence(trained_deep_network):
    """A single string variable is wrapped in a list; sequences add rows."""
    axs_str = trained_deep_network.plot_layers(variables="precision")
    assert axs_str.shape == (1, trained_deep_network.n_layers)
    assert axs_str[0, 0].get_ylabel() == "precision"

    axs_seq = trained_deep_network.plot_layers(variables=("expected_mean", "precision"))
    assert axs_seq.shape == (2, trained_deep_network.n_layers)
    assert axs_seq[0, 0].get_ylabel() == "expected_mean"
    assert axs_seq[1, 0].get_ylabel() == "precision"
    plt.close("all")


def test_plot_layers_pwpe_derived_variable(trained_deep_network):
    """The derived ``"PWPE"`` variable plots ``|mean - expected_mean| * EP`` ≥ 0."""
    from pyhgf.plots.matplotlib.plot_layers import _DERIVED_VARIABLES

    axs = trained_deep_network.plot_layers(variables="PWPE")
    assert axs.shape == (1, trained_deep_network.n_layers)

    required, fn = _DERIVED_VARIABLES["PWPE"]
    for i in range(trained_deep_network.n_layers):
        per_layer = {f: trained_deep_network.trajectories[f][i] for f in required}
        vals = np.asarray(fn(per_layer))
        finite = vals[np.isfinite(vals)]
        if finite.size:
            assert (finite >= 0).all()
    plt.close("all")


def test_plot_layers_mode_all_uses_line_collection(trained_deep_network):
    """``mode='all'`` uses a fast-path LineCollection (no per-node Line2D)."""
    axs = trained_deep_network.plot_layers(mode="all")
    ax = axs[0, 0]
    assert any(isinstance(c, LineCollection) for c in ax.collections)
    assert len(ax.lines) == 0
    plt.close("all")


def test_plot_layers_mode_mean_ci_draws_line_and_band(trained_deep_network):
    """``mode='mean_ci'`` draws one mean line plus a fill-between band."""
    axs = trained_deep_network.plot_layers(mode="mean_ci")
    ax = axs[0, 0]
    assert len(ax.lines) == 1
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close("all")


def test_plot_layers_mean_ci_single_node_skips_band():
    """With ``n_nodes == 1`` the CI band is skipped (only the mean line)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 3)).astype(np.float32)
    y = rng.standard_normal((10, 1)).astype(np.float32)

    from pyhgf.typing.vectorised import RECORD_ALL

    net = DeepNetwork().add_layer(size=1).add_layer(size=4).add_layer(size=3)
    net.fit(x=x, y=y, optimizer=optax.adam(1e-3), record=RECORD_ALL)

    axs = net.plot_layers(layers=0, mode="mean_ci")
    ax = axs[0, 0]
    assert len(ax.lines) == 1
    assert not any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close("all")


def test_plot_layers_color_all_mode(trained_deep_network):
    """A user-supplied colour reaches the LineCollection in ``mode='all'``."""
    axs = trained_deep_network.plot_layers(color="#c44e52", mode="all")
    lc = next(c for c in axs[0, 0].collections if isinstance(c, LineCollection))
    rgba = lc.get_colors()[0]
    # #c44e52 = (196, 78, 82) in 0-255 → ≈ (0.769, 0.306, 0.322).
    assert np.allclose(rgba[:3], (0.769, 0.306, 0.322), atol=2e-3)
    plt.close("all")


def test_plot_layers_color_mean_ci_mode(trained_deep_network):
    """A user-supplied colour reaches both the mean line and the CI band."""
    color = (0.2, 0.5, 0.8)
    axs = trained_deep_network.plot_layers(color=color, mode="mean_ci")
    ax = axs[0, 0]
    line_color = ax.lines[0].get_color()
    assert tuple(np.round(line_color, 3)) == color
    band = next(c for c in ax.collections if isinstance(c, PolyCollection))
    band_rgba = band.get_facecolor()[0]
    assert np.allclose(band_rgba[:3], color, atol=1e-6)
    plt.close("all")


def test_plot_layers_user_supplied_axs(trained_deep_network):
    """Passing ``axs`` reuses the array and skips figure creation."""
    fig, my_axs = plt.subplots(2, trained_deep_network.n_layers)
    out = trained_deep_network.plot_layers(
        variables=("expected_mean", "precision"),
        axs=my_axs,
    )
    assert out is my_axs or np.array_equal(out, np.atleast_2d(my_axs))
    plt.close(fig)


def test_plot_layers_axs_shape_mismatch(trained_deep_network):
    """A user-supplied ``axs`` whose shape doesn't match raises ValueError."""
    _, wrong = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="axs.*shape"):
        trained_deep_network.plot_layers(axs=wrong)
    with pytest.raises(ValueError, match="axs.*shape"):
        trained_deep_network.plot_layers(layers=[0], axs=wrong)
    plt.close("all")


def test_plot_layers_invalid_variable(trained_deep_network):
    """Unknown variable names are rejected with a helpful message."""
    with pytest.raises(ValueError, match="Unknown variable"):
        trained_deep_network.plot_layers(variables="not_a_field")
    plt.close("all")


def test_plot_layers_invalid_layer_index(trained_deep_network):
    """Out-of-range layer indices are rejected."""
    with pytest.raises(ValueError, match="out of range"):
        trained_deep_network.plot_layers(layers=[0, 99])
    with pytest.raises(ValueError, match="out of range"):
        trained_deep_network.plot_layers(layers=99)
    plt.close("all")


def test_plot_layers_invalid_mode(trained_deep_network):
    """An unrecognised ``mode`` value raises ValueError."""
    with pytest.raises(ValueError, match="mode must be"):
        trained_deep_network.plot_layers(mode="bogus")
    plt.close("all")


def test_plot_layers_requires_trajectories():
    """Calling without recorded trajectories raises ValueError."""
    net = DeepNetwork().add_layer(size=2).add_layer(size=2)
    with pytest.raises(ValueError, match="record="):
        net.plot_layers()
