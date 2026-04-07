# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Aleksandrs Baskakovs <aleks@cas.au.dk>

"""Test that VDN produces identical node trajectories to a hand-wired vanilla Network.

A 2->4->2 network is built two ways:
  1. Vanilla Network  — hand-wired nodes, ground truth
  2. VectorizedDeepNetwork (VDN) — layer-wise matrix ops

Both start from identical coupling weights and node parameters, are fed the same data
sequence, and should produce matching posterior means, precisions, and implicit
volatility states at every time step.
"""

import jax.numpy as jnp
import numpy as np

from pyhgf.model import Network, VectorizedDeepNetwork

# ──────────────────────────────────────────────────────────────────────────────
# Shared test configuration
# ──────────────────────────────────────────────────────────────────────────────
N_SAMPLES = 5
COUPLING = 0.5
TONIC_VOL = -2.0
LR = 0.0  # no weight learning — pure trajectory comparison
ATOL = 1e-7  # tolerance for VDN vs node-level comparison


def _make_data(seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(N_SAMPLES, 2).astype(np.float32) * 0.5
    y = rng.randn(N_SAMPLES, 2).astype(np.float32) * 0.5
    return x, y


def _build_vanilla_network():
    net = Network()
    for _ in range(2):
        net.add_nodes(
            kind="volatile-state",
            tonic_volatility=TONIC_VOL,
            autoconnection_strength=0.0,
        )
    for _ in range(4):
        net.add_nodes(
            kind="volatile-state",
            value_children=([0, 1], [COUPLING, COUPLING]),
            coupling_fn=(jnp.tanh,),
            tonic_volatility=TONIC_VOL,
            autoconnection_strength=0.0,
        )
    for _ in range(2):
        net.add_nodes(
            kind="volatile-state",
            value_children=([2, 3, 4, 5], [COUPLING] * 4),
            coupling_fn=(jnp.tanh,),
            tonic_volatility=TONIC_VOL,
            autoconnection_strength=0.0,
        )
    return net


def _build_vdn():
    vdn = (
        VectorizedDeepNetwork(coupling_fn=jnp.tanh)
        .add_layer(
            size=2,
            tonic_volatility=TONIC_VOL,
            volatility_coupling=1.0,
            add_constant_input=False,
        )
        .add_layer(
            size=4,
            tonic_volatility=TONIC_VOL,
            volatility_coupling=1.0,
            add_constant_input=False,
        )
        .add_layer(
            size=2,
            tonic_volatility=TONIC_VOL,
            volatility_coupling=1.0,
            add_constant_input=False,
        )
    )
    state = vdn._init_state()
    state = state._replace(
        weights=tuple(jnp.full_like(w, COUPLING) for w in state.weights)
    )
    vdn.state = state
    return vdn


def _fit_both(seed=42):
    """Build, fit, and return (vanilla, vdn) on the same data."""
    x, y = _make_data(seed)
    vanilla = _build_vanilla_network()
    vdn = _build_vdn()
    vanilla.fit(x=x, y=y, inputs_x_idxs=(6, 7), inputs_y_idxs=(0, 1), lr=LR)
    vdn.fit(x, y, lr=LR, reset_state=False)
    return vanilla, vdn


# Node index mapping: vanilla node idx → (VDN layer, position within layer)
# Layer 0 = output (nodes 0-1), layer 1 = hidden (nodes 2-5), layer 2 = input (nodes 6-7)
LAYER_MAP = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
    4: (1, 2),
    5: (1, 3),
    6: (2, 0),
    7: (2, 1),
}
METRICS = ("mean", "precision", "mean_vol", "precision_vol")


class TestTrajectoryEquivalence:
    """Test that VDN produces identical trajectories to a hand-wired vanilla Network."""

    def test_vdn_vs_vanilla_output_layer(self):
        """VDN output layer (layer 0) matches vanilla nodes 0-1."""
        vanilla, vdn = _fit_both()
        for node_idx in [0, 1]:
            layer_idx, pos = LAYER_MAP[node_idx]
            for metric in METRICS:
                van_val = np.array(vanilla.node_trajectories[node_idx][metric])
                vdn_val = np.array(
                    getattr(vdn.trajectories.layers[layer_idx], metric)[:, pos]
                )
                np.testing.assert_allclose(
                    van_val,
                    vdn_val,
                    atol=ATOL,
                    err_msg=f"output layer: {metric} mismatch at node {node_idx}",
                )

    def test_vdn_vs_vanilla_hidden_layer(self):
        """VDN hidden layer (layer 1) matches vanilla nodes 2-5."""
        vanilla, vdn = _fit_both()
        for node_idx in [2, 3, 4, 5]:
            layer_idx, pos = LAYER_MAP[node_idx]
            for metric in METRICS:
                van_val = np.array(vanilla.node_trajectories[node_idx][metric])
                vdn_val = np.array(
                    getattr(vdn.trajectories.layers[layer_idx], metric)[:, pos]
                )
                np.testing.assert_allclose(
                    van_val,
                    vdn_val,
                    atol=ATOL,
                    err_msg=f"hidden layer: {metric} mismatch at node {node_idx}",
                )

    def test_vdn_vs_vanilla_input_layer(self):
        """VDN input layer (layer 2) matches vanilla nodes 6-7."""
        vanilla, vdn = _fit_both()
        for node_idx in [6, 7]:
            layer_idx, pos = LAYER_MAP[node_idx]
            for metric in METRICS:
                van_val = np.array(vanilla.node_trajectories[node_idx][metric])
                vdn_val = np.array(
                    getattr(vdn.trajectories.layers[layer_idx], metric)[:, pos]
                )
                np.testing.assert_allclose(
                    van_val,
                    vdn_val,
                    atol=ATOL,
                    err_msg=f"input layer: {metric} mismatch at node {node_idx}",
                )
