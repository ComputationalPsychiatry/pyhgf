# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

"""Parity tests: vectorized continuous layers vs.

the nodalised backend. Every test builds the same topology twice — once with the
nodalised :class:`pyhgf.model.Network`, once with the vectorized
:class:`pyhgf.model.DeepNetwork` continuous layers — runs the same observations through
both, and compares the per-node trajectories.

The tests at the bottom of the file exercise the update kernels and the sweep guards
directly, for the paths the builder cannot reach on its own.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from pyhgf.model import Network
from pyhgf.model.deep_network import DeepNetwork
from pyhgf.typing.vectorised import LayerParams, LayerStack, LayerState
from pyhgf.updates.vectorized.continuous import (
    ValueChild,
    VolatilityChild,
    vectorized_continuous_posterior_update,
    vectorized_continuous_posterior_update_ehgf,
    vectorized_continuous_posterior_update_standard,
)
from pyhgf.utils.vectorized_belief_propagation import run_continuous_scan

U = jnp.array([0.2, 0.5, -0.3, 1.0, 0.1, -0.7, 0.4])
FIELDS = ("mean", "precision", "expected_mean", "expected_precision")


def _state(**fields) -> LayerState:
    """Build a one-node continuous ``LayerState``, overriding the named fields."""
    base = LayerState.create_continuous(1, has_volatility_parent=True)
    return dataclasses.replace(base, **{k: jnp.array([v]) for k, v in fields.items()})


def assert_parity(nod: Network, vec: DeepNetwork, node_to_layer: dict):
    """Compare nodalised node trajectories against vectorized layer trajectories.

    ``node_to_layer`` maps a nodalised node index to ``(layer_idx, node_idx)`` in the
    vectorized network.
    """
    for node, (layer, pos) in node_to_layer.items():
        for field in FIELDS:
            a = nod.node_trajectories[node][field]
            b = vec.trajectories[field][layer][:, pos]
            assert jnp.allclose(a, b, rtol=1e-5, atol=1e-6), (
                f"node {node} / layer {layer}[{pos}] diverge on {field}: {a} vs {b}"
            )


def test_two_level_chain():
    """Observation node with one value parent (the one-node HGF)."""
    nod = (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0)})


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF", "unbounded"])
def test_volatility_children_leaf(volatility_updates):
    """Leaf with both a value parent and a volatility parent (two-node HGF)."""
    nod = (
        Network(volatility_updates=volatility_updates)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=0)
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates)
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            volatility_children=0,
            volatility_fully_connected=(volatility_updates != "unbounded"),
        )
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF", "unbounded"])
def test_classic_three_level(volatility_updates):
    """The classic 3-level continuous HGF: x2 is the volatility parent of x1."""
    nod = (
        Network(volatility_updates=volatility_updates)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates)
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            volatility_children=1,
            volatility_fully_connected=(volatility_updates != "unbounded"),
        )
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF", "unbounded"])
def test_mean_field_three_level(volatility_updates):
    """Mean-field parity on the classic 3-level continuous HGF.

    Both backends run with ``mean_field_updates=True``: no MGF correction in the log-
    volatility exponent, no Laplace value-coupling variance, and canonical child
    precisions in the posterior messages.
    """
    nod = (
        Network(volatility_updates=volatility_updates, mean_field_updates=True)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates, mean_field_updates=True)
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            volatility_children=1,
            volatility_fully_connected=(volatility_updates != "unbounded"),
        )
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF"])
def test_mean_field_fully_connected(volatility_updates):
    """Mean-field parity on dense width-2 layers with a dense volatility parent."""
    nod = (
        Network(volatility_updates=volatility_updates, mean_field_updates=True)
        .add_nodes(n_nodes=2)
        .add_nodes(n_nodes=2, value_children=([0, 1], [0.5, 0.5]))
        .add_nodes(volatility_children=[2, 3])
        .input_data(input_data=jnp.stack([U, U * 0.5 + 0.1], axis=1))
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates, mean_field_updates=True)
        .add_layer(2, kind="continuous")
        .add_layer(2, kind="continuous")
        .add_layer(1, kind="continuous", volatility_children=1)
        .input_data(jnp.stack([U, U * 0.5 + 0.1], axis=1), record=FIELDS)
    )
    assert_parity(
        nod,
        vec,
        {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0)},
    )


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF"])
def test_fully_connected_layers(volatility_updates):
    """Width-2 layers with dense value coupling and a dense volatility parent.

    Nodalised equivalent: two leaf nodes, two value parents each coupled to
    both leaves, and one volatility parent modulating both value parents.
    """
    nod = (
        Network(volatility_updates=volatility_updates)
        .add_nodes(n_nodes=2)
        .add_nodes(n_nodes=2, value_children=([0, 1], [0.5, 0.5]))
        .add_nodes(volatility_children=[2, 3])
        .input_data(input_data=jnp.stack([U, U * 0.5 + 0.1], axis=1))
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates)
        .add_layer(2, kind="continuous")
        .add_layer(2, kind="continuous")
        .add_layer(1, kind="continuous", volatility_children=1)
        .input_data(jnp.stack([U, U * 0.5 + 0.1], axis=1), record=FIELDS)
    )
    assert_parity(
        nod,
        vec,
        {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0)},
    )


def test_unbounded_one_to_one_multi_node():
    """Width-2 one-to-one volatility coupling under the unbounded update."""
    nod = (
        Network(volatility_updates="unbounded")
        .add_nodes(n_nodes=2)
        .add_nodes(n_nodes=2, value_children=([0, 1], [0.5, 0.5]))
        .add_nodes(volatility_children=2)
        .add_nodes(volatility_children=3)
        .input_data(input_data=jnp.stack([U, U * 0.5 + 0.1], axis=1))
    )
    vec = (
        DeepNetwork(volatility_updates="unbounded")
        .add_layer(2, kind="continuous")
        .add_layer(2, kind="continuous")
        .add_layer(
            2,
            kind="continuous",
            volatility_children=1,
            volatility_fully_connected=False,
        )
        .input_data(jnp.stack([U, U * 0.5 + 0.1], axis=1), record=FIELDS)
    )
    assert_parity(
        nod,
        vec,
        {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0), 5: (2, 1)},
    )


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF"])
def test_dual_role_parent(volatility_updates):
    """A layer that is a value parent of one layer and volatility parent of another."""
    nod = (
        Network(volatility_updates=volatility_updates)
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(value_children=1, volatility_children=0)
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates)
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            value_children=1,
            volatility_children=0,
        )
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


@pytest.mark.parametrize("volatility_updates", ["standard", "eHGF"])
def test_nonlinear_coupling(volatility_updates):
    """A sinusoidal coupling function on the value edge."""
    nod = (
        Network(volatility_updates=volatility_updates)
        .add_nodes()
        .add_nodes(value_children=0, coupling_fn=(jnp.sin,))
        .add_nodes(volatility_children=0)
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates=volatility_updates)
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous", coupling_fn=jnp.sin)
        .add_layer(1, kind="continuous", volatility_children=0)
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


def test_irregular_time_steps():
    """Non-uniform inter-observation intervals."""
    time_steps = jnp.array([1.0, 0.5, 2.0, 1.5, 1.0, 3.0, 0.1])
    nod = (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .input_data(input_data=U, time_steps=time_steps)
    )
    vec = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous", volatility_children=1)
        .input_data(U, time_steps=time_steps, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


def test_parameter_overrides():
    """Per-layer parameter overrides reach the filter."""
    nod = (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(
            value_children=0,
            node_parameters={
                "tonic_volatility": -2.0,
                "tonic_drift": 0.1,
                "autoconnection_strength": 0.9,
            },
        )
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            tonic_volatility=-2.0,
            tonic_drift=0.1,
            autoconnection_strength=0.9,
        )
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0)})


def test_builder_validation():
    """Topology and API constraints raise clear errors."""
    # continuous cannot mix with volatile
    with pytest.raises(ValueError, match="cannot mix"):
        DeepNetwork().add_layer(2).add_layer(2, kind="continuous")
    with pytest.raises(ValueError, match="cannot mix"):
        DeepNetwork().add_layer(2, kind="continuous").add_layer(2)

    # no bias column on continuous layers
    with pytest.raises(ValueError, match="tonic_drift"):
        DeepNetwork().add_layer(2, kind="continuous", add_constant_input=True)

    # every continuous-only argument is rejected on other kinds rather than
    # silently ignored
    for kwargs in (
        {"value_children": 0},
        {"volatility_children": 0},
        {"value_coupling": 0.5},
        {"volatility_coupling": 0.5},
        {"volatility_fully_connected": False},
    ):
        with pytest.raises(ValueError, match="only valid for continuous layers"):
            DeepNetwork().add_layer(2).add_layer(2, **kwargs)

    # child indices must name an existing layer
    with pytest.raises(IndexError, match="does not name an existing layer"):
        (
            DeepNetwork()
            .add_layer(2, kind="continuous")
            .add_layer(2, kind="continuous", volatility_children=5)
        )

    # a child can have at most one parent of each type
    with pytest.raises(ValueError, match="already has a value parent"):
        (
            DeepNetwork()
            .add_layer(2, kind="continuous")
            .add_layer(2, kind="continuous")
            .add_layer(2, kind="continuous", value_children=0)
        )
    with pytest.raises(ValueError, match="already has a volatility parent"):
        (
            DeepNetwork(volatility_updates="standard")
            .add_layer(2, kind="continuous")
            .add_layer(2, kind="continuous", volatility_children=0)
            .add_layer(2, kind="continuous", volatility_children=0)
        )

    # a one-to-one κ needs matching widths
    with pytest.raises(ValueError, match="requires equal sizes"):
        (
            DeepNetwork()
            .add_layer(2, kind="continuous")
            .add_layer(
                3,
                kind="continuous",
                volatility_children=0,
                volatility_fully_connected=False,
            )
        )

    # unbounded requires one-to-one volatility coupling
    with pytest.raises(ValueError, match="one-to-one"):
        (
            DeepNetwork(volatility_updates="unbounded")
            .add_layer(2, kind="continuous")
            .add_layer(2, kind="continuous")
            .add_layer(2, kind="continuous", volatility_children=1)
        )

    # ...and rejects a parent that is both a value and a volatility parent
    with pytest.raises(ValueError, match="cannot also be a value parent"):
        (
            DeepNetwork(volatility_updates="unbounded")
            .add_layer(1, kind="continuous")
            .add_layer(1, kind="continuous")
            .add_layer(
                1,
                kind="continuous",
                value_children=1,
                volatility_children=0,
                volatility_fully_connected=False,
            )
        )

    # kind-specific parameter overrides
    with pytest.raises(ValueError, match="do not apply"):
        DeepNetwork().add_layer(2, tonic_drift=0.1)
    with pytest.raises(ValueError, match="do not apply"):
        DeepNetwork().add_layer(2, kind="continuous", tonic_volatility_vol=-2.0)

    # deep-network entry points are rejected on continuous networks
    import optax

    net = DeepNetwork().add_layer(2, kind="continuous").add_layer(2, kind="continuous")
    with pytest.raises(ValueError, match="input_data"):
        net.fit(jnp.zeros((3, 2)), jnp.zeros((3, 2)), optimizer=optax.sgd(0.1))
    with pytest.raises(ValueError, match="input_data"):
        net.predict(jnp.zeros((3, 2)))

    # and input_data is rejected on volatile networks
    with pytest.raises(ValueError, match="continuous"):
        DeepNetwork().add_layer(2).add_layer(2).input_data(jnp.zeros((3, 2)))

    # observations must be as wide as layer 0
    continuous = (
        DeepNetwork().add_layer(2, kind="continuous").add_layer(2, kind="continuous")
    )
    with pytest.raises(ValueError, match="but layer 0 has"):
        continuous.input_data(jnp.zeros((3, 5)))


def test_record_field_validation():
    """``record`` only accepts fields the network actually allocates."""
    net = DeepNetwork().add_layer(1, kind="continuous").add_layer(1, kind="continuous")

    # Continuous layers leave the internal-volatility fields at ``None``, so
    # recording them would silently yield nothing.
    with pytest.raises(ValueError, match="Unknown record field"):
        net.input_data(U, record=("mean_vol",))
    with pytest.raises(ValueError, match="Unknown record field"):
        net.input_data(U, record=("not_a_field",))

    # ``volatility_prediction_error`` is allocated only where a volatility
    # parent exists, so it is recordable on that topology and not otherwise.
    with pytest.raises(ValueError, match="Unknown record field"):
        net.input_data(U, record=("volatility_prediction_error",))

    with_vol = (
        DeepNetwork()
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous", volatility_children=1)
        .input_data(U, record=("volatility_prediction_error",))
    )
    assert with_vol.trajectories["volatility_prediction_error"][1].shape == (len(U), 1)


def test_volatility_coupling_strength():
    """An explicit coupling strength reaches the filter (fan-in 1: no scaling)."""
    nod = (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=([0], [0.7]))
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(
            1,
            kind="continuous",
            volatility_children=0,
            volatility_coupling=0.7,
        )
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0), 2: (2, 0)})


def test_coupling_fan_in_normalisation():
    """Dense W and κ entries are strength / n_parents; one-to-one keeps the strength."""
    dense = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(2, kind="continuous")
        .add_layer(2, kind="continuous", value_coupling=0.6)
        .add_layer(4, kind="continuous", volatility_children=1, volatility_coupling=0.8)
    )
    weights = dense.state.layers[1].weights_mean
    assert weights.shape == (2, 2)
    assert jnp.allclose(weights, 0.6 / 2)
    kappa = dense.state.layers[2].volatility_weights
    assert kappa.shape == (2, 4)
    assert jnp.allclose(kappa, 0.8 / 4)

    one_to_one = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(2, kind="continuous")
        .add_layer(2, kind="continuous")
        .add_layer(
            2,
            kind="continuous",
            volatility_children=1,
            volatility_coupling=0.8,
            volatility_fully_connected=False,
        )
    )
    kappa = one_to_one.state.layers[2].volatility_weights
    assert jnp.allclose(kappa, 0.8 * jnp.eye(2))


def test_value_coupling_strength():
    """An explicit value coupling strength reaches the filter (fan-in 1)."""
    nod = (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(value_children=([0], [0.7]))
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous", value_coupling=0.7)
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0)})


def test_state_overrides():
    """Per-layer state overrides reach the filter."""
    nod = (
        Network(volatility_updates="standard")
        .add_nodes()
        .add_nodes(
            value_children=0,
            node_parameters={"mean": 0.5, "precision": 2.0},
        )
        .input_data(input_data=U)
    )
    vec = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous", mean=0.5, precision=2.0)
        .input_data(U, record=FIELDS)
    )
    assert_parity(nod, vec, {0: (0, 0), 1: (1, 0)})


def test_input_data_without_record():
    """Without ``record`` only the one-step-ahead predictions are kept."""

    def build():
        return (
            DeepNetwork(volatility_updates="standard")
            .add_layer(1, kind="continuous")
            .add_layer(1, kind="continuous")
        )

    plain = build().input_data(U)
    assert plain.trajectories is None
    assert plain.predictions.shape == (len(U), 1)

    # Recording changes what is reported, never what is filtered.
    recorded = build().input_data(U, record=FIELDS)
    assert jnp.allclose(plain.predictions, recorded.predictions)
    assert jnp.allclose(plain.predictions, recorded.trajectories["expected_mean"][0])


def test_continuous_sweeps_reject_other_networks():
    """The continuous scan refuses networks it cannot interpret."""
    volatile = DeepNetwork().add_layer(2).add_layer(2)
    with pytest.raises(ValueError, match="all-continuous network"):
        run_continuous_scan(volatile.state, jnp.zeros((3, 2)), jnp.ones(3))

    # Continuous layers cannot be stacked and cannot mix with volatile ones, so
    # the only way to present a LayerStack to the sweep is to splice one in.
    stacked = (
        DeepNetwork()
        .add_layer(size=4)
        .add_layer_stack(layer_sizes=[4] * 5)
        .add_layer(size=2)
    )
    stack = next(e for e in stacked.state.layers if isinstance(e, LayerStack))
    continuous = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
    )
    spliced = dataclasses.replace(continuous.state, layers=(stack,))
    with pytest.raises(NotImplementedError, match="LayerStack"):
        run_continuous_scan(spliced, jnp.zeros((3, 4)), jnp.ones(3))


def test_layer_without_children_is_skipped():
    """A layer nobody names as parent receives no message and stays at its prior."""
    net = (
        DeepNetwork(volatility_updates="standard")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
        .add_layer(1, kind="continuous")
    )
    # Detach the top layer: the builder's chain default always gives a layer a
    # child, so this topology can only be reached by editing the PyTree.
    elements = list(net.state.layers)
    elements[2] = dataclasses.replace(
        elements[2], value_child_idx=None, volatility_child_idx=None
    )
    net.state = dataclasses.replace(net.state, layers=tuple(elements))
    detached = net.state.layers[2].state

    net.input_data(U, record=("mean", "precision"))

    assert jnp.allclose(net.trajectories["mean"][2], detached.mean)
    assert jnp.allclose(net.trajectories["precision"][2], detached.precision)
    # The two layers that are still connected keep filtering.
    assert not jnp.allclose(net.trajectories["mean"][1], detached.mean)


def _value_child() -> ValueChild:
    """Build a one-node value child with a linear coupling."""
    return ValueChild(
        state=_state(
            precision=3.0,
            expected_precision=2.0,
            conditional_expected_precision=2.5,
            value_prediction_error=0.4,
        ),
        weights=jnp.array([[0.8]]),
        coupling_fn=lambda x: x,
        precision_is_clamped=False,
    )


def _volatility_child() -> VolatilityChild:
    """Build a one-node volatility child."""
    return VolatilityChild(
        state=_state(
            mean=0.2,
            precision=2.0,
            expected_mean=0.1,
            expected_precision=1.6,
            conditional_expected_precision=1.8,
            effective_precision=0.3,
            volatility_prediction_error=0.5,
        ),
        kappa=jnp.array([[1.0]]),
        params=LayerParams.create_continuous(1),
    )


def test_ehgf_with_a_value_child_only():
    """EHGF differs from the standard scheme only in the mean's denominator.

    The dispatcher never routes a layer without a volatility child to the eHGF update,
    so this exercises the kernel directly.
    """
    parent = _state(mean=0.3, precision=1.5, expected_mean=0.25, expected_precision=1.2)
    child = _value_child()

    standard = vectorized_continuous_posterior_update_standard(
        parent, value_child=child
    )
    ehgf = vectorized_continuous_posterior_update_ehgf(parent, value_child=child)

    # A linear coupling has g'' = 0, so the precision increment does not depend
    # on where the derivatives are evaluated and the two schemes agree.
    assert jnp.allclose(standard.precision, ehgf.precision)

    # The mean shift shares a numerator: the standard scheme divides it by the
    # posterior precision, eHGF by the predicted one.
    assert not jnp.allclose(standard.mean, ehgf.mean)
    assert jnp.allclose(
        (ehgf.mean - parent.expected_mean) * parent.expected_precision,
        (standard.mean - parent.expected_mean) * standard.precision,
    )


def test_posterior_dispatch_errors():
    """The dispatcher rejects an unusable scheme rather than dropping an edge."""
    parent = _state(mean=0.3, precision=1.5, expected_mean=0.25, expected_precision=1.2)

    with pytest.raises(ValueError, match="pure volatility parents"):
        vectorized_continuous_posterior_update(
            parent,
            value_child=_value_child(),
            volatility_child=_volatility_child(),
            volatility_updates="unbounded",
        )

    with pytest.raises(ValueError, match="Invalid volatility_updates"):
        vectorized_continuous_posterior_update(
            parent,
            volatility_child=_volatility_child(),
            volatility_updates="nonsense",
        )
