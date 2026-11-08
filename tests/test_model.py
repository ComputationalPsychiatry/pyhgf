# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import importlib

import jax.numpy as jnp
import numpy as np
from pandas.testing import assert_frame_equal
from pytest import raises, warns

import pyhgf.model
import pyhgf.model.hgf
from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.response import (
    first_level_binary_surprise,
    first_level_gaussian_surprise,
    total_gaussian_surprise,
)
from pyhgf.typing import UpdateSequence


def test_network():
    """Test the network class."""
    #####################
    # Creating networks #
    #####################
    custom_hgf = (
        Network()
        .add_nodes(kind="continuous-state")
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0)
        .add_nodes(
            value_children=1,
        )
        .add_nodes(value_children=[2, 3])
        .add_nodes(value_children=4)
        .add_nodes(volatility_children=[2, 3])
        .add_nodes(volatility_children=2)
        .add_nodes(volatility_children=7)
        .add_nodes(value_parents=8)
        .add_nodes(volatility_parents=9)
    )

    # sanity check on the network structure
    # ensure that the number of parents and children match the number of coupling values
    for i in range(len(custom_hgf.edges)):
        if custom_hgf.edges[i].node_type == 2:
            # value parents ------------------------------------------------------------
            if custom_hgf.edges[i].value_parents:
                assert len(custom_hgf.edges[i].value_parents) == len(
                    custom_hgf.attributes[i]["value_coupling_parents"]
                )
            else:
                assert (custom_hgf.edges[i].value_parents is None) and (
                    custom_hgf.attributes[i]["value_coupling_parents"] is None
                )

            # value children -----------------------------------------------------------
            if custom_hgf.edges[i].value_children:
                assert len(custom_hgf.edges[i].value_children) == len(
                    custom_hgf.attributes[i]["value_coupling_children"]
                )
            else:
                assert (custom_hgf.edges[i].value_children is None) and (
                    custom_hgf.attributes[i]["value_coupling_children"] is None
                )

            # volatility parents -------------------------------------------------------
            if custom_hgf.edges[i].volatility_parents:
                assert len(custom_hgf.edges[i].volatility_parents) == len(
                    custom_hgf.attributes[i]["volatility_coupling_parents"]
                )
            else:
                assert (custom_hgf.edges[i].volatility_parents is None) and (
                    custom_hgf.attributes[i]["volatility_coupling_parents"] is None
                )

            # volatility children ------------------------------------------------------
            if custom_hgf.edges[i].volatility_children:
                assert len(custom_hgf.edges[i].volatility_children) == len(
                    custom_hgf.attributes[i]["volatility_coupling_children"]
                )
            else:
                assert (custom_hgf.edges[i].volatility_children is None) and (
                    custom_hgf.attributes[i]["volatility_coupling_children"] is None
                )

    custom_hgf.create_belief_propagation_fn(overwrite=False)
    custom_hgf.create_belief_propagation_fn(overwrite=True)

    custom_hgf.input_data(input_data=np.ones((10, 2)), observed=np.ones((10, 2)))

    # expected error for invalid type
    with raises(Exception):
        custom_hgf.add_nodes(kind="error")


def test_constant_state_nodes_reject_invalid_parent_links():
    """Constant-state nodes must not be children of any other node."""
    net = Network().add_nodes(kind="continuous-state")
    net = net.add_nodes(kind="constant-state", value_children=0)

    with raises(ValueError, match="Constant-state nodes cannot have parents"):
        net.add_nodes(kind="continuous-state", value_children=1)

    with raises(
        ValueError, match="Constant-state nodes cannot have volatility children"
    ):
        Network().add_nodes(kind="continuous-state").add_nodes(
            kind="constant-state", volatility_children=0
        )


def test_continuous_hgf():
    """Test the continuous HGF."""
    ##############
    # Continuous #
    ##############
    timeserie = load_data("continuous")

    # two-level
    # ---------
    two_level_continuous_hgf = (
        Network()
        .add_nodes(precision=1e4)
        .add_nodes(
            value_children=([0], [1.0]),
            node_parameters={
                "mean": timeserie[0],
                "precision": 1e4,
                "tonic_volatility": -3.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 0.0,
                "precision": 1e1,
                "tonic_volatility": -3.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
    )

    two_level_continuous_hgf.input_data(input_data=timeserie)

    # Sum the surprise for this model
    surprise = two_level_continuous_hgf.surprise(
        response_function=first_level_gaussian_surprise
    )
    assert jnp.isclose(surprise.sum(), -1924.7515)
    assert len(two_level_continuous_hgf.node_trajectories[1]["mean"]) == 614

    # three-level
    # -----------
    three_level_continuous_hgf = (
        Network()
        .add_nodes(precision=1e4)
        .add_nodes(
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 1.04,
                "precision": 1e4,
                "tonic_volatility": -13.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 1.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([2], [1.0]),
            node_parameters={
                "mean": 1.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
    )
    three_level_continuous_hgf.input_data(input_data=timeserie)
    surprise = three_level_continuous_hgf.surprise(
        response_function=first_level_gaussian_surprise
    )
    assert jnp.isclose(surprise.sum(), -2034.3989)

    # test an alternative response function
    sp = total_gaussian_surprise(three_level_continuous_hgf)
    assert jnp.isclose(sp.sum(), -2535.604)


def test_binary_hgf():
    """Test the binary HGF."""
    ##########
    # Binary #
    ##########
    u, _ = load_data("binary")

    # two-level
    # ---------
    two_level_binary_hgf = (
        Network()
        .add_nodes(
            kind="binary-state",
            node_parameters={"mean": 0.0, "precision": 0.0},
        )
        .add_nodes(
            kind="continuous-state",
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 0.5,
                "precision": 1e4,
                "tonic_volatility": -6.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
    )

    # Provide new observations
    two_level_binary_hgf = two_level_binary_hgf.input_data(u)
    surprise = two_level_binary_hgf.surprise(
        response_function=first_level_binary_surprise
    )
    assert jnp.isclose(surprise.sum(), 215.58821)

    # three-level
    # -----------
    three_level_binary_hgf = (
        Network()
        .add_nodes(
            kind="binary-state",
            node_parameters={"mean": 0.0, "precision": 0.0},
        )
        .add_nodes(
            kind="continuous-state",
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 0.5,
                "precision": 1e4,
                "tonic_volatility": -6.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 0.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
    )
    three_level_binary_hgf.input_data(input_data=u)
    surprise = three_level_binary_hgf.surprise(
        response_function=first_level_binary_surprise
    )
    assert jnp.isclose(surprise.sum(), 1242.3856)


def test_custom_sequence():
    """Test the continuous HGF."""
    ############################
    # dynamic update sequences #
    ############################
    u, _ = load_data("binary")

    three_level_binary_hgf = (
        Network()
        .add_nodes(
            kind="binary-state",
            node_parameters={"mean": 0.0, "precision": 0.0},
        )
        .add_nodes(
            kind="continuous-state",
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 0.5,
                "precision": 1e4,
                "tonic_volatility": -6.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 0.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
    )

    # create a custom update series
    update_sequence1: UpdateSequence = three_level_binary_hgf.update_sequence
    update_sequence2: UpdateSequence = three_level_binary_hgf.update_sequence
    update_branches = (update_sequence1, update_sequence2)
    branches_idx = np.random.binomial(n=1, p=0.5, size=len(u))

    three_level_binary_hgf.scan_fn = None
    three_level_binary_hgf.input_custom_sequence(
        update_branches=update_branches,
        branches_idx=branches_idx,
        input_data=u,
    )


def test_network_input_data_no_trajectories():
    """Test Network.input_data() with record_trajectories=False."""
    timeserie = load_data("continuous")

    hgf = (
        Network()
        .add_nodes(precision=1e4)
        .add_nodes(
            value_children=([0], [1.0]),
            node_parameters={
                "mean": 1.04,
                "precision": 1e4,
                "tonic_volatility": -13.0,
                "tonic_drift": 0.0,
            },
        )
        .add_nodes(
            volatility_children=([1], [1.0]),
            node_parameters={
                "mean": 1.0,
                "precision": 1e1,
                "tonic_volatility": -2.0,
                "tonic_drift": 0.0,
            },
        )
        .create_belief_propagation_fn()
        .input_data(input_data=timeserie, record_trajectories=False)
    )

    assert hgf.node_trajectories is None
    assert hgf.last_attributes is not None


def test_network_input_data_batches():
    """Compare 20 observations processed at once or in four batches."""
    observations = np.linspace(0.0, 1.0, 20)

    full_network = Network().add_nodes().add_nodes(value_children=0)
    batch_network = Network().add_nodes().add_nodes(value_children=0)

    # First network: all 20 observations at once.
    full_network.input_data(observations)

    # Second network: the same observations in four batches of five.
    for batch in np.split(observations, 4):
        batch_network.input_data(batch, append=True)

    # Compare the full recorded history of both networks.
    assert_frame_equal(
        full_network.to_pandas(),
        batch_network.to_pandas(),
        check_exact=False,
        rtol=1e-6,
        atol=1e-7,
    )


def test_input_idxs_route_observations_to_the_selected_nodes():
    """Changing, reordering or resetting the selection reroutes the data columns."""
    net = Network().add_nodes(n_nodes=2, mean=3.0)
    net.input_data(np.array([1.0, 2.0]), input_idxs=(0,))
    carry = net.last_attributes
    net.attributes[1]["precision"] = 9.0

    # A new selection keeps the parameters and the final beliefs.
    net.input_idxs = (1,)
    assert net.last_attributes is carry
    assert net.attributes[1]["precision"] == 9.0
    net.input_data(np.array([11.0, 12.0]), input_idxs=(1,))
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], [11.0, 12.0])
    np.testing.assert_array_equal(net.node_trajectories[0]["mean"], [3.0, 3.0])

    observations = np.array([[11.0, 12.0], [13.0, 14.0]])
    net.input_data(observations, input_idxs=(1, 0))
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], [11.0, 13.0])
    np.testing.assert_array_equal(net.node_trajectories[0]["mean"], [12.0, 14.0])

    net.input_idxs = None
    net.input_data(observations)
    assert net.input_idxs == (0, 1)
    np.testing.assert_array_equal(net.node_trajectories[0]["mean"], [11.0, 13.0])


def test_input_idxs_selection_keeps_parameters_and_rebuilds_dimensions():
    """An explicit selection leaves parameters as set and rebuilds input dimensions."""
    net = Network().add_nodes(
        n_nodes=2, autoconnection_strength=0.8, tonic_volatility=-7.0
    )
    net.input_data(np.array([5.0, 6.0]), input_idxs=np.array([1]))
    assert net.input_idxs == (1,)
    for i in range(2):
        assert net.attributes[i]["autoconnection_strength"] == 0.8
        assert net.attributes[i]["tonic_volatility"] == -7.0

    net = (
        Network()
        .add_nodes(kind="ef-state")
        .add_nodes(kind="ef-state", distribution="multivariate-normal", dimension=2)
    )
    net.input_idxs = (0,)
    net.create_belief_propagation_fn()
    assert net.input_dim == [1]
    observations = np.array([[1.0, 2.0], [3.0, 4.0]])
    net.input_data(observations, input_idxs=(1,))
    assert net.input_dim == [2]
    np.testing.assert_array_equal(net.node_trajectories[1]["mean"], observations)


def test_same_input_idxs_keep_history_and_parameter_edits():
    """Reselecting the same nodes keeps appended history and later parameter edits."""
    observations = np.linspace(0.0, 1.0, 6)

    def build(**parameters):
        return Network().add_nodes().add_nodes(value_children=0, **parameters)

    net = build().input_data(observations[:3], input_idxs=(0,), append=True)
    net.input_idxs = (0,)
    net.input_data(observations[3:], input_idxs=(0,), append=True)
    reference = build().input_data(observations, input_idxs=(0,))
    assert_frame_equal(net.to_pandas(), reference.to_pandas(), rtol=1e-6, atol=1e-7)

    net.attributes[1]["tonic_drift"] = 0.5
    net.input_data(observations, input_idxs=(0,))
    reference = build(tonic_drift=0.5).input_data(observations, input_idxs=(0,))
    assert_frame_equal(net.to_pandas(), reference.to_pandas(), rtol=1e-6, atol=1e-7)


def test_custom_sequence_append():
    """``append=True`` continues either method; other calls restart from priors."""

    def build():
        return Network().add_nodes().add_nodes(value_children=0)

    def run(net, method, observations, **kwargs):
        if method == "data":
            return net.input_data(observations, **kwargs)
        return net.input_custom_sequence(
            update_branches=(net.get_network()[2],),
            branches_idx=np.zeros(len(observations), dtype=int),
            input_data=observations,
            **kwargs,
        )

    observations = np.array([0.0, 1.0, 2.0, 3.0])
    reference = build().input_data(observations).to_pandas()

    for first, second in [("custom", "data"), ("data", "custom"), ("custom", "custom")]:
        net = run(build(), first, observations[:2])
        run(net, second, observations[2:], append=True)
        assert_frame_equal(net.to_pandas(), reference, rtol=1e-6, atol=1e-7)

    for method in ("custom", "data"):
        assert_frame_equal(
            run(net, method, observations).to_pandas(), reference, rtol=1e-6, atol=1e-7
        )


def test_hgf_class_is_deprecated():
    """Loading the deprecated HGF module should display the warning."""
    with warns(DeprecationWarning, match="deprecated"):
        importlib.reload(pyhgf.model.hgf)

    # accessing the removed `HGF` class from the package should also fail
    with raises(ImportError, match="deprecated"):
        pyhgf.model.HGF
