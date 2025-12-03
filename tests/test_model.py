# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np
from pytest import raises

from pyhgf import load_data
from pyhgf.model import HGF, Network
from pyhgf.response import total_gaussian_surprise
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


def test_continuous_hgf():
    """Test the continuous HGF."""
    ##############
    # Continuous #
    ##############
    timeserie = load_data("continuous")

    # two-level
    # ---------
    two_level_continuous_hgf = HGF(
        n_levels=2,
        model_type="continuous",
        initial_mean={"1": timeserie[0], "2": 0.0},
        initial_precision={"1": 1e4, "2": 1e1},
        tonic_volatility={"1": -3.0, "2": -3.0},
        tonic_drift={"1": 0.0, "2": 0.0},
        volatility_coupling={"1": 1.0},
    )

    two_level_continuous_hgf.input_data(input_data=timeserie)

    surprise = two_level_continuous_hgf.surprise()  # Sum the surprise for this model
    assert jnp.isclose(surprise.sum(), -1141.0911)
    assert len(two_level_continuous_hgf.node_trajectories[1]["mean"]) == 614

    # three-level
    # -----------
    three_level_continuous_hgf = HGF(
        n_levels=3,
        model_type="continuous",
        initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
        initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
        tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
        tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": 1.0, "2": 1.0},
    )
    three_level_continuous_hgf.input_data(input_data=timeserie)
    surprise = three_level_continuous_hgf.surprise()
    assert jnp.isclose(surprise.sum(), -892.82227)

    # test an alternative response function
    sp = total_gaussian_surprise(three_level_continuous_hgf)
    assert jnp.isclose(sp.sum(), -2545.4248)


def test_binary_hgf():
    """Test the binary HGF."""
    ##########
    # Binary #
    ##########
    u, _ = load_data("binary")

    # two-level
    # ---------
    two_level_binary_hgf = HGF(
        n_levels=2,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5},
        initial_precision={"1": 0.0, "2": 1e4},
        tonic_volatility={"1": None, "2": -6.0},
        tonic_drift={"1": None, "2": 0.0},
        volatility_coupling={"1": None},
        binary_precision=jnp.inf,
    )

    # Provide new observations
    two_level_binary_hgf = two_level_binary_hgf.input_data(u)
    surprise = two_level_binary_hgf.surprise()
    assert jnp.isclose(surprise.sum(), 215.58821)

    # three-level
    # -----------
    three_level_binary_hgf = HGF(
        n_levels=3,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5, "3": 0.0},
        initial_precision={"1": 0.0, "2": 1e4, "3": 1e1},
        tonic_volatility={"1": None, "2": -6.0, "3": -2.0},
        tonic_drift={"1": None, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": None, "2": 1.0},
        binary_precision=jnp.inf,
    )
    three_level_binary_hgf.input_data(input_data=u)
    surprise = three_level_binary_hgf.surprise()
    assert jnp.isclose(surprise.sum(), 215.59067)


def test_custom_sequence():
    """Test the continuous HGF."""
    ############################
    # dynamic update sequences #
    ############################
    u, _ = load_data("binary")

    three_level_binary_hgf = HGF(
        n_levels=3,
        model_type="binary",
        initial_mean={"1": 0.0, "2": 0.5, "3": 0.0},
        initial_precision={"1": 0.0, "2": 1e4, "3": 1e1},
        tonic_volatility={"1": None, "2": -6.0, "3": -2.0},
        tonic_drift={"1": None, "2": 0.0, "3": 0.0},
        volatility_coupling={"1": None, "2": 1.0},
        binary_precision=jnp.inf,
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

def test_add_value_parent_layer():
    """Test building a fully connected parent layer."""
    net = Network()

    # Create 4 bottom nodes
    net = net.add_nodes(kind="continuous-state", n_nodes=4, precision=1.0)
    bottom = list(range(4))

    # Add one parent layer of size 3
    net, parents = net.add_value_parent_layer(
        children_idxs=bottom,
        n_parents=3,
        precision=1.0,
        tonic_volatility=-1.0,
        autoconnection_strength=0.2,
    )

    # Expect exactly 3 new nodes
    assert len(parents) == 3

    # For each parent, check fully-connected structure
    for p in parents:
        assert net.edges[p].value_children == bottom
        assert len(net.attributes[p]["value_coupling_children"]) == len(bottom)

def test_add_value_parent_stack():
    """Test building a multi-layer deep parent stack."""
    net = Network()

    # Base layer of 4 nodes
    net = net.add_nodes(kind="continuous-state", n_nodes=4, precision=1.0)
    bottom = list(range(4))

    # Build 3 → 2 → 1 parent stack
    net, layers = net.add_value_parent_stack(
        start_children_idxs=bottom,
        layer_sizes=[3, 2, 1],
        precision=1.0,
        tonic_volatility=-1.0,
        autoconnection_strength=0.3,
    )

    # Check layer sizes
    assert layers[0] and len(layers[0]) == 3
    assert layers[1] and len(layers[1]) == 2
    assert layers[2] and len(layers[2]) == 1

    # Check connections are fully dense
    # Layer 0 → bottom
    for p in layers[0]:
        assert net.edges[p].value_children == bottom

    # Layer 1 → layer 0
    for p in layers[1]:
        assert net.edges[p].value_children == layers[0]

    # Layer 2 → layer 1
    for p in layers[2]:
        assert net.edges[p].value_children == layers[1]