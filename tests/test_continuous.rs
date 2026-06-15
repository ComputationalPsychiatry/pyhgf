use rshgf::model::Network;

/// Helper to check approximate equality of f64 values
fn assert_close(actual: f64, expected: f64, label: &str) {
    let tol = 1e-5;
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {} (diff = {})",
        label,
        expected,
        actual,
        (actual - expected).abs()
    );
}

#[test]
fn test_one_node_hgf() {
    // Two-node continuous HGF: one input node and one value parent
    // Node 0: input (leaf)
    // Node 1: value parent of node 0
    let mut network = Network::new("eHGF");

    // Node 0: input node (no parents or children specified)
    network.add_nodes("continuous-state", 1, None, None, None, None, None, None);
    // Node 1: value parent of node 0
    network.add_nodes(
        "continuous-state",
        1,
        None,
        Some(vec![0].into()),
        None,
        None,
        None,
        None,
    );

    network.set_update_sequence();
    network.input_data(vec![vec![0.2]], None, true);

    // Check node 0 trajectories
    let node0 = &network.node_trajectories.nodes[0];
    assert_close(node0.precision[0], 1.0, "node0 precision");
    assert_close(node0.expected_precision[0], 1.0, "node0 expected_precision");
    assert_close(node0.mean[0], 0.2, "node0 mean");
    assert_close(node0.expected_mean[0], 0.0, "node0 expected_mean");

    // Check node 1 trajectories (no volatility parent → piHGF correction inactive)
    let node1 = &network.node_trajectories.nodes[1];
    assert_close(node1.precision[0], 1.9820137, "node1 precision");
    assert_close(
        node1.expected_precision[0],
        0.98201376,
        "node1 expected_precision",
    );
    assert_close(node1.mean[0], 0.10090748, "node1 mean");
    assert_close(node1.expected_mean[0], 0.0, "node1 expected_mean");
}

#[test]
fn test_two_nodes_hgf() {
    // Three-node continuous HGF:
    // Node 0: input (root)
    // Node 1: value parent of node 0
    // Node 2: volatility parent of node 0
    //
    // Expected values reflect the relaxed prediction step: node 0's predicted
    // precision picks up the moment-generating-function correction
    // `κ² / (2 π̂_2)` inside its log-volatility exponent, shifting it from the
    // canonical 0.5 to 0.27158 and propagating through node 0's posterior to
    // node 1's and node 2's value-coupling updates.
    let mut network = Network::new("eHGF");

    // Node 0: input node
    network.add_nodes("continuous-state", 1, None, None, None, None, None, None);
    // Node 1: value parent of node 0
    network.add_nodes(
        "continuous-state",
        1,
        None,
        Some(vec![0].into()),
        None,
        None,
        None,
        None,
    );
    // Node 2: volatility parent of node 0
    network.add_nodes(
        "continuous-state",
        1,
        None,
        None,
        None,
        Some(vec![0].into()),
        None,
        None,
    );

    network.set_update_sequence();
    network.input_data(vec![vec![0.2]], None, true);

    // Check node 0 trajectories
    let node0 = &network.node_trajectories.nodes[0];
    assert_close(node0.precision[0], 1.0, "node0 precision");
    assert_close(
        node0.expected_precision[0],
        0.27157641,
        "node0 expected_precision",
    );
    assert_close(node0.mean[0], 0.2, "node0 mean");
    assert_close(node0.expected_mean[0], 0.0, "node0 expected_mean");

    // Check node 1 trajectories
    let node1 = &network.node_trajectories.nodes[1];
    assert_close(node1.precision[0], 1.25359020, "node1 precision");
    assert_close(
        node1.expected_precision[0],
        0.98201376,
        "node1 expected_precision",
    );
    assert_close(node1.mean[0], 0.04332778, "node1 mean");
    assert_close(node1.expected_mean[0], 0.0, "node1 expected_mean");

    // Check node 2 trajectories
    let node2 = &network.node_trajectories.nodes[2];
    assert_close(node2.precision[0], 1.09553182, "node2 precision");
    assert_close(
        node2.expected_precision[0],
        0.98201376,
        "node2 expected_precision",
    );
    assert_close(node2.mean[0], -0.16509254, "node2 mean");
    assert_close(node2.expected_mean[0], 0.0, "node2 expected_mean");
}
