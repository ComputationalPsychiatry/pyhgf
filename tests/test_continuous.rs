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
    let mut network = Network::new();

    // Node 0: input node (no parents or children specified)
    network.add_nodes("continuous-state", None, None, None, None);
    // Node 1: value parent of node 0
    network.add_nodes("continuous-state", None, Some(vec![0].into()), None, None);

    network.set_update_sequence();
    network.input_data(vec![0.2], None);

    // Check node 0 trajectories
    let node0 = network.node_trajectories.floats.get(&0).expect("node 0 trajectories");
    assert_close(node0["precision"][0], 1.0, "node0 precision");
    assert_close(node0["expected_precision"][0], 1.0, "node0 expected_precision");
    assert_close(node0["mean"][0], 0.2, "node0 mean");
    assert_close(node0["expected_mean"][0], 0.0, "node0 expected_mean");

    // Check node 1 trajectories
    let node1 = network.node_trajectories.floats.get(&1).expect("node 1 trajectories");
    assert_close(node1["precision"][0], 1.9820137, "node1 precision");
    assert_close(node1["expected_precision"][0], 0.98201376, "node1 expected_precision");
    assert_close(node1["mean"][0], 0.10090748, "node1 mean");
    assert_close(node1["expected_mean"][0], 0.0, "node1 expected_mean");
}

#[test]
fn test_two_nodes_hgf() {
    // Three-node continuous HGF:
    // Node 0: input (leaf)
    // Node 1: value parent of node 0
    // Node 2: volatility parent of node 0
    let mut network = Network::new();

    // Node 0: input node
    network.add_nodes("continuous-state", None, None, None, None);
    // Node 1: value parent of node 0
    network.add_nodes("continuous-state", None, Some(vec![0].into()), None, None);
    // Node 2: volatility parent of node 0
    network.add_nodes("continuous-state", None, None, None, Some(vec![0].into()));

    network.set_update_sequence();
    network.input_data(vec![0.2], None);

    // Check node 0 trajectories
    let node0 = network.node_trajectories.floats.get(&0).expect("node 0 trajectories");
    assert_close(node0["precision"][0], 1.0, "node0 precision");
    assert_close(node0["expected_precision"][0], 0.5, "node0 expected_precision");
    assert_close(node0["mean"][0], 0.2, "node0 mean");
    assert_close(node0["expected_mean"][0], 0.0, "node0 expected_mean");

    // Check node 1 trajectories
    let node1 = network.node_trajectories.floats.get(&1).expect("node 1 trajectories");
    assert_close(node1["precision"][0], 1.4820137, "node1 precision");
    assert_close(node1["expected_precision"][0], 0.98201376, "node1 expected_precision");
    assert_close(node1["mean"][0], 0.06747576, "node1 mean");
    assert_close(node1["expected_mean"][0], 0.0, "node1 expected_mean");

    // Check node 2 trajectories
    let node2 = network.node_trajectories.floats.get(&2).expect("node 2 trajectories");
    assert_close(node2["precision"][0], 1.1070137, "node2 precision");
    assert_close(node2["expected_precision"][0], 0.98201376, "node2 expected_precision");
    assert_close(node2["mean"][0], -0.12219789, "node2 mean");
    assert_close(node2["expected_mean"][0], 0.0, "node2 expected_mean");
}
