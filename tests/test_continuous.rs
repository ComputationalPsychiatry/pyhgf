use rshgf::model::network::Network;

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
    network.input_data(vec![vec![0.2]], None, None, true);

    // Check node 0 trajectories
    let node0 = &network.node_trajectories.nodes[0];
    assert_close(node0.precision[0], 1.0, "node0 precision");
    assert_close(node0.expected_precision[0], 1.0, "node0 expected_precision");
    assert_close(node0.mean[0], 0.2, "node0 mean");
    assert_close(node0.expected_mean[0], 0.0, "node0 expected_mean");

    // Check node 1 trajectories (no volatility parent → rHGF MGF term inactive)
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
    // Expected values reflect the relaxed HGF. Node 0's conditional predicted
    // precision picks up the moment-generating-function term `κ² / (2 π̂_2)`
    // inside its log-volatility exponent: π̄_0 = 1 / (1 + e^{1 / (2 π̂_2)}) =
    // 0.37539, against the canonical 0.5. Its marginal predicted precision adds
    // node 1's variance 1 / π̂_1, giving π̂_0 = 0.27158. Node 1 conditions on the
    // observation through π̄_0: π_1 = π̂_1 + π̄_0 and μ_1 = π̄_0 · u / π_1.
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
    network.input_data(vec![vec![0.2]], None, None, true);

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
    assert_close(node1.precision[0], 1.35740476, "node1 precision");
    assert_close(
        node1.expected_precision[0],
        0.98201376,
        "node1 expected_precision",
    );
    assert_close(node1.mean[0], 0.05531010, "node1 mean");
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

// ---------------------------------------------------------------------------
// Unbounded update of a node with several children
//
// Node 4 is the shared volatility parent of nodes 2 and 3, or the value parent of
// node 2 and the volatility parent of node 3. Expected values were computed with the
// JAX backend under `jax_enable_x64`, for the same networks and inputs.
// ---------------------------------------------------------------------------

/// Check a trajectory against the JAX (float64) reference values.
fn assert_trajectory_matches(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length", label);
    for (step, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "{} at step {}: expected {}, got {} (diff = {:.2e})",
            label,
            step,
            e,
            a,
            (a - e).abs()
        );
    }
}

/// Two inputs (nodes 0 and 1) with their value parents (nodes 2 and 3), and node 4
/// with the given volatility and value children, fed with three observations.
fn unbounded_network(
    volatility_children: Vec<usize>,
    value_children: Option<Vec<usize>>,
    mean_field_updates: bool,
) -> Network {
    let mut network = Network::new("unbounded");
    network.mean_field_updates = mean_field_updates;
    network.add_nodes("continuous-state", 2, None, None, None, None, None, None);
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
    network.add_nodes(
        "continuous-state",
        1,
        None,
        Some(vec![1].into()),
        None,
        None,
        None,
        None,
    );
    network.add_nodes(
        "continuous-state",
        1,
        None,
        value_children.map(Into::into),
        None,
        Some(volatility_children.into()),
        None,
        None,
    );
    network.set_update_sequence();
    network.input_data(
        vec![vec![0.5, -0.2], vec![2.5, 0.1], vec![-1.0, 3.0]],
        None,
        None,
        true,
    );
    network
}

#[test]
fn test_unbounded_shared_volatility_parent() {
    // Both volatility children enter the posterior of node 4.
    let network = unbounded_network(vec![2, 3], None, false);
    let node4 = &network.node_trajectories.nodes[4];
    assert_trajectory_matches(
        &node4.mean,
        &[
            0.026489553894727819,
            0.085598485010175571,
            0.24508979947868789,
        ],
        "node4 mean",
    );
    assert_trajectory_matches(
        &node4.precision,
        &[1.0226162894216737, 1.0378626556476724, 1.0414612182845073],
        "node4 precision",
    );
}

#[test]
fn test_unbounded_volatility_children_order() {
    // The posterior is a symmetric function of the volatility children.
    let a = unbounded_network(vec![2, 3], None, false);
    let b = unbounded_network(vec![3, 2], None, false);
    let (a4, b4) = (&a.node_trajectories.nodes[4], &b.node_trajectories.nodes[4]);
    for step in 0..a4.mean.len() {
        assert!(
            (a4.mean[step] - b4.mean[step]).abs() < 1e-12,
            "mean, step {}",
            step
        );
        assert!(
            (a4.precision[step] - b4.precision[step]).abs() < 1e-12,
            "precision, step {}",
            step
        );
    }
}

#[test]
fn test_unbounded_value_and_volatility_parent() {
    // Node 4 uses its value child (relaxed value-coupling update) in the effective
    // prior of the unbounded update.
    let network = unbounded_network(vec![3], Some(vec![2]), false);
    let node4 = &network.node_trajectories.nodes[4];
    assert_trajectory_matches(
        &node4.mean,
        &[
            0.16932032739006528,
            0.74129590542993129,
            -0.040523526657161725,
        ],
        "node4 mean",
    );
    assert_trajectory_matches(
        &node4.precision,
        &[1.4932261453879994, 2.0706138669446332, 2.6135937369334141],
        "node4 precision",
    );
}

#[test]
fn test_unbounded_value_and_volatility_parent_mean_field() {
    // Same network with the mean-field value-coupling update.
    let network = unbounded_network(vec![3], Some(vec![2]), true);
    let node4 = &network.node_trajectories.nodes[4];
    assert_trajectory_matches(
        &node4.mean,
        &[0.12646916368770927, 0.4884466188901479, 0.1974466902491045],
        "node4 mean",
    );
    assert_trajectory_matches(
        &node4.precision,
        &[1.9781454115279276, 3.8414366381660594, 6.3420205208283953],
        "node4 precision",
    );
}
