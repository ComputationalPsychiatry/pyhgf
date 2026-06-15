use rshgf::model::Network;

/// Helper to check approximate equality of f64 values.
fn assert_close(actual: f64, expected: f64, label: &str) {
    let tol = 1e-10;
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {} (diff = {:.2e})",
        label,
        expected,
        actual,
        (actual - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// Two-level binary HGF  (binary-state → continuous-state value parent)
// ---------------------------------------------------------------------------

#[test]
fn test_binary_2_levels_single_obs() {
    // Node 0: binary-state input
    // Node 1: continuous-state value parent of node 0
    let mut network = Network::new("eHGF");
    network.add_nodes("binary-state", 1, None, None, None, None, None, None);
    network.add_nodes(
        "continuous-state",
        1,
        None,
        Some(vec![0].into()),
        None,
        None,
        None,
        Some([("mean".into(), 1.0), ("tonic_volatility".into(), 1.0)].into()),
    );
    network.set_update_sequence();
    network.input_data(vec![vec![1.0]], None, true);

    // Node 0 — binary state
    let n0 = &network.node_trajectories.nodes[0];
    assert_close(n0.mean[0], 1.0, "n0 mean");
    assert_close(n0.expected_mean[0], 0.731058578630005, "n0 expected_mean");
    assert_close(n0.precision[0], 0.196611933241482, "n0 precision");
    assert_close(
        n0.expected_precision[0],
        0.196611933241482,
        "n0 expected_precision",
    );
    assert_close(n0.observed[0], 1.0, "n0 observed");
    assert_close(
        n0.value_prediction_error[0],
        1.367879441171442,
        "n0 value_pe",
    );

    // Node 1 — continuous value parent
    let n1 = &network.node_trajectories.nodes[1];
    assert_close(n1.mean[0], 1.577681201748482, "n1 mean");
    assert_close(n1.expected_mean[0], 1.0, "n1 expected_mean");
    assert_close(n1.precision[0], 0.465553354611477, "n1 precision");
    assert_close(
        n1.expected_precision[0],
        0.268941421369995,
        "n1 expected_precision",
    );
}

// ---------------------------------------------------------------------------
// Three-level binary HGF  (binary → continuous value → continuous volatility)
//
// Expected values reflect the piHGF prediction step: relative to the canonical
// (g)HGF, the volatility parent's variance enters the child's predicted
// log-volatility through the exact moment-generating-function correction
// `κ² / (2 π̂_parent)`, which shifts node-1's predicted precision (and every
// downstream quantity that depends on it).
// ---------------------------------------------------------------------------

#[test]
fn test_binary_3_levels_single_obs() {
    // Node 0: binary-state input
    // Node 1: continuous-state value parent of node 0
    // Node 2: continuous-state volatility parent of node 1
    let mut network = Network::new("eHGF");
    network.add_nodes("binary-state", 1, None, None, None, None, None, None);
    network.add_nodes(
        "continuous-state",
        1,
        None,
        Some(vec![0].into()),
        None,
        None,
        None,
        Some([("mean".into(), 1.0), ("tonic_volatility".into(), 1.0)].into()),
    );
    network.add_nodes(
        "continuous-state",
        1,
        None,
        None,
        None,
        Some(vec![1].into()),
        None,
        Some([("mean".into(), 1.0), ("tonic_volatility".into(), 1.0)].into()),
    );
    network.set_update_sequence();
    network.input_data(vec![vec![1.0]], None, true);

    // Node 0 — binary state (no volatility parent → unchanged from canonical)
    let n0 = &network.node_trajectories.nodes[0];
    assert_close(n0.mean[0], 1.0, "n0 mean");
    assert_close(n0.expected_mean[0], 0.731058578630005, "n0 expected_mean");
    assert_close(n0.precision[0], 0.196611933241482, "n0 precision");
    assert_close(
        n0.expected_precision[0],
        0.196611933241482,
        "n0 expected_precision",
    );

    // Node 1 — continuous value parent (has volatility parent node 2 → piHGF correction applies)
    let n1 = &network.node_trajectories.nodes[1];
    assert_close(n1.mean[0], 2.237863416338325, "n1 mean");
    assert_close(n1.expected_mean[0], 1.0, "n1 expected_mean");
    assert_close(n1.precision[0], 0.217262597650345, "n1 precision");
    assert_close(
        n1.expected_precision[0],
        0.020650664408863,
        "n1 expected_precision",
    );

    // Node 2 — continuous volatility parent (top of chain, no parent above → expected_precision unchanged)
    let n2 = &network.node_trajectories.nodes[2];
    assert_close(n2.mean[0], -0.590073315695407, "n2 mean");
    assert_close(n2.expected_mean[0], 1.0, "n2 expected_mean");
    assert_close(n2.precision[0], 0.537517033851118, "n2 precision");
    assert_close(
        n2.expected_precision[0],
        0.268941421369995,
        "n2 expected_precision",
    );
}

#[test]
fn test_binary_3_levels_two_obs() {
    // Feed two observations [1.0, 0.0] and verify both time steps.
    let mut network = Network::new("eHGF");
    network.add_nodes("binary-state", 1, None, None, None, None, None, None);
    network.add_nodes(
        "continuous-state",
        1,
        None,
        Some(vec![0].into()),
        None,
        None,
        None,
        Some([("mean".into(), 1.0), ("tonic_volatility".into(), 1.0)].into()),
    );
    network.add_nodes(
        "continuous-state",
        1,
        None,
        None,
        None,
        Some(vec![1].into()),
        None,
        Some([("mean".into(), 1.0), ("tonic_volatility".into(), 1.0)].into()),
    );
    network.set_update_sequence();
    network.input_data(vec![vec![1.0], vec![0.0]], None, true);

    // ---- Step 0 (observation = 1.0) ----

    // Node 0
    let n0 = &network.node_trajectories.nodes[0];
    assert_close(n0.mean[0], 1.0, "s0 n0 mean");
    assert_close(
        n0.expected_mean[0],
        0.731058578630005,
        "s0 n0 expected_mean",
    );
    assert_close(n0.precision[0], 0.196611933241482, "s0 n0 precision");
    assert_close(
        n0.expected_precision[0],
        0.196611933241482,
        "s0 n0 expected_precision",
    );

    // Node 1
    let n1 = &network.node_trajectories.nodes[1];
    assert_close(n1.mean[0], 2.237863416338325, "s0 n1 mean");
    assert_close(n1.expected_mean[0], 1.0, "s0 n1 expected_mean");
    assert_close(n1.precision[0], 0.217262597650345, "s0 n1 precision");
    assert_close(
        n1.expected_precision[0],
        0.020650664408863,
        "s0 n1 expected_precision",
    );

    // Node 2
    let n2 = &network.node_trajectories.nodes[2];
    assert_close(n2.mean[0], -0.590073315695407, "s0 n2 mean");
    assert_close(n2.expected_mean[0], 1.0, "s0 n2 expected_mean");
    assert_close(n2.precision[0], 0.537517033851118, "s0 n2 precision");
    assert_close(
        n2.expected_precision[0],
        0.268941421369995,
        "s0 n2 expected_precision",
    );

    // ---- Step 1 (observation = 0.0) ----

    // Node 0
    assert_close(n0.mean[1], 0.0, "s1 n0 mean");
    assert_close(
        n0.expected_mean[1],
        0.903598504654528,
        "s1 n0 expected_mean",
    );
    assert_close(n0.precision[1], 0.087108247040629, "s1 n0 precision");
    assert_close(
        n0.expected_precision[1],
        0.087108247040629,
        "s1 n0 expected_precision",
    );

    // Node 1
    assert_close(n1.mean[1], -4.287959232795414, "s1 n1 mean");
    assert_close(
        n1.expected_mean[1],
        2.237863416338325,
        "s1 n1 expected_mean",
    );
    assert_close(n1.precision[1], 0.138465072257898, "s1 n1 precision");
    assert_close(
        n1.expected_precision[1],
        0.051356825217268,
        "s1 n1 expected_precision",
    );

    // Node 2
    assert_close(n2.mean[1], 2.133602312876886, "s1 n2 mean");
    assert_close(
        n2.expected_mean[1],
        -0.590073315695407,
        "s1 n2 expected_mean",
    );
    assert_close(n2.precision[1], 0.789268236136739, "s1 n2 precision");
    assert_close(
        n2.expected_precision[1],
        0.218403176385631,
        "s1 n2 expected_precision",
    );
}
