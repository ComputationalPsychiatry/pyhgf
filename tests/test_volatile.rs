use rshgf::model::network::Network;
use std::collections::HashMap;

/// Helper to assert approximate equality of f64 values.
fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {} (diff = {})",
        label,
        expected,
        actual,
        (actual - expected).abs()
    );
}

/// Assert that the value-level trajectories of two nodes in two networks match.
fn assert_value_level_match(
    net_a: &Network,
    node_a: usize,
    net_b: &Network,
    node_b: usize,
    label: &str,
) {
    let traj_a = &net_a.node_trajectories.nodes[node_a];
    let traj_b = &net_b.node_trajectories.nodes[node_b];

    let fields: Vec<(&str, &Vec<f64>, &Vec<f64>)> = vec![
        ("mean", &traj_a.mean, &traj_b.mean),
        (
            "expected_mean",
            &traj_a.expected_mean,
            &traj_b.expected_mean,
        ),
        ("precision", &traj_a.precision, &traj_b.precision),
        (
            "expected_precision",
            &traj_a.expected_precision,
            &traj_b.expected_precision,
        ),
    ];

    for (key, a, b) in &fields {
        for (t, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert_close(
                *va,
                *vb,
                1e-6,
                &format!("{} value-level '{}' t={}", label, key, t),
            );
        }
    }
}

/// Assert that the volatility-level trajectories of a volatile node match the
/// corresponding explicit node trajectories.
fn assert_vol_level_match(
    volatile_net: &Network,
    vol_node: usize,
    explicit_net: &Network,
    exp_node: usize,
    label: &str,
) {
    let vol_traj = &volatile_net.node_trajectories.nodes[vol_node];
    let exp_traj = &explicit_net.node_trajectories.nodes[exp_node];

    let fields: Vec<(&str, &Vec<f64>, &str, &Vec<f64>)> = vec![
        ("mean_vol", &vol_traj.mean_vol, "mean", &exp_traj.mean),
        (
            "expected_mean_vol",
            &vol_traj.expected_mean_vol,
            "expected_mean",
            &exp_traj.expected_mean,
        ),
        (
            "precision_vol",
            &vol_traj.precision_vol,
            "precision",
            &exp_traj.precision,
        ),
        (
            "expected_precision_vol",
            &vol_traj.expected_precision_vol,
            "expected_precision",
            &exp_traj.expected_precision,
        ),
    ];

    for (vol_key, a, exp_key, b) in &fields {
        for (t, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert_close(
                *va,
                *vb,
                1e-6,
                &format!("{} vol-level '{}' vs '{}' t={}", label, vol_key, exp_key, t),
            );
        }
    }
}

/// Build a volatile network: input (node 0) + volatile-state value parent (node 1).
fn build_volatile_network(volatility_updates: &str, data: &[f64]) -> Network {
    let mut net = Network::new(volatility_updates);
    net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
    net.add_nodes(
        "volatile-state",
        1,
        None,
        Some(0.into()),
        None,
        None,
        None,
        Some(HashMap::from([("autoconnection_strength".into(), 1.0)])),
    );
    net.set_update_sequence();
    net.input_data(data.iter().map(|v| vec![*v]).collect(), None, true);
    net
}

/// Build an explicit network: input (node 0) + value parent (node 1) + volatility
/// parent of node 1 (node 2).
fn build_explicit_network(volatility_updates: &str, data: &[f64]) -> Network {
    let mut net = Network::new(volatility_updates);
    net.add_nodes("continuous-state", 1, None, None, None, None, None, None);
    net.add_nodes(
        "continuous-state",
        1,
        None,
        Some(0.into()),
        None,
        None,
        None,
        None,
    );
    net.add_nodes(
        "continuous-state",
        1,
        None,
        None,
        None,
        Some(1.into()),
        None,
        None,
    );
    net.set_update_sequence();
    net.input_data(data.iter().map(|v| vec![*v]).collect(), None, true);
    net
}

/// Run the volatile-vs-explicit comparison for the given update type.
fn compare_volatile_and_explicit(volatility_updates: &str) {
    let data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();

    let volatile_net = build_volatile_network(volatility_updates, &data);
    let explicit_net = build_explicit_network(volatility_updates, &data);

    let label = format!("{} volatile vs explicit", volatility_updates);

    // Input nodes should agree
    assert_value_level_match(
        &volatile_net,
        0,
        &explicit_net,
        0,
        &format!("{} input", label),
    );

    // Value level of volatile node 1 should match explicit node 1
    assert_value_level_match(&volatile_net, 1, &explicit_net, 1, &label);

    // Volatility level of volatile node 1 should match explicit node 2
    assert_vol_level_match(&volatile_net, 1, &explicit_net, 2, &label);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_volatile_standard_matches_explicit() {
    compare_volatile_and_explicit("standard");
}

#[test]
fn test_volatile_ehgf_matches_explicit() {
    compare_volatile_and_explicit("eHGF");
}

#[test]
fn test_volatile_unbounded_matches_explicit() {
    compare_volatile_and_explicit("unbounded");
}
