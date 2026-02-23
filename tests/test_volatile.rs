use rshgf::model::Network;

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
    let floats_a = net_a
        .node_trajectories
        .floats
        .get(&node_a)
        .unwrap_or_else(|| panic!("{}: no trajectories for node {}", label, node_a));
    let floats_b = net_b
        .node_trajectories
        .floats
        .get(&node_b)
        .unwrap_or_else(|| panic!("{}: no trajectories for node {}", label, node_b));

    for key in ["mean", "expected_mean", "precision", "expected_precision"] {
        let a = &floats_a[key];
        let b = &floats_b[key];
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
    let vol_floats = volatile_net
        .node_trajectories
        .floats
        .get(&vol_node)
        .unwrap_or_else(|| panic!("{}: no trajectories for volatile node {}", label, vol_node));
    let exp_floats = explicit_net
        .node_trajectories
        .floats
        .get(&exp_node)
        .unwrap_or_else(|| panic!("{}: no trajectories for explicit node {}", label, exp_node));

    let key_map = [
        ("mean_vol", "mean"),
        ("expected_mean_vol", "expected_mean"),
        ("precision_vol", "precision"),
        ("expected_precision_vol", "expected_precision"),
    ];

    for (vol_key, exp_key) in key_map {
        let a = &vol_floats[vol_key];
        let b = &exp_floats[exp_key];
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
fn build_volatile_network(update_type: &str, data: &[f64]) -> Network {
    let mut net = Network::new(update_type);
    net.add_nodes("continuous-state", 1, None, None, None, None);
    net.add_nodes("volatile-state", 1, None, Some(0.into()), None, None);
    net.set_update_sequence();
    net.input_data(data.to_vec(), None);
    net
}

/// Build an explicit network: input (node 0) + value parent (node 1) + volatility
/// parent of node 1 (node 2).
fn build_explicit_network(update_type: &str, data: &[f64]) -> Network {
    let mut net = Network::new(update_type);
    net.add_nodes("continuous-state", 1, None, None, None, None);
    net.add_nodes("continuous-state", 1, None, Some(0.into()), None, None);
    net.add_nodes("continuous-state", 1, None, None, None, Some(1.into()));
    net.set_update_sequence();
    net.input_data(data.to_vec(), None);
    net
}

/// Run the volatile-vs-explicit comparison for the given update type.
fn compare_volatile_and_explicit(update_type: &str) {
    let data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();

    let volatile_net = build_volatile_network(update_type, &data);
    let explicit_net = build_explicit_network(update_type, &data);

    let label = format!("{} volatile vs explicit", update_type);

    // Input nodes should agree
    assert_value_level_match(&volatile_net, 0, &explicit_net, 0, &format!("{} input", label));

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
