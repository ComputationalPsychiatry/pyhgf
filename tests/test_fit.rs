use rshgf::model::Network;

/// Build a 3-layer network (2 targets → 2 hidden → 2 predictors), call `fit`
/// with a fixed learning rate, and verify that coupling weights change and
/// trajectories are recorded.
#[test]
fn test_fit_fixed_lr_3layer() {
    let mut net = Network::new("eHGF");

    // Bottom layer: 2 target (child) nodes
    net.add_nodes("continuous-state", 2, None, None, None, None, None);

    // Hidden layer: 2 nodes – each connected to both targets
    net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0);

    // Top layer: 2 predictor nodes – each connected to both hidden nodes
    net.add_layer(2, "continuous-state", None, 1.0);

    // Record initial coupling strengths on hidden nodes (towards targets)
    let initial_couplings_2: Vec<f64> = net.attributes.vectors
        .get(&2).unwrap()
        .get("value_coupling_children").unwrap()
        .clone();
    let initial_couplings_3: Vec<f64> = net.attributes.vectors
        .get(&3).unwrap()
        .get("value_coupling_children").unwrap()
        .clone();

    // Synthetic data: 5 time steps, 2 predictors (x) and 2 targets (y)
    let x: Vec<Vec<f64>> = vec![
        vec![1.0, 0.5],
        vec![0.8, 0.3],
        vec![1.2, 0.7],
        vec![0.9, 0.4],
        vec![1.1, 0.6],
    ];
    let y: Vec<Vec<f64>> = vec![
        vec![0.5, 0.3],
        vec![0.4, 0.2],
        vec![0.6, 0.4],
        vec![0.45, 0.25],
        vec![0.55, 0.35],
    ];

    let inputs_x_idxs = vec![4, 5]; // top layer (predictors)
    let inputs_y_idxs = vec![0, 1]; // bottom layer (targets)

    // Fit with fixed lr = 0.2
    net.fit(&x, &y, &inputs_x_idxs, &inputs_y_idxs, Some(0.2));

    // Coupling strengths on hidden nodes should have changed
    let final_couplings_2: Vec<f64> = net.attributes.vectors
        .get(&2).unwrap()
        .get("value_coupling_children").unwrap()
        .clone();
    let final_couplings_3: Vec<f64> = net.attributes.vectors
        .get(&3).unwrap()
        .get("value_coupling_children").unwrap()
        .clone();

    assert_ne!(
        initial_couplings_2, final_couplings_2,
        "Coupling weights for hidden node 2 should change after fit"
    );
    assert_ne!(
        initial_couplings_3, final_couplings_3,
        "Coupling weights for hidden node 3 should change after fit"
    );

    // Trajectories should be recorded for all 6 nodes with correct length
    for node_idx in 0..6 {
        let traj = net.node_trajectories.floats.get(&node_idx)
            .unwrap_or_else(|| panic!("No trajectory recorded for node {}", node_idx));
        let mean_traj = traj.get("mean")
            .unwrap_or_else(|| panic!("No 'mean' trajectory for node {}", node_idx));
        assert_eq!(
            mean_traj.len(), 5,
            "Mean trajectory for node {} should have 5 entries", node_idx
        );
    }
}

/// 2-layer network: verify weight learning still works via end-flush.
#[test]
fn test_fit_fixed_lr_2layer() {
    let mut net = Network::new("eHGF");
    net.add_nodes("continuous-state", 2, None, None, None, None, None);
    net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0);

    let initial_couplings_2: Vec<f64> = net.attributes.vectors
        .get(&2).unwrap()
        .get("value_coupling_children").unwrap()
        .clone();

    let x = vec![vec![1.0, 0.5], vec![0.8, 0.3], vec![1.2, 0.7]];
    let y = vec![vec![0.5, 0.3], vec![0.4, 0.2], vec![0.6, 0.4]];

    net.fit(&x, &y, &[2, 3], &[0, 1], Some(0.2));

    let final_couplings_2: Vec<f64> = net.attributes.vectors
        .get(&2).unwrap()
        .get("value_coupling_children").unwrap()
        .clone();

    assert_ne!(
        initial_couplings_2, final_couplings_2,
        "Coupling weights for parent node 2 should change after fit (2-layer)"
    );
}

/// Verify that dynamic learning also updates couplings (3-layer).
#[test]
fn test_fit_dynamic_lr_3layer() {
    let mut net = Network::new("eHGF");
    net.add_nodes("continuous-state", 2, None, None, None, None, None);
    net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0);
    net.add_layer(2, "continuous-state", None, 1.0);

    let initial_c2 = net.attributes.vectors.get(&2).unwrap()
        .get("value_coupling_children").unwrap().clone();

    let x = vec![vec![1.0, 0.5], vec![0.8, 0.3]];
    let y = vec![vec![0.5, 0.3], vec![0.4, 0.2]];

    // lr = None → dynamic
    net.fit(&x, &y, &[4, 5], &[0, 1], None);

    let final_c2 = net.attributes.vectors.get(&2).unwrap()
        .get("value_coupling_children").unwrap().clone();

    assert_ne!(initial_c2, final_c2, "Dynamic lr should update couplings");
}

/// Verify that the lr attribute is set on non-predictor nodes.
#[test]
fn test_fit_sets_lr_attribute() {
    let mut net = Network::new("eHGF");
    net.add_nodes("continuous-state", 2, None, None, None, None, None);
    net.add_layer(1, "continuous-state", Some(vec![0, 1]), 1.0);

    let x = vec![vec![1.0]];
    let y = vec![vec![0.5, 0.3]];

    net.fit(&x, &y, &[2], &[0, 1], Some(0.05));

    // lr should be set on non-predictor nodes (0, 1), not on predictor node (2)
    for &node_idx in &[0_usize, 1] {
        let lr = net.attributes.floats.get(&node_idx)
            .and_then(|f| f.get("lr"));
        assert_eq!(lr, Some(&0.05), "lr should be 0.05 on node {}", node_idx);
    }
}

/// Running fit with zero time steps should not panic.
#[test]
fn test_fit_empty_data() {
    let mut net = Network::new("eHGF");
    net.add_nodes("continuous-state", 1, None, None, None, None, None);
    net.add_layer(1, "continuous-state", Some(vec![0]), 1.0);

    let x: Vec<Vec<f64>> = vec![];
    let y: Vec<Vec<f64>> = vec![];

    net.fit(&x, &y, &[1], &[0], Some(0.1));

    // No crash, and trajectories should be empty
    assert!(
        net.node_trajectories.floats.get(&0)
            .and_then(|t| t.get("mean"))
            .map(|v| v.is_empty())
            .unwrap_or(true),
        "Trajectories should be empty with no data"
    );
}

/// Multiple calls to fit should overwrite trajectories.
#[test]
fn test_fit_overwrites_trajectories() {
    let mut net = Network::new("eHGF");
    net.add_nodes("continuous-state", 1, None, None, None, None, None);
    net.add_layer(1, "continuous-state", Some(vec![0]), 1.0);

    let x1 = vec![vec![1.0], vec![2.0], vec![3.0]];
    let y1 = vec![vec![0.5], vec![0.6], vec![0.7]];
    net.fit(&x1, &y1, &[1], &[0], Some(0.1));

    let len_after_first = net.node_trajectories.floats.get(&0)
        .unwrap().get("mean").unwrap().len();
    assert_eq!(len_after_first, 3);

    // Second fit with only 2 time steps
    let x2 = vec![vec![4.0], vec![5.0]];
    let y2 = vec![vec![0.8], vec![0.9]];
    net.fit(&x2, &y2, &[1], &[0], Some(0.1));

    let len_after_second = net.node_trajectories.floats.get(&0)
        .unwrap().get("mean").unwrap().len();
    assert_eq!(len_after_second, 2, "Second fit should overwrite trajectories");
}

/// Verify vector trajectories (coupling history) are recorded.
#[test]
fn test_fit_records_vector_trajectories() {
    let mut net = Network::new("eHGF");
    net.add_nodes("continuous-state", 2, None, None, None, None, None);
    net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0);
    net.add_layer(2, "continuous-state", None, 1.0);

    let x = vec![vec![1.0, 0.5], vec![0.8, 0.3], vec![1.2, 0.7]];
    let y = vec![vec![0.5, 0.3], vec![0.4, 0.2], vec![0.6, 0.4]];

    net.fit(&x, &y, &[4, 5], &[0, 1], Some(0.2));

    // Hidden node 2 should have vector trajectories for coupling children
    let vec_traj = net.node_trajectories.vectors.get(&2)
        .expect("No vector trajectories for node 2");
    let coupling_traj = vec_traj.get("value_coupling_children")
        .expect("No coupling-children trajectory for node 2");
    assert_eq!(
        coupling_traj.len(), 3,
        "Coupling trajectory for node 2 should have 3 entries"
    );
}
