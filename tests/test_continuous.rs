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
    // Node 2: volatility parent of node 0, which is its noise parent
    //
    // Expected values follow Eqs. 64 to 78 of Weber et al. (2026). The precision of the
    // observation noise is π̂_0 = exp(−μ̂_2) = 1 (Eq. 66). Node 1 is updated first:
    // π_1 = π̂_1 + π̂_0 and μ_1 = π̂_0 u / π_1 (Eqs. 69 and 70). The noise prediction
    // error reads this posterior, ε = π̂_0 / π_1 + π̂_0 (u − μ_1)² − 1 (Eq. 72), and node
    // 2 takes π_2 = π̂_2 + ½ (1 + ε) and μ_2 = μ̂_2 + ε / (2 π_2) (Eqs. 74 and 78)
    // under the eHGF as under the other schemes.
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
    // Node 2: noise parent of node 0
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
    assert_close(node0.expected_precision[0], 1.0, "node0 expected_precision");
    assert_close(node0.mean[0], 0.2, "node0 mean");
    assert_close(node0.expected_mean[0], 0.0, "node0 expected_mean");
    assert_close(
        node0.volatility_prediction_error[0],
        -0.48564331,
        "node0 noise prediction error",
    );

    // Check node 1 trajectories
    let node1 = &network.node_trajectories.nodes[1];
    assert_close(node1.precision[0], 1.98201379, "node1 precision");
    assert_close(
        node1.expected_precision[0],
        0.98201379,
        "node1 expected_precision",
    );
    assert_close(node1.mean[0], 0.10090747, "node1 mean");
    assert_close(node1.expected_mean[0], 0.0, "node1 expected_mean");

    // Check node 2 trajectories
    let node2 = &network.node_trajectories.nodes[2];
    assert_close(node2.precision[0], 1.23919213, "node2 precision");
    assert_close(
        node2.expected_precision[0],
        0.98201379,
        "node2 expected_precision",
    );
    assert_close(node2.mean[0], -0.19595158, "node2 mean");
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

// ---------------------------------------------------------------------------
// Noise parent of an input node (Weber et al., 2026, Eqs. 64 to 78)
//
// An input with a value parent x and a noise parent a, each with or without a
// volatility parent, under the mean-field updates, with couplings κ_u = 1.4 (input and
// noise parent), κ_x = 0.7 and κ_a = 0.8 (volatility parents of x and a). Expected values
// come from the transcription of the equations of the paper in
// `tests/test_noise_parent.py` (`_weber_filter`), for the same parameters and inputs.
// ---------------------------------------------------------------------------

/// The tonic observation noise ε_u = 0.2 is a shift of ε_u / κ_u of the mean of the
/// noise parent, whose updates depend on κ_u μ_a + ε_u only.
const NOISE_SHIFT: f64 = 0.2 / 1.4;

/// Build and filter the network. Node 0 is the input and node 1 its value parent x.
/// With `hierarchy`, node 2 is the volatility parent of x, node 3 the noise parent a of
/// the input and node 4 the volatility parent of a; otherwise node 2 is the noise parent.
/// The priors are μ_x = 0.5, μ_a = 0 (before the shift), μ = 1 for the volatility
/// parents, unit precisions, and tonic volatilities ω_x = −2, ω_a = −4 and −6 for the
/// volatility parents.
fn noise_parent_network(volatility_updates: &str, mean_field: bool, hierarchy: bool) -> Network {
    let mut network = Network::new(volatility_updates);
    network.mean_field_updates = mean_field;
    network.add_nodes("continuous-state", 1, None, None, None, None, None, None);
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
    let volatility_children: Vec<usize> = if hierarchy { vec![1, 0, 3] } else { vec![0] };
    for child_idx in volatility_children {
        network.add_nodes(
            "continuous-state",
            1,
            None,
            None,
            None,
            Some(vec![child_idx].into()),
            None,
            None,
        );
    }

    let (kappa_u, kappa_x, kappa_a) = (1.4, 0.7, 0.8);
    let noise_parent = if hierarchy { 3 } else { 2 };
    let vectors = &mut network.attributes.vectors;
    vectors[0].volatility_coupling_parents = vec![kappa_u];
    vectors[noise_parent].volatility_coupling_children = vec![kappa_u];
    if hierarchy {
        vectors[1].volatility_coupling_parents = vec![kappa_x];
        vectors[2].volatility_coupling_children = vec![kappa_x];
        vectors[3].volatility_coupling_parents = vec![kappa_a];
        vectors[4].volatility_coupling_children = vec![kappa_a];
    }

    let priors: Vec<(usize, f64, f64)> = if hierarchy {
        vec![
            (1, 0.5, -2.0),
            (2, 1.0, -6.0),
            (3, NOISE_SHIFT, -4.0),
            (4, 1.0, -6.0),
        ]
    } else {
        vec![(1, 0.5, -2.0), (2, NOISE_SHIFT, -4.0)]
    };
    let states = &mut network.attributes.states;
    for (node_idx, mean, tonic_volatility) in priors {
        states[node_idx].mean = mean;
        states[node_idx].precision = 1.0;
        states[node_idx].tonic_volatility = tonic_volatility;
    }

    network.set_update_sequence();
    network.input_data(
        vec![
            vec![0.8],
            vec![-0.4],
            vec![2.6],
            vec![1.1],
            vec![-1.9],
            vec![0.3],
        ],
        None,
        None,
        true,
    );
    network
}

/// Compare the trajectories of the network with the reference, with the means of the
/// noise parent shifted by [`NOISE_SHIFT`].
fn assert_noise_network_matches(
    network: &Network,
    noise_parent: usize,
    expected: &[(usize, &str, [f64; 6])],
) {
    for (node_idx, field, values) in expected {
        let node = &network.node_trajectories.nodes[*node_idx];
        let actual = match *field {
            "mean" => &node.mean,
            "precision" => &node.precision,
            "volatility_prediction_error" => &node.volatility_prediction_error,
            _ => unreachable!(),
        };
        let shift = if *node_idx == noise_parent && *field == "mean" {
            NOISE_SHIFT
        } else {
            0.0
        };
        let shifted: Vec<f64> = values.iter().map(|v| v + shift).collect();
        assert_trajectory_matches(actual, &shifted, &format!("node{node_idx} {field}"));
    }
}

#[test]
fn test_noise_parent_hierarchy_follows_the_equations() {
    // Standard and mean-field updates: every node follows Eqs. 34 to 45 and 64 to 78.
    let network = noise_parent_network("standard", true, true);
    assert_noise_network_matches(
        &network,
        3,
        &[
            (
                0,
                "volatility_prediction_error",
                [
                    -0.47207546746669116,
                    -0.1893131338922841,
                    1.6100867670521684,
                    -0.6214012845869202,
                    2.7092442875947764,
                    -0.6134451838283241,
                ],
            ),
            (
                1,
                "mean",
                [
                    0.6530752030871877,
                    0.12778109815300276,
                    1.272670647449769,
                    1.2090700162996646,
                    -0.08154553693242073,
                    0.0493823248625119,
                ],
            ),
            (
                1,
                "precision",
                [
                    1.6045657361205405,
                    2.244499838266063,
                    2.63144862236397,
                    2.34236097508705,
                    2.399770136067108,
                    2.013795817169324,
                ],
            ),
            (
                2,
                "mean",
                [
                    0.963980110105499,
                    0.9451755400792926,
                    1.1172585785295683,
                    1.066594479234706,
                    1.3129956133580223,
                    1.26901639775609,
                ],
            ),
            (
                2,
                "precision",
                [
                    1.023517735292245,
                    1.0483710181793005,
                    1.0466489065579554,
                    1.0957238071576498,
                    1.094641347686738,
                    1.1446355216394517,
                ],
            ),
            (
                3,
                "mean",
                [
                    -0.22355077519604136,
                    -0.2840904158490775,
                    -0.03742626457633219,
                    -0.140120415996557,
                    0.12111592917662406,
                    0.04984501945132061,
                ],
            ),
            (
                3,
                "precision",
                [
                    1.4782003190858783,
                    2.1889656478817052,
                    4.569211744475535,
                    4.235693008756661,
                    7.259598583264202,
                    6.025061702381667,
                ],
            ),
            (
                4,
                "mean",
                [
                    0.9952762166125465,
                    0.9872175887853234,
                    0.9732269574420649,
                    0.9703991259099571,
                    0.9562566473297803,
                    0.953357538402315,
                ],
            ),
            (
                4,
                "precision",
                [
                    1.0015064993099914,
                    1.005802511196272,
                    1.0149243994921517,
                    1.0215841570681488,
                    1.0339883869832758,
                    1.0485007072651213,
                ],
            ),
        ],
    );
}

#[test]
fn test_noise_parent_update_does_not_depend_on_the_scheme() {
    // Without volatility parents above x and a, the relaxed and the mean-field
    // predictions coincide, and the noise parent takes Eqs. 74 and 78 in every scheme.
    for volatility_updates in ["standard", "eHGF", "unbounded"] {
        for mean_field in [false, true] {
            let network = noise_parent_network(volatility_updates, mean_field, false);
            assert_noise_network_matches(
                &network,
                2,
                &[
                    (
                        0,
                        "volatility_prediction_error",
                        [
                            -0.49846836981885867,
                            -0.17437771846822603,
                            2.045696626388524,
                            -0.7238481024250025,
                            3.500167581189679,
                            -0.7645021510983101,
                        ],
                    ),
                    (
                        1,
                        "mean",
                        [
                            0.6445220380832475,
                            0.1721959197264899,
                            1.1326495622233663,
                            1.1236485393697282,
                            0.18804069973937332,
                            0.2138622759865886,
                        ],
                    ),
                    (
                        1,
                        "precision",
                        [
                            1.6995278310558641,
                            2.5222833523633748,
                            3.1112168184533084,
                            3.0226766759044947,
                            3.106347191670671,
                            2.8425401668890675,
                        ],
                    ),
                    (
                        2,
                        "mean",
                        [
                            -0.2367996994708599,
                            -0.2911979783061238,
                            -0.01260680655820412,
                            -0.11458886184290515,
                            0.15873350447387483,
                            0.091255621782415,
                        ],
                    ),
                    (
                        2,
                        "precision",
                        [
                            1.4735147876154269,
                            2.243901931114583,
                            5.140104151497255,
                            4.9684591106050595,
                            8.964203478295278,
                            7.930769082008366,
                        ],
                    ),
                ],
            );
        }
    }
}
