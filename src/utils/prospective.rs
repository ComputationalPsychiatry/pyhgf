use crate::model::Network;

// =============================================================================
// Prospective helpers (value-level only, no volatility children)
// =============================================================================

/// Compute a prospective posterior precision for `node_idx`, considering only
/// its *value* children (mirrors Python `posterior_update_precision_value_level`).
///
/// Formula per child:
///   π_post += π̂_child · (κ² · g'(μ)² − g''(μ) · δ_child)
///
/// For the linear (identity) coupling function this simplifies to
///   π_post += π̂_child · κ²
pub fn prospective_precision(network: &Network, node_idx: usize) -> f64 {
    let expected_precision = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("expected_precision"))
        .unwrap_or(&1.0);

    let mut precision = expected_precision;

    let vc_idxs = match network.edges.get(&node_idx)
        .and_then(|e| e.value_children.clone())
    {
        Some(v) => v,
        None => return precision,
    };

    let coupling_strengths = network.attributes.vectors
        .get(&node_idx)
        .and_then(|v| v.get("value_coupling_children").cloned());

    let parent_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("mean"))
        .unwrap_or(&0.0);

    for (i, &child_idx) in vc_idxs.iter().enumerate() {
        let child_floats = match network.attributes.floats.get(&child_idx) {
            Some(f) => f,
            None => continue,
        };
        let child_expected_precision = *child_floats.get("expected_precision")
            .unwrap_or(&1.0);
        let kappa = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

        // Find the coupling function stored on the child for this parent.
        let parent_pos = network.edges.get(&child_idx)
            .and_then(|e| e.value_parents.as_ref())
            .and_then(|vp| vp.iter().position(|&p| p == node_idx));

        let coupling_fn = parent_pos.and_then(|pos| {
            network.attributes.fn_ptrs
                .get(&child_idx)
                .and_then(|fp| fp.get("value_coupling_fn_parents"))
                .and_then(|fns| fns.get(pos).copied())
        });

        let (g_prime_sq, g_second_term) = match coupling_fn {
            Some(cf) => {
                let g_prime = (cf.df)(parent_mean);
                let g_second = (cf.d2f)(parent_mean);
                let child_vape = *child_floats.get("value_prediction_error")
                    .unwrap_or(&0.0);
                (g_prime.powi(2), g_second * child_vape)
            }
            None => (1.0, 0.0),
        };

        precision += child_expected_precision
            * (kappa.powi(2) * g_prime_sq - g_second_term);
    }

    precision
}

/// Compute a prospective posterior mean for `node_idx`, considering only its
/// *value* children (mirrors Python `posterior_update_mean_value_level`).
///
/// Formula per child:
///   mean += (κ · g'(μ) · π̂_child / π_node) · δ_child
pub fn prospective_mean(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let expected_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("expected_mean"))
        .unwrap_or(&0.0);

    let mut posterior_mean = expected_mean;

    let vc_idxs = match network.edges.get(&node_idx)
        .and_then(|e| e.value_children.clone())
    {
        Some(v) => v,
        None => return posterior_mean,
    };

    let coupling_strengths = network.attributes.vectors
        .get(&node_idx)
        .and_then(|v| v.get("value_coupling_children").cloned());

    let parent_mean = *network.attributes.floats
        .get(&node_idx)
        .and_then(|f| f.get("mean"))
        .unwrap_or(&0.0);

    for (i, &child_idx) in vc_idxs.iter().enumerate() {
        let child_floats = match network.attributes.floats.get(&child_idx) {
            Some(f) => f,
            None => continue,
        };
        let child_expected_precision = *child_floats.get("expected_precision")
            .unwrap_or(&1.0);
        let child_vape = *child_floats.get("value_prediction_error")
            .unwrap_or(&0.0);
        let kappa = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

        // g'(μ_parent) from the coupling function stored on the child.
        let parent_pos = network.edges.get(&child_idx)
            .and_then(|e| e.value_parents.as_ref())
            .and_then(|vp| vp.iter().position(|&p| p == node_idx));

        let coupling_fn_prime = parent_pos
            .and_then(|pos| {
                network.attributes.fn_ptrs
                    .get(&child_idx)
                    .and_then(|fp| fp.get("value_coupling_fn_parents"))
                    .and_then(|fns| fns.get(pos).copied())
            })
            .map(|cf| (cf.df)(parent_mean))
            .unwrap_or(1.0);

        posterior_mean += (kappa * coupling_fn_prime * child_expected_precision
            / node_precision)
            * child_vape;
    }

    posterior_mean
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AdjacencyLists, Attributes, Network, NodeTrajectories, UpdateSequence};
    use crate::math;
    use std::collections::HashMap;

    const TOL: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64, label: &str) {
        assert!(
            (actual - expected).abs() < TOL,
            "{}: expected {:.12}, got {:.12} (diff = {:.2e})",
            label, expected, actual, (actual - expected).abs()
        );
    }

    /// Build a minimal 2-node network:
    ///   node 0 (child)  — value_parents: [1]
    ///   node 1 (parent) — value_children: [0]
    ///
    /// Default attributes:
    ///   parent: mean=2.0, expected_mean=1.5, expected_precision=4.0, κ=1.0
    ///   child:  expected_precision=2.0, value_prediction_error=0.5
    fn make_two_node_network() -> Network {
        let mut net = Network {
            attributes: Attributes {
                floats: HashMap::new(),
                vectors: HashMap::new(),
                fn_ptrs: HashMap::new(),
            },
            edges: HashMap::new(),
            inputs: vec![0],
            update_type: "standard".into(),
            update_sequence: UpdateSequence {
                predictions: Vec::new(),
                updates: Vec::new(),
            },
            node_trajectories: NodeTrajectories {
                floats: HashMap::new(),
                vectors: HashMap::new(),
            },
            layers: Vec::new(),
        };

        // Node 0 (child): value_parents = [1]
        net.edges.insert(0, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: Some(vec![1]),
            value_children: None,
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.floats.insert(0, HashMap::from([
            ("expected_precision".into(), 2.0),
            ("value_prediction_error".into(), 0.5),
        ]));

        // Node 1 (parent): value_children = [0]
        net.edges.insert(1, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: None,
            value_children: Some(vec![0]),
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.floats.insert(1, HashMap::from([
            ("mean".into(), 2.0),
            ("expected_mean".into(), 1.5),
            ("expected_precision".into(), 4.0),
        ]));
        net.attributes.vectors.insert(1, HashMap::from([
            ("value_coupling_children".into(), vec![1.0]),
        ]));

        net
    }

    /// Build a 3-node network with two children:
    ///   node 0 (child-A) — value_parents: [2]
    ///   node 1 (child-B) — value_parents: [2]
    ///   node 2 (parent)  — value_children: [0, 1]
    fn make_three_node_network() -> Network {
        let mut net = Network {
            attributes: Attributes {
                floats: HashMap::new(),
                vectors: HashMap::new(),
                fn_ptrs: HashMap::new(),
            },
            edges: HashMap::new(),
            inputs: vec![0, 1],
            update_type: "standard".into(),
            update_sequence: UpdateSequence {
                predictions: Vec::new(),
                updates: Vec::new(),
            },
            node_trajectories: NodeTrajectories {
                floats: HashMap::new(),
                vectors: HashMap::new(),
            },
            layers: Vec::new(),
        };

        // Node 0 (child-A): value_parents = [2]
        net.edges.insert(0, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: Some(vec![2]),
            value_children: None,
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.floats.insert(0, HashMap::from([
            ("expected_precision".into(), 2.0),
            ("value_prediction_error".into(), 0.3),
        ]));

        // Node 1 (child-B): value_parents = [2]
        net.edges.insert(1, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: Some(vec![2]),
            value_children: None,
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.floats.insert(1, HashMap::from([
            ("expected_precision".into(), 3.0),
            ("value_prediction_error".into(), -0.2),
        ]));

        // Node 2 (parent): value_children = [0, 1]
        net.edges.insert(2, AdjacencyLists {
            node_type: "continuous-state".into(),
            value_parents: None,
            value_children: Some(vec![0, 1]),
            volatility_parents: None,
            volatility_children: None,
        });
        net.attributes.floats.insert(2, HashMap::from([
            ("mean".into(), 1.0),
            ("expected_mean".into(), 0.8),
            ("expected_precision".into(), 5.0),
        ]));
        net.attributes.vectors.insert(2, HashMap::from([
            ("value_coupling_children".into(), vec![1.0, 0.5]),
        ]));

        net
    }

    // ── prospective_precision ────────────────────────────────────────────

    #[test]
    fn test_prospective_precision_no_children() {
        // A node with no value children should return expected_precision unchanged.
        let net = make_two_node_network();
        let result = prospective_precision(&net, 0); // node 0 has no children
        assert_close(result, 2.0, "precision for childless node");
    }

    #[test]
    fn test_prospective_precision_linear_single_child() {
        // Node 1 has one linear child (node 0).
        // No coupling function stored → linear: g'=1, g''=0
        // π = expected_precision + child_expected_precision * κ²
        //   = 4.0 + 2.0 * 1.0² = 6.0
        let net = make_two_node_network();
        let result = prospective_precision(&net, 1);
        assert_close(result, 6.0, "linear precision single child");
    }

    #[test]
    fn test_prospective_precision_linear_two_children() {
        // Node 2 has children [0, 1] with κ = [1.0, 0.5]
        // π = 5.0 + 2.0*1.0² + 3.0*0.5² = 5.0 + 2.0 + 0.75 = 7.75
        let net = make_three_node_network();
        let result = prospective_precision(&net, 2);
        assert_close(result, 7.75, "linear precision two children");
    }

    #[test]
    fn test_prospective_precision_with_coupling_fn() {
        // Use sigmoid coupling on a single child.
        // g'(μ)² and g''(μ)·δ should contribute.
        let mut net = make_two_node_network();

        // Store sigmoid coupling fn on child (node 0) for its parent (node 1)
        let sigmoid_fn = math::resolve_coupling_fn("sigmoid");
        net.attributes.fn_ptrs.insert(0, HashMap::from([
            ("value_coupling_fn_parents".into(), vec![sigmoid_fn]),
        ]));

        let parent_mean = 2.0;
        let g_prime = math::sigmoid_d1(parent_mean);
        let g_second = math::sigmoid_d2(parent_mean);
        let child_vape = 0.5;
        let child_exp_prec = 2.0;

        // π = 4.0 + child_exp_prec * (κ²·g'² − g''·δ)
        let expected = 4.0 + child_exp_prec
            * (1.0_f64.powi(2) * g_prime.powi(2) - g_second * child_vape);

        let result = prospective_precision(&net, 1);
        assert_close(result, expected, "sigmoid precision");
    }

    #[test]
    fn test_prospective_precision_custom_kappa() {
        // Same as linear single child but with κ = 2.0
        let mut net = make_two_node_network();
        net.attributes.vectors.insert(1, HashMap::from([
            ("value_coupling_children".into(), vec![2.0]),
        ]));

        // π = 4.0 + 2.0 * 2.0² = 4.0 + 8.0 = 12.0
        let result = prospective_precision(&net, 1);
        assert_close(result, 12.0, "linear precision κ=2");
    }

    // ── prospective_mean ─────────────────────────────────────────────────

    #[test]
    fn test_prospective_mean_no_children() {
        let net = make_two_node_network();
        // Node 0 has no value children → returns expected_mean (not stored → 0.0)
        let floats = net.attributes.floats.get(&0).unwrap();
        let has_expected_mean = floats.contains_key("expected_mean");
        let default_mean = if has_expected_mean {
            *floats.get("expected_mean").unwrap()
        } else {
            0.0
        };
        let result = prospective_mean(&net, 0, 1.0);
        assert_close(result, default_mean, "mean for childless node");
    }

    #[test]
    fn test_prospective_mean_linear_single_child() {
        // Node 1 has one linear child.
        // mean = expected_mean + (κ * g'(μ) * π̂_child / π_node) * δ_child
        //      = 1.5 + (1.0 * 1.0 * 2.0 / 6.0) * 0.5
        //      = 1.5 + 0.1667
        let net = make_two_node_network();
        let node_precision = 6.0; // as computed by prospective_precision
        let expected = 1.5 + (1.0 * 1.0 * 2.0 / node_precision) * 0.5;
        let result = prospective_mean(&net, 1, node_precision);
        assert_close(result, expected, "linear mean single child");
    }

    #[test]
    fn test_prospective_mean_linear_two_children() {
        // Node 2 with children [0, 1], κ = [1.0, 0.5]
        // node_precision = 7.75
        // mean = 0.8 + (1.0*1.0*2.0/7.75)*0.3 + (0.5*1.0*3.0/7.75)*(-0.2)
        let net = make_three_node_network();
        let node_precision = 7.75;
        let expected = 0.8
            + (1.0 * 1.0 * 2.0 / node_precision) * 0.3
            + (0.5 * 1.0 * 3.0 / node_precision) * (-0.2);
        let result = prospective_mean(&net, 2, node_precision);
        assert_close(result, expected, "linear mean two children");
    }

    #[test]
    fn test_prospective_mean_with_coupling_fn() {
        let mut net = make_two_node_network();

        let sigmoid_fn = math::resolve_coupling_fn("sigmoid");
        net.attributes.fn_ptrs.insert(0, HashMap::from([
            ("value_coupling_fn_parents".into(), vec![sigmoid_fn]),
        ]));

        let parent_mean = 2.0;
        let g_prime = math::sigmoid_d1(parent_mean);
        let node_precision = 6.0;

        // mean = 1.5 + (κ · g'(μ) · π̂_child / π_node) · δ_child
        let expected = 1.5 + (1.0 * g_prime * 2.0 / node_precision) * 0.5;
        let result = prospective_mean(&net, 1, node_precision);
        assert_close(result, expected, "sigmoid mean");
    }

    #[test]
    fn test_prospective_precision_then_mean_consistency() {
        // Verify that computing precision first and feeding it to mean
        // produces a self-consistent pair.
        let net = make_three_node_network();
        let precision = prospective_precision(&net, 2);
        let mean = prospective_mean(&net, 2, precision);

        // Expected precision: 5.0 + 2.0*1² + 3.0*0.5² = 7.75
        assert_close(precision, 7.75, "consistency precision");

        // Expected mean using that precision
        let expected_mean = 0.8
            + (1.0 * 2.0 / 7.75) * 0.3
            + (0.5 * 3.0 / 7.75) * (-0.2);
        assert_close(mean, expected_mean, "consistency mean");
    }
}
