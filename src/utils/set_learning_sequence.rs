use crate::model::AdjacencyLists;
use crate::utils::function_pointer::{FnType, get_func_map};
use std::collections::HashMap;

// =============================================================================
// Learning sequence builder
// =============================================================================

/// A learning sequence that mirrors Python's `LearningSequence` named tuple.
///
/// Contains three separate step lists that are executed in order:
/// 1. `prediction_steps` — top-down predictions (excluding predictor input nodes).
/// 2. `update_steps`     — prediction errors + posterior updates (excluding predictor
///    input nodes).
/// 3. `learning_steps`   — weight updates, one per prediction-error step, in the same
///    order as the prediction errors appear in `update_steps`.
pub struct LearningSequence {
    pub prediction_steps: Vec<(usize, FnType)>,
    pub update_steps: Vec<(usize, FnType)>,
    pub learning_steps: Vec<(usize, FnType)>,
}

/// Build a learning sequence from the network's standard update sequence.
///
/// This mirrors Python `Network.create_learning_propagation_fn`:
///
/// 1. Filter out all steps targeting predictor-input nodes (`inputs_x_idxs`)
///    from both predictions and updates.
/// 2. Scan the filtered update steps for prediction-error functions and build a
///    parallel `learning_steps` list that applies `learning_fn` to the same node,
///    in the same order.  Only continuous-state and volatile-state nodes are
///    included — other node types (e.g. binary-state) do not use coupling
///    weights in their prediction, so the linear learning rule does not apply.
///
/// # Returns
/// A [`LearningSequence`] with prediction, update, and learning steps.
pub fn build_learning_sequence(
    predictions: &[(usize, FnType)],
    updates: &[(usize, FnType)],
    inputs_x_idxs: &[usize],
    learning_fn: FnType,
    edges: &HashMap<usize, AdjacencyLists>,
) -> LearningSequence {
    let func_map = get_func_map();

    // Filter predictions: exclude predictor input nodes
    let prediction_steps: Vec<(usize, FnType)> = predictions
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    // Filter updates: exclude predictor input nodes
    let update_steps: Vec<(usize, FnType)> = updates
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    // Build learning steps: one weight-update per prediction-error step,
    // in the same order as the prediction errors appear.
    // Only continuous-state and volatile-state nodes are eligible.
    let learning_steps: Vec<(usize, FnType)> = update_steps
        .iter()
        .filter_map(|&(idx, func)| {
            let func_name = func_map.get(&func).unwrap_or(&"unknown");
            if func_name.contains("prediction_error") {
                let is_learnable = edges.get(&idx)
                    .map(|e| e.node_type == "continuous-state" || e.node_type == "volatile-state")
                    .unwrap_or(false);
                if is_learnable {
                    Some((idx, learning_fn))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    LearningSequence {
        prediction_steps,
        update_steps,
        learning_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Network;
    use crate::updates::prediction::continuous::prediction_continuous_state_node;
    use crate::updates::prediction_error::continuous::prediction_error_continuous_state_node;
    use crate::updates::posterior::continuous::posterior_update_continuous_state_node;
    use crate::utils::learning::learning_weights_fixed;

    /// Helper: a dummy function with the right signature, used as learning_fn.
    fn dummy_learning(_net: &mut Network, _idx: usize, _ts: f64) {}

    // ------------------------------------------------------------------
    // Basic filtering
    // ------------------------------------------------------------------

    /// Build a minimal edges map with continuous-state nodes for the given indices.
    fn make_edges(idxs: &[usize]) -> HashMap<usize, AdjacencyLists> {
        idxs.iter().map(|&i| (i, AdjacencyLists {
            node_type: String::from("continuous-state"),
            value_parents: None,
            value_children: None,
            volatility_parents: None,
            volatility_children: None,
        })).collect()
    }

    #[test]
    fn test_filters_predictor_nodes_from_predictions() {
        let pred_fn = prediction_continuous_state_node as FnType;
        let predictions = vec![(0, pred_fn), (1, pred_fn), (2, pred_fn)];
        let updates: Vec<(usize, FnType)> = vec![];
        let inputs_x = [1, 2]; // nodes 1 and 2 are predictors
        let edges = make_edges(&[0, 1, 2]);

        let seq = build_learning_sequence(&predictions, &updates, &inputs_x, dummy_learning, &edges);

        assert_eq!(seq.prediction_steps.len(), 1);
        assert_eq!(seq.prediction_steps[0].0, 0);
    }

    #[test]
    fn test_filters_predictor_nodes_from_updates() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let updates = vec![(0, pe_fn), (1, pe_fn), (0, po_fn), (1, po_fn)];
        let inputs_x = [1]; // node 1 is a predictor
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&[], &updates, &inputs_x, dummy_learning, &edges);

        // Node 1 steps should be gone from the update sequence
        let result_idxs: Vec<usize> = seq.update_steps.iter()
            .map(|(idx, _)| *idx)
            .collect();
        assert!(result_idxs.iter().all(|&idx| idx != 1));
        // Should have PE(0) and PO(0)
        assert_eq!(seq.update_steps.len(), 2);
    }

    // ------------------------------------------------------------------
    // Learning steps mirror PE order
    // ------------------------------------------------------------------

    #[test]
    fn test_learning_steps_match_pe_order() {
        // Typical sequence: PE(0), PE(1), PO(2)
        // Expected learning_steps: learn(0), learn(1)
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;
        let edges = make_edges(&[0, 1, 2]);

        let updates = vec![(0, pe_fn), (1, pe_fn), (2, po_fn)];

        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);

        // update_steps should be unchanged (3 steps)
        assert_eq!(seq.update_steps.len(), 3);

        // learning_steps: one per PE, same order
        assert_eq!(seq.learning_steps.len(), 2);

        let func_map = get_func_map();
        assert_eq!(seq.learning_steps[0].0, 0);
        assert!(func_map.get(&seq.learning_steps[0].1).unwrap().contains("learning"));
        assert_eq!(seq.learning_steps[1].0, 1);
        assert!(func_map.get(&seq.learning_steps[1].1).unwrap().contains("learning"));
    }

    #[test]
    fn test_learning_steps_with_multiple_pe_groups() {
        // PE(0), PO(2), PE(1), PO(3)
        // Expected learning_steps: learn(0), learn(1)
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;
        let edges = make_edges(&[0, 1, 2, 3]);

        let updates = vec![(0, pe_fn), (2, po_fn), (1, pe_fn), (3, po_fn)];

        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);

        // update_steps: all 4 steps preserved
        assert_eq!(seq.update_steps.len(), 4);

        // learning_steps: 2 PE nodes → 2 learning steps
        assert_eq!(seq.learning_steps.len(), 2);
        assert_eq!(seq.learning_steps[0].0, 0);
        assert_eq!(seq.learning_steps[1].0, 1);
    }

    #[test]
    fn test_learning_step_for_trailing_pe() {
        // PE(0) with no following posterior — learning step should still appear
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;
        let edges = make_edges(&[0]);

        let updates = vec![(0, pe_fn)];

        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);

        assert_eq!(seq.update_steps.len(), 1);
        assert_eq!(seq.learning_steps.len(), 1);
        assert_eq!(seq.learning_steps[0].0, 0);

        let func_map = get_func_map();
        assert!(func_map.get(&seq.learning_steps[0].1).unwrap().contains("learning"));
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_sequences() {
        let edges = HashMap::new();
        let seq = build_learning_sequence(&[], &[], &[], dummy_learning, &edges);
        assert!(seq.prediction_steps.is_empty());
        assert!(seq.update_steps.is_empty());
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_no_pe_steps_means_no_learning() {
        // Only posterior steps — no learning should be added
        let po_fn = posterior_update_continuous_state_node as FnType;
        let updates = vec![(0, po_fn), (1, po_fn)];
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&[], &updates, &[], dummy_learning, &edges);

        assert_eq!(seq.update_steps.len(), 2);
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_all_nodes_are_predictors() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let pred_fn = prediction_continuous_state_node as FnType;
        let predictions = vec![(0, pred_fn), (1, pred_fn)];
        let updates = vec![(0, pe_fn), (1, pe_fn)];
        let inputs_x = [0, 1];
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&predictions, &updates, &inputs_x, dummy_learning, &edges);

        assert!(seq.prediction_steps.is_empty());
        assert!(seq.update_steps.is_empty());
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_binary_node_excluded_from_learning() {
        // A binary-state node should NOT get a learning step
        use crate::updates::prediction_error::binary::prediction_error_binary_state_node;
        let pe_binary = prediction_error_binary_state_node as FnType;
        let pe_cont = prediction_error_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;

        let mut edges = make_edges(&[1]);
        edges.insert(0, AdjacencyLists {
            node_type: String::from("binary-state"),
            value_parents: None,
            value_children: None,
            volatility_parents: None,
            volatility_children: None,
        });

        let updates = vec![(0, pe_binary), (1, pe_cont)];
        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);

        // Only node 1 (continuous) should have a learning step
        assert_eq!(seq.learning_steps.len(), 1);
        assert_eq!(seq.learning_steps[0].0, 1);
    }

    #[test]
    fn test_from_real_network_2layer() {
        // Build a real 2-layer network and verify the learning sequence
        let mut net = Network::new("eHGF");
        net.add_nodes("continuous-state", 2, None, None, None, None, None, None);
        net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0, None, None);
        net.set_update_sequence();

        let learn_fn = learning_weights_fixed as FnType;
        let inputs_x = [2_usize, 3];

        let seq = build_learning_sequence(
            &net.update_sequence.predictions,
            &net.update_sequence.updates,
            &inputs_x,
            learn_fn,
            &net.edges,
        );

        // Predictions for predictor nodes (2, 3) should be filtered out
        for (idx, _) in &seq.prediction_steps {
            assert!(!inputs_x.contains(idx), "Predictor node {} should be filtered", idx);
        }

        // Learning steps should be present
        let func_map = get_func_map();
        assert!(
            !seq.learning_steps.is_empty(),
            "Learning steps should be present"
        );
        for (_, f) in &seq.learning_steps {
            assert!(
                func_map.get(f).map_or(false, |n| n.contains("learning")),
                "Each learning step should be a learning function"
            );
        }
    }
}
