use crate::utils::function_pointer::{FnType, get_func_map};

// =============================================================================
// Learning sequence builder
// =============================================================================

/// Build a learning update sequence by interleaving weight-update steps.
///
/// Given the network's standard prediction and update sequences, this function:
/// 1. Filters out all steps targeting predictor-input nodes (`inputs_x_idxs`).
/// 2. Scans the update sequence for prediction-error steps and accumulates a
///    weight-update step (using `learning_fn`) for each.
/// 3. When a posterior-update step is encountered, all accumulated weight-update
///    steps are flushed **before** the posterior step.
///
/// This mirrors Python `Network.create_learning_propagation_fn`.
///
/// # Returns
/// A tuple `(filtered_predictions, interleaved_updates)`.
pub fn build_learning_sequence(
    predictions: &[(usize, FnType)],
    updates: &[(usize, FnType)],
    inputs_x_idxs: &[usize],
    learning_fn: FnType,
) -> (Vec<(usize, FnType)>, Vec<(usize, FnType)>) {
    let func_map = get_func_map();

    // Filter predictions: exclude predictor input nodes
    let filtered_predictions: Vec<(usize, FnType)> = predictions
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    // Filter updates: exclude predictor input nodes
    let filtered_updates: Vec<(usize, FnType)> = updates
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    // Interleave weight-update steps between prediction-error and posterior steps
    let mut result_updates: Vec<(usize, FnType)> = Vec::new();
    let mut weight_updates: Vec<(usize, FnType)> = Vec::new();

    for &(idx, func) in &filtered_updates {
        let func_name = func_map.get(&func).unwrap_or(&"unknown");

        if func_name.contains("prediction_error") {
            weight_updates.push((idx, learning_fn));
        }

        if func_name.contains("posterior") && !weight_updates.is_empty() {
            // Flush all accumulated weight updates before this posterior step
            result_updates.extend(weight_updates.drain(..));
        }

        result_updates.push((idx, func));
    }

    // Flush any remaining weight updates at the end.
    // This handles PEs whose parents' posterior steps were filtered out
    // (e.g. predictor nodes) — weights are still learned, applied after
    // all other updates so they're ready for the next time step.
    if !weight_updates.is_empty() {
        result_updates.extend(weight_updates.drain(..));
    }

    (filtered_predictions, result_updates)
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

    #[test]
    fn test_filters_predictor_nodes_from_predictions() {
        let pred_fn = prediction_continuous_state_node as FnType;
        let predictions = vec![(0, pred_fn), (1, pred_fn), (2, pred_fn)];
        let updates: Vec<(usize, FnType)> = vec![];
        let inputs_x = [1, 2]; // nodes 1 and 2 are predictors

        let (filtered_preds, _) =
            build_learning_sequence(&predictions, &updates, &inputs_x, dummy_learning);

        assert_eq!(filtered_preds.len(), 1);
        assert_eq!(filtered_preds[0].0, 0);
    }

    #[test]
    fn test_filters_predictor_nodes_from_updates() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let updates = vec![(0, pe_fn), (1, pe_fn), (0, po_fn), (1, po_fn)];
        let inputs_x = [1]; // node 1 is a predictor

        let (_, result_updates) =
            build_learning_sequence(&[], &updates, &inputs_x, dummy_learning);

        // Node 1 steps should be gone from the original sequence
        let original_idxs: Vec<usize> = updates.iter()
            .filter(|(idx, _)| *idx != 1)
            .map(|(idx, _)| *idx)
            .collect();
        let result_original_idxs: Vec<usize> = result_updates.iter()
            .filter(|(_, f)| !std::ptr::fn_addr_eq(*f, dummy_learning as FnType))
            .map(|(idx, _)| *idx)
            .collect();
        assert_eq!(result_original_idxs, original_idxs);
    }

    // ------------------------------------------------------------------
    // Interleaving
    // ------------------------------------------------------------------

    #[test]
    fn test_interleaves_weight_updates_before_posterior() {
        // Typical sequence: PE(0), PE(1), PO(2)
        // Expected: PE(0), PE(1), learn(0), learn(1), PO(2)
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;

        let updates = vec![(0, pe_fn), (1, pe_fn), (2, po_fn)];

        let (_, result) =
            build_learning_sequence(&[], &updates, &[], learn_fn);

        // Collect names via func_map for verification
        let func_map = get_func_map();
        let names: Vec<(usize, &str)> = result.iter()
            .map(|(idx, f)| (*idx, *func_map.get(f).unwrap_or(&"unknown")))
            .collect();

        // PE(0), PE(1), learn(0), learn(1), PO(2)
        assert_eq!(names.len(), 5);
        assert!(names[0].1.contains("prediction_error") && names[0].0 == 0);
        assert!(names[1].1.contains("prediction_error") && names[1].0 == 1);
        assert!(names[2].1.contains("learning") && names[2].0 == 0);
        assert!(names[3].1.contains("learning") && names[3].0 == 1);
        assert!(names[4].1.contains("posterior") && names[4].0 == 2);
    }

    #[test]
    fn test_multiple_posterior_steps_get_separate_flushes() {
        // PE(0), PO(2), PE(1), PO(3)
        // Expected: PE(0), learn(0), PO(2), PE(1), learn(1), PO(3)
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;

        let updates = vec![(0, pe_fn), (2, po_fn), (1, pe_fn), (3, po_fn)];

        let (_, result) =
            build_learning_sequence(&[], &updates, &[], learn_fn);

        let func_map = get_func_map();
        let names: Vec<(usize, &str)> = result.iter()
            .map(|(idx, f)| (*idx, *func_map.get(f).unwrap_or(&"unknown")))
            .collect();

        assert_eq!(names.len(), 6);
        // First batch
        assert!(names[0].1.contains("prediction_error") && names[0].0 == 0);
        assert!(names[1].1.contains("learning") && names[1].0 == 0);
        assert!(names[2].1.contains("posterior") && names[2].0 == 2);
        // Second batch
        assert!(names[3].1.contains("prediction_error") && names[3].0 == 1);
        assert!(names[4].1.contains("learning") && names[4].0 == 1);
        assert!(names[5].1.contains("posterior") && names[5].0 == 3);
    }

    #[test]
    fn test_trailing_pe_flushed_at_end() {
        // PE(0) with no following posterior — weight update should still appear
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let learn_fn = learning_weights_fixed as FnType;

        let updates = vec![(0, pe_fn)];

        let (_, result) =
            build_learning_sequence(&[], &updates, &[], learn_fn);

        let func_map = get_func_map();
        let names: Vec<&str> = result.iter()
            .map(|(_, f)| *func_map.get(f).unwrap_or(&"unknown"))
            .collect();

        assert_eq!(result.len(), 2);
        assert!(names[0].contains("prediction_error"));
        assert!(names[1].contains("learning"));
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_sequences() {
        let (preds, updates) =
            build_learning_sequence(&[], &[], &[], dummy_learning);
        assert!(preds.is_empty());
        assert!(updates.is_empty());
    }

    #[test]
    fn test_no_pe_steps_means_no_learning_inserted() {
        // Only posterior steps — no learning should be added
        let po_fn = posterior_update_continuous_state_node as FnType;
        let updates = vec![(0, po_fn), (1, po_fn)];

        let (_, result) =
            build_learning_sequence(&[], &updates, &[], dummy_learning);

        assert_eq!(result.len(), 2);
        for (_, f) in &result {
            assert!(!std::ptr::fn_addr_eq(*f, dummy_learning as FnType));
        }
    }

    #[test]
    fn test_all_nodes_are_predictors() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let pred_fn = prediction_continuous_state_node as FnType;
        let predictions = vec![(0, pred_fn), (1, pred_fn)];
        let updates = vec![(0, pe_fn), (1, pe_fn)];
        let inputs_x = [0, 1];

        let (preds, upds) =
            build_learning_sequence(&predictions, &updates, &inputs_x, dummy_learning);

        assert!(preds.is_empty());
        assert!(upds.is_empty());
    }

    #[test]
    fn test_from_real_network_2layer() {
        // Build a real 2-layer network and verify the learning sequence
        let mut net = Network::new("eHGF");
        net.add_nodes("continuous-state", 2, None, None, None, None, None, None);
        net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0);
        net.set_update_sequence();

        let learn_fn = learning_weights_fixed as FnType;
        let inputs_x = [2_usize, 3];

        let (preds, upds) = build_learning_sequence(
            &net.update_sequence.predictions,
            &net.update_sequence.updates,
            &inputs_x,
            learn_fn,
        );

        // Predictions for predictor nodes (2, 3) should be filtered out
        for (idx, _) in &preds {
            assert!(!inputs_x.contains(idx), "Predictor node {} should be filtered", idx);
        }

        // The update sequence should contain learning steps
        let func_map = get_func_map();
        let has_learning = upds.iter().any(|(_, f)| {
            func_map.get(f).map_or(false, |n| n.contains("learning"))
        });
        assert!(has_learning, "Learning steps should be present in the interleaved sequence");
    }
}
