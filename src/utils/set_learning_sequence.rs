use crate::model::AdjacencyLists;
use crate::utils::function_pointer::{FnType, get_func_map};

pub struct LearningSequence {
    pub prediction_steps: Vec<(usize, FnType)>,
    pub update_steps: Vec<(usize, FnType)>,
    pub learning_steps: Vec<(usize, FnType)>,
}

/// Build a learning sequence from the network's standard update sequence.
pub fn build_learning_sequence(
    predictions: &[(usize, FnType)],
    updates: &[(usize, FnType)],
    inputs_x_idxs: &[usize],
    learning_fn: FnType,
    edges: &[AdjacencyLists],
) -> LearningSequence {
    let func_map = get_func_map();

    let prediction_steps: Vec<(usize, FnType)> = predictions
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    let update_steps: Vec<(usize, FnType)> = updates
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    let learning_steps: Vec<(usize, FnType)> = update_steps
        .iter()
        .filter_map(|&(idx, func)| {
            let func_name = func_map.get(&func).unwrap_or(&"unknown");
            if func_name.contains("prediction_error") {
                let is_learnable = edges.get(idx)
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
    use crate::updates::learning::learning_weights;
    use crate::updates::prediction_error::binary::prediction_error_binary_state_node;

    fn dummy_learning(_net: &mut Network, _idx: usize, _ts: f64) {}

    fn make_edges(idxs: &[usize]) -> Vec<AdjacencyLists> {
        let max_idx = idxs.iter().copied().max().unwrap_or(0);
        let mut edges = Vec::with_capacity(max_idx + 1);
        for i in 0..=max_idx {
            edges.push(AdjacencyLists {
                node_type: if idxs.contains(&i) { String::from("continuous-state") } else { String::from("unknown") },
                value_parents: None,
                value_children: None,
                volatility_parents: None,
                volatility_children: None,
            });
        }
        edges
    }

    #[test]
    fn test_filters_predictor_nodes_from_predictions() {
        let pred_fn = prediction_continuous_state_node as FnType;
        let predictions = vec![(0, pred_fn), (1, pred_fn), (2, pred_fn)];
        let updates: Vec<(usize, FnType)> = vec![];
        let inputs_x = [1, 2];
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
        let inputs_x = [1];
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&[], &updates, &inputs_x, dummy_learning, &edges);
        let result_idxs: Vec<usize> = seq.update_steps.iter().map(|(idx, _)| *idx).collect();
        assert!(result_idxs.iter().all(|&idx| idx != 1));
        assert_eq!(seq.update_steps.len(), 2);
    }

    #[test]
    fn test_learning_steps_match_pe_order() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let learn_fn = learning_weights as FnType;
        let edges = make_edges(&[0, 1, 2]);
        let updates = vec![(0, pe_fn), (1, pe_fn), (2, po_fn)];

        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);
        assert_eq!(seq.update_steps.len(), 3);
        assert_eq!(seq.learning_steps.len(), 2);

        let func_map = get_func_map();
        assert_eq!(seq.learning_steps[0].0, 0);
        assert!(func_map.get(&seq.learning_steps[0].1).unwrap().contains("learning"));
        assert_eq!(seq.learning_steps[1].0, 1);
    }

    #[test]
    fn test_learning_steps_with_multiple_pe_groups() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let po_fn = posterior_update_continuous_state_node as FnType;
        let learn_fn = learning_weights as FnType;
        let edges = make_edges(&[0, 1, 2, 3]);
        let updates = vec![(0, pe_fn), (2, po_fn), (1, pe_fn), (3, po_fn)];

        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);
        assert_eq!(seq.update_steps.len(), 4);
        assert_eq!(seq.learning_steps.len(), 2);
        assert_eq!(seq.learning_steps[0].0, 0);
        assert_eq!(seq.learning_steps[1].0, 1);
    }

    #[test]
    fn test_learning_step_for_trailing_pe() {
        let pe_fn = prediction_error_continuous_state_node as FnType;
        let learn_fn = learning_weights as FnType;
        let edges = make_edges(&[0]);
        let updates = vec![(0, pe_fn)];

        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);
        assert_eq!(seq.update_steps.len(), 1);
        assert_eq!(seq.learning_steps.len(), 1);
        assert_eq!(seq.learning_steps[0].0, 0);
    }

    #[test]
    fn test_empty_sequences() {
        let edges: Vec<AdjacencyLists> = Vec::new();
        let seq = build_learning_sequence(&[], &[], &[], dummy_learning, &edges);
        assert!(seq.prediction_steps.is_empty());
        assert!(seq.update_steps.is_empty());
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_no_pe_steps_means_no_learning() {
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
        let pe_binary = prediction_error_binary_state_node as FnType;
        let pe_cont = prediction_error_continuous_state_node as FnType;
        let learn_fn = learning_weights as FnType;

        let mut edges = make_edges(&[0, 1]);
        edges[0].node_type = String::from("binary-state");

        let updates = vec![(0, pe_binary), (1, pe_cont)];
        let seq = build_learning_sequence(&[], &updates, &[], learn_fn, &edges);

        assert_eq!(seq.learning_steps.len(), 1);
        assert_eq!(seq.learning_steps[0].0, 1);
    }

    #[test]
    fn test_from_real_network_2layer() {
        let mut net = Network::new("eHGF");
        net.add_nodes("continuous-state", 2, None, None, None, None, None, None);
        net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0, None, None, true);
        net.set_update_sequence();

        let learn_fn = learning_weights as FnType;
        let inputs_x = [2_usize, 3];

        let seq = build_learning_sequence(
            &net.update_sequence.predictions,
            &net.update_sequence.updates,
            &inputs_x,
            learn_fn,
            &net.edges,
        );

        for (idx, _) in &seq.prediction_steps {
            assert!(!inputs_x.contains(idx), "Predictor node {} should be filtered", idx);
        }

        let func_map = get_func_map();
        assert!(!seq.learning_steps.is_empty());
        for (_, f) in &seq.learning_steps {
            assert!(func_map.get(f).map_or(false, |n| n.contains("learning")));
        }
    }
}
