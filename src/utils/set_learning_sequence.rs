use crate::model::AdjacencyLists;
use crate::utils::function_pointer::UpdateStep;

pub struct LearningSequence {
    pub prediction_steps: Vec<(usize, UpdateStep)>,
    pub update_steps: Vec<(usize, UpdateStep)>,
    pub learning_steps: Vec<(usize, UpdateStep)>,
}

/// Build a learning sequence from the network's standard update sequence.
pub fn build_learning_sequence(
    predictions: &[(usize, UpdateStep)],
    updates: &[(usize, UpdateStep)],
    inputs_x_idxs: &[usize],
    edges: &[AdjacencyLists],
) -> LearningSequence {
    let prediction_steps: Vec<(usize, UpdateStep)> = predictions
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    let update_steps: Vec<(usize, UpdateStep)> = updates
        .iter()
        .filter(|(idx, _)| !inputs_x_idxs.contains(idx))
        .cloned()
        .collect();

    let learning_steps: Vec<(usize, UpdateStep)> = update_steps
        .iter()
        .filter_map(|&(idx, step)| {
            if step.name().contains("prediction_error") {
                let is_learnable = edges.get(idx)
                    .map(|e| {
                        e.node_type == "continuous-state"
                            || e.node_type == "volatile-state"
                            || e.node_type == "binary-state"
                    })
                    .unwrap_or(false);
                if is_learnable {
                    Some((idx, UpdateStep::LearningWeights))
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

    fn make_edges(idxs: &[usize]) -> Vec<AdjacencyLists> {
        let max_idx = idxs.iter().copied().max().unwrap_or(0);
        let mut edges = Vec::with_capacity(max_idx + 1);
        for i in 0..=max_idx {
            edges.push(AdjacencyLists {
                node_type: if idxs.contains(&i) { String::from("continuous-state") } else { String::from("unknown") },
                learning_kind: String::from("precision_weighted"),
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
        let predictions = vec![
            (0, UpdateStep::PredictionContinuous),
            (1, UpdateStep::PredictionContinuous),
            (2, UpdateStep::PredictionContinuous),
        ];
        let updates: Vec<(usize, UpdateStep)> = vec![];
        let inputs_x = [1, 2];
        let edges = make_edges(&[0, 1, 2]);

        let seq = build_learning_sequence(&predictions, &updates, &inputs_x, &edges);
        assert_eq!(seq.prediction_steps.len(), 1);
        assert_eq!(seq.prediction_steps[0].0, 0);
    }

    #[test]
    fn test_filters_predictor_nodes_from_updates() {
        let updates = vec![
            (0, UpdateStep::PredictionErrorContinuous),
            (1, UpdateStep::PredictionErrorContinuous),
            (0, UpdateStep::PosteriorContinuous),
            (1, UpdateStep::PosteriorContinuous),
        ];
        let inputs_x = [1];
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&[], &updates, &inputs_x, &edges);
        let result_idxs: Vec<usize> = seq.update_steps.iter().map(|(idx, _)| *idx).collect();
        assert!(result_idxs.iter().all(|&idx| idx != 1));
        assert_eq!(seq.update_steps.len(), 2);
    }

    #[test]
    fn test_learning_steps_match_pe_order() {
        let edges = make_edges(&[0, 1, 2]);
        let updates = vec![
            (0, UpdateStep::PredictionErrorContinuous),
            (1, UpdateStep::PredictionErrorContinuous),
            (2, UpdateStep::PosteriorContinuous),
        ];

        let seq = build_learning_sequence(&[], &updates, &[], &edges);
        assert_eq!(seq.update_steps.len(), 3);
        assert_eq!(seq.learning_steps.len(), 2);

        assert_eq!(seq.learning_steps[0].0, 0);
        assert_eq!(seq.learning_steps[0].1, UpdateStep::LearningWeights);
        assert_eq!(seq.learning_steps[1].0, 1);
    }

    #[test]
    fn test_learning_steps_with_multiple_pe_groups() {
        let edges = make_edges(&[0, 1, 2, 3]);
        let updates = vec![
            (0, UpdateStep::PredictionErrorContinuous),
            (2, UpdateStep::PosteriorContinuous),
            (1, UpdateStep::PredictionErrorContinuous),
            (3, UpdateStep::PosteriorContinuous),
        ];

        let seq = build_learning_sequence(&[], &updates, &[], &edges);
        assert_eq!(seq.update_steps.len(), 4);
        assert_eq!(seq.learning_steps.len(), 2);
        assert_eq!(seq.learning_steps[0].0, 0);
        assert_eq!(seq.learning_steps[1].0, 1);
    }

    #[test]
    fn test_learning_step_for_trailing_pe() {
        let edges = make_edges(&[0]);
        let updates = vec![(0, UpdateStep::PredictionErrorContinuous)];

        let seq = build_learning_sequence(&[], &updates, &[], &edges);
        assert_eq!(seq.update_steps.len(), 1);
        assert_eq!(seq.learning_steps.len(), 1);
        assert_eq!(seq.learning_steps[0].0, 0);
    }

    #[test]
    fn test_empty_sequences() {
        let edges: Vec<AdjacencyLists> = Vec::new();
        let seq = build_learning_sequence(&[], &[], &[], &edges);
        assert!(seq.prediction_steps.is_empty());
        assert!(seq.update_steps.is_empty());
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_no_pe_steps_means_no_learning() {
        let updates = vec![
            (0, UpdateStep::PosteriorContinuous),
            (1, UpdateStep::PosteriorContinuous),
        ];
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&[], &updates, &[], &edges);
        assert_eq!(seq.update_steps.len(), 2);
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_all_nodes_are_predictors() {
        let predictions = vec![
            (0, UpdateStep::PredictionContinuous),
            (1, UpdateStep::PredictionContinuous),
        ];
        let updates = vec![
            (0, UpdateStep::PredictionErrorContinuous),
            (1, UpdateStep::PredictionErrorContinuous),
        ];
        let inputs_x = [0, 1];
        let edges = make_edges(&[0, 1]);

        let seq = build_learning_sequence(&predictions, &updates, &inputs_x, &edges);
        assert!(seq.prediction_steps.is_empty());
        assert!(seq.update_steps.is_empty());
        assert!(seq.learning_steps.is_empty());
    }

    #[test]
    fn test_binary_node_included_in_learning() {
        let mut edges = make_edges(&[0, 1]);
        edges[0].node_type = String::from("binary-state");

        let updates = vec![
            (0, UpdateStep::PredictionErrorBinary),
            (1, UpdateStep::PredictionErrorContinuous),
        ];
        let seq = build_learning_sequence(&[], &updates, &[], &edges);

        // Both binary-state (0) and continuous-state (1) are now learnable.
        assert_eq!(seq.learning_steps.len(), 2);
        assert!(seq.learning_steps.iter().any(|(idx, _)| *idx == 0));
        assert!(seq.learning_steps.iter().any(|(idx, _)| *idx == 1));
    }

    #[test]
    fn test_from_real_network_2layer() {
        let mut net = Network::new("eHGF");
        net.add_nodes("continuous-state", 2, None, None, None, None, None, None);
        net.add_layer(2, "continuous-state", Some(vec![0, 1]), 1.0, None, None, true);
        net.set_update_sequence();

        let inputs_x = [2_usize, 3];

        let seq = build_learning_sequence(
            &net.update_sequence.predictions,
            &net.update_sequence.updates,
            &inputs_x,
            &net.edges,
        );

        for (idx, _) in &seq.prediction_steps {
            assert!(!inputs_x.contains(idx), "Predictor node {} should be filtered", idx);
        }

        assert!(!seq.learning_steps.is_empty());
        for (_, step) in &seq.learning_steps {
            assert!(step.name().contains("learning"));
        }
    }
}
