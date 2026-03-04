use crate::model::Network;

/// Prediction from a binary state node
///
/// The expected mean is obtained by applying the sigmoid transform to the sum
/// of the value parents' expected means:
///
/// ```text
/// μ̂_b = σ(Σ μ̂_parent)
/// ```
///
/// The expected precision encodes the *uncertainty* at the first level:
///
/// ```text
/// π̂_b = μ̂_b · (1 − μ̂_b)
/// ```
///
/// This convention lets the parent's posterior update use the value directly
/// (eq. 81, Weber et al., v2).  To compensate, the prediction‑error step
/// divides by expected_precision.
pub fn prediction_binary_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {

    // Sum the expected means of all value parents
    let mut expected_mean: f64 = 0.0;

    let value_parents = network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.clone());

    if let Some(ref vp_idxs) = value_parents {
        for &parent_idx in vp_idxs {
            let parent_expected_mean = *network.attributes.floats
                .get(&parent_idx)
                .expect("No floats attributes found for value parent")
                .get("expected_mean")
                .expect("expected_mean not found for value parent");
            expected_mean += parent_expected_mean;
        }
    }

    // Sigmoid transform
    expected_mean = 1.0 / (1.0 + (-expected_mean).exp());

    // Clip for numerical stability
    expected_mean = expected_mean.clamp(1e-6, 1.0 - 1e-6);

    // Store expected mean
    let floats = network.attributes.floats
        .get_mut(&node_idx)
        .expect("No floats attributes found for binary node");
    floats.insert("expected_mean".into(), expected_mean);

    // expected_precision = μ̂ · (1 − μ̂)
    let expected_precision = expected_mean * (1.0 - expected_mean);
    floats.insert("expected_precision".into(), expected_precision);
}
