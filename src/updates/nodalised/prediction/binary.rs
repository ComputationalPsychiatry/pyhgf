use crate::model::network::Network;

/// Prediction from a binary state node
pub fn prediction_binary_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    let mut expected_mean: f64 = 0.0;

    if let Some(ref vp_idxs) = network.edges[node_idx].value_parents {
        for &parent_idx in vp_idxs {
            expected_mean += network.attributes.states[parent_idx].expected_mean;
        }
    }

    // Sigmoid transform
    expected_mean = 1.0 / (1.0 + (-expected_mean).exp());
    // Bound away from 0/1 for numerical stability (configurable via
    // `Network(precision_clipping_value=...)`): a larger value (e.g. 1e-3, matching the
    // TAPAS HGF Toolbox) keeps the binary predicted precision from collapsing the
    // level-2 update in high-volatility regimes; a very small value avoids flat,
    // zero-gradient plateaus that hurt gradient-based inference.
    let v = network.precision_clipping_value;
    expected_mean = expected_mean.clamp(v, 1.0 - v);

    let state = &mut network.attributes.states[node_idx];
    state.expected_mean = expected_mean;
    state.expected_precision = expected_mean * (1.0 - expected_mean);
}
