use crate::model::Network;

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
    // Bound away from 0/1 to match the TAPAS HGF Toolbox (hgf_binary_level1.m): a
    // looser bound lets the binary predicted precision collapse the level-2 update in
    // high-volatility regimes and blow up the variance (esp. under the uHGF update).
    expected_mean = expected_mean.clamp(1e-3, 1.0 - 1e-3);

    let state = &mut network.attributes.states[node_idx];
    state.expected_mean = expected_mean;
    state.expected_precision = expected_mean * (1.0 - expected_mean);
}
