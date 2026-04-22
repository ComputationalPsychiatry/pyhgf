use crate::model::Network;

/// Prediction error for a binary state node
pub fn prediction_error_binary_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    let mean = network.attributes.states[node_idx].mean;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let observed = network.attributes.states[node_idx].observed;

    let value_prediction_error =
        (mean - expected_mean) * observed / expected_precision;

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_prediction_error;
    state.precision = expected_precision;
}
