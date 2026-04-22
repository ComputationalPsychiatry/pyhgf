use crate::model::Network;

/// Prediction error for a binary state node
pub fn prediction_error_binary_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    let mean = network.attributes.states[node_idx].mean;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let observed = network.attributes.states[node_idx].observed;

    let n_parents = network.edges[node_idx].value_parents
        .as_ref()
        .map_or(1, |vp| vp.len()) as f64;

    let value_prediction_error =
        (mean - expected_mean) * observed / expected_precision / n_parents;

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_prediction_error;
    state.precision = expected_precision;
}
