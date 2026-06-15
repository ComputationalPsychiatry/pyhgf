use crate::model::Network;

/// Prediction error from a continuous state node
pub fn prediction_error_continuous_state_node(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let n_volatility_parents = network.edges[node_idx]
        .volatility_parents
        .as_ref()
        .map(|vp| vp.len());

    let mean = network.attributes.states[node_idx].mean;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let precision = network.attributes.states[node_idx].precision;
    let expected_precision = network.attributes.states[node_idx].expected_precision;

    // Value prediction error: δ = μ - μ̂
    let value_prediction_error = mean - expected_mean;

    // Volatility prediction error: Δ = (π̂ / π) + π̂ · δ² - 1
    let mut volatility_prediction_error = (expected_precision / precision)
        + expected_precision * (mean - expected_mean).powi(2)
        - 1.0;
    if let Some(n) = n_volatility_parents {
        volatility_prediction_error /= n as f64;
    }

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_prediction_error;
    state.volatility_prediction_error = volatility_prediction_error;
}
