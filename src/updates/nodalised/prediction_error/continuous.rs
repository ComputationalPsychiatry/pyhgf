use crate::model::network::Network;
use crate::updates::nodalised::posterior::continuous::has_observations;

/// Prediction error from a continuous state node.
///
/// A node with value or volatility children also records whether at least one of
/// them was observed at this step, in its `observed` flag. Its parents then treat it
/// as unobserved after a step without observations: they receive no prediction error
/// from it and keep their predicted precision. The flag of a node without children is
/// the one set by the observation step (mirrors `continuous_node_prediction_error` in
/// the JAX backend).
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

    // A node with children is observed when at least one of its children was.
    let edges = &network.edges[node_idx];
    let observed = if edges.value_children.is_some() || edges.volatility_children.is_some() {
        if has_observations(network, node_idx) {
            1.0
        } else {
            0.0
        }
    } else {
        network.attributes.states[node_idx].observed
    };

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_prediction_error;
    state.volatility_prediction_error = volatility_prediction_error;
    state.observed = observed;
}
