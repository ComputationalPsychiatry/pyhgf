use crate::model::Network;

/// Inject new observations into an input node
pub fn observation_update(network: &mut Network, node_idx: usize, observations: f64) {
    network.attributes.states[node_idx].mean = observations;
}

/// Set predictor values on top-layer nodes.
pub fn set_predictors(network: &mut Network, node_idx: usize, value: f64) {
    let state = &mut network.attributes.states[node_idx];
    state.mean = value;
    state.expected_mean = value;
}

/// Set observation values on bottom-layer (target) nodes.
pub fn set_observation(network: &mut Network, node_idx: usize, value: f64) {
    let state = &mut network.attributes.states[node_idx];
    state.mean = value;
    state.observed = 1.0;
}
