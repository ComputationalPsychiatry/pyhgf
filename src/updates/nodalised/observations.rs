use crate::model::network::Network;

/// Inject new observations into an input node, with their observation flag (`0.0` for
/// a missing observation).
pub fn observation_update(
    network: &mut Network,
    node_idx: usize,
    observations: f64,
    observed: f64,
) {
    let state = &mut network.attributes.states[node_idx];
    state.mean = observations;
    state.observed = observed;
}

/// Set observation values on bottom-layer (target) nodes.
pub fn set_observation(network: &mut Network, node_idx: usize, value: f64) {
    let state = &mut network.attributes.states[node_idx];
    state.mean = value;
    state.observed = 1.0;
}
