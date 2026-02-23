use crate::model::Network;


/// Inject new observations into an input node
/// 
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The input node index.
/// * `observations` - The new observations.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn observation_update(network: &mut Network, node_idx: usize, observations: f64) {

    if let Some(node) = network.attributes.floats.get_mut(&node_idx) {
        if let Some(mean) = node.get_mut("mean") {
            *mean = observations;
        }
    }
}

/// Set predictor values on top-layer nodes.
///
/// Writes the given value into the node's `"expected_mean"` attribute.
/// This mirrors Python `set_predictors` — predictor nodes have their
/// expected means clamped to the input features before the prediction pass.
pub fn set_predictors(network: &mut Network, node_idx: usize, value: f64) {
    if let Some(node) = network.attributes.floats.get_mut(&node_idx) {
        node.insert("expected_mean".into(), value);
    }
}

/// Set observation values on bottom-layer (target) nodes.
///
/// Writes the given value into the node's `"mean"` attribute and marks
/// the node as observed (`"observed" = 1.0`).
/// This mirrors Python `set_observation`.
pub fn set_observation(network: &mut Network, node_idx: usize, value: f64) {
    if let Some(node) = network.attributes.floats.get_mut(&node_idx) {
        node.insert("mean".into(), value);
        node.insert("observed".into(), 1.0);
    }
}