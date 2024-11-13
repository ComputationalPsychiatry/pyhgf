use crate::model::Network;

/// Prediction from a continuous state node
/// 
/// # Arguments
/// * `network` - The main network structure.
/// * `node_idx` - The node index.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn prediction_continuous_state_node(network: &mut Network, node_idx: usize) {
    let a = 1;
}