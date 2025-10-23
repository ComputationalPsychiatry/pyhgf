use crate::model::Network;

/// Posterior update from a continuous state node
/// 
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn posterior_update_continuous_state_node(network: &mut Network, node_idx: usize) {
    let a = 1;
}