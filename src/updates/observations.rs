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