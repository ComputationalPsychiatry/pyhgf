use crate::model::Network;

/// Updating an exponential family state node
/// 
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `sufficient_statistics` - A function computing the sufficient statistics of an exponential family distribution.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn prediction_error_exponential_state_node(network: &mut Network, node_idx: usize, sufficient_statistics: fn(&f64) -> Vec<f64>) {

    let floats_attributes = network.attributes.floats.get_mut(&node_idx).expect("No floats attributes");
    let vectors_attributes = network.attributes.vectors.get_mut(&node_idx).expect("No vector attributes");
    let mean = floats_attributes.get("mean").expect("Mean not found");
    let nus = floats_attributes.get("nus").expect("Nus not found");
    let xis = vectors_attributes.get_mut("xis").expect("Xis not found");

    let suf_stats = sufficient_statistics(mean);
    for i in 0..suf_stats.len() {
        xis[i] = xis[i] + (1.0 / (1.0 + nus)) * (suf_stats[i] - xis[i]);
    }    
}