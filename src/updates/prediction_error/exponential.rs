use crate::model::Network;
use crate::math::sufficient_statistics;

/// Updating an exponential family state node
pub fn prediction_error_exponential_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    let mean = network.attributes.states[node_idx].mean;
    let nus = network.attributes.states[node_idx].nus;

    let suf_stats = sufficient_statistics(&mean);
    let xis = &mut network.attributes.vectors[node_idx].xis;
    for i in 0..suf_stats.len() {
        xis[i] = xis[i] + (1.0 / (1.0 + nus)) * (suf_stats[i] - xis[i]);
    }
}
