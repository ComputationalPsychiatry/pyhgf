use crate::{utils::function_pointer::FnType, model::Network, updates::observations::observation_update};

/// Single time slice belief propagation.
/// 
/// # Arguments
/// * `observations` - A vector of values, each value is one new observation associated
/// with one node.
pub fn belief_propagation(network: &mut Network, observations_set: Vec<f64>, predictions: & Vec<(usize, FnType)>, updates: & Vec<(usize, FnType)>) {

    // 1. prediction steps
    for (idx, step) in predictions.iter() {
        step(network, *idx);
    }
    
    // 2. observation steps
    for (i, observations) in observations_set.iter().enumerate() {
        let idx = network.inputs[i];
        observation_update(network, idx, *observations);
    } 

    // 3. update steps
    for (idx, step) in updates.iter() {
        step(network, *idx);
    }
}
