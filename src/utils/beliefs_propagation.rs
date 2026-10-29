use crate::{
    model::network::Network, updates::nodalised::observations::observation_update,
    utils::function_pointer::UpdateStep,
};

/// Single time slice belief propagation. `observed_set` flags each observation (`0.0`
/// for a missing one).
#[inline(always)]
pub fn belief_propagation(
    network: &mut Network,
    observations_set: &[f64],
    observed_set: &[f64],
    predictions: &[(usize, UpdateStep)],
    updates: &[(usize, UpdateStep)],
    time_step: f64,
) {
    // 1. prediction steps
    for &(idx, step) in predictions {
        step.call(network, idx, time_step);
    }

    // 2. observation steps
    for (i, (&observation, &observed)) in observations_set.iter().zip(observed_set).enumerate() {
        let idx = network.inputs[i];
        observation_update(network, idx, observation, observed);
    }

    // 3. update steps
    for &(idx, step) in updates {
        step.call(network, idx, time_step);
    }
}
