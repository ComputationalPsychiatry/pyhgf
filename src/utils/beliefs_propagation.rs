use crate::{utils::function_pointer::UpdateStep, model::Network, updates::observations::observation_update};

/// Single time slice belief propagation.
#[inline(always)]
pub fn belief_propagation(network: &mut Network, observations_set: &[f64], predictions: &[(usize, UpdateStep)], updates: &[(usize, UpdateStep)], time_step: f64) {

    // 1. prediction steps
    for &(idx, step) in predictions {
        step.call(network, idx, time_step);
    }

    // 2. observation steps
    for (i, &observation) in observations_set.iter().enumerate() {
        let idx = network.inputs[i];
        observation_update(network, idx, observation);
    }

    // 3. update steps
    for &(idx, step) in updates {
        step.call(network, idx, time_step);
    }
}
