use crate::model::Network;

/// Prediction from a continuous state node
pub fn prediction_continuous_state_node(network: &mut Network, node_idx: usize, time_step: f64) {
    // Copy own scalar state (f64 is Copy — no borrow held)
    let mean = network.attributes.states[node_idx].mean;
    let tonic_drift = network.attributes.states[node_idx].tonic_drift;
    let autoconnection_strength = network.attributes.states[node_idx].autoconnection_strength;
    let precision = network.attributes.states[node_idx].precision;
    let tonic_volatility = network.attributes.states[node_idx].tonic_volatility;

    // -------------------------------------------------------
    // 1. Predict the mean: μ̂ = λ · μ + Δt · driftrate
    // -------------------------------------------------------
    let mut driftrate = tonic_drift;

    if let Some(ref vp_idxs) = network.edges[node_idx].value_parents {
        let couplings = &network.attributes.vectors[node_idx].value_coupling_parents;
        let coupling_fns = &network.attributes.fn_ptrs[node_idx].value_coupling_fn_parents;

        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let psi = couplings.get(i).copied().unwrap_or(1.0);
            let fn_ptr = coupling_fns.get(i).copied().unwrap_or(&crate::math::LINEAR);
            let parent_value = (fn_ptr.f)(parent_expected_mean);
            driftrate += psi * parent_value;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // -------------------------------------------------------
    // 2. Predict the precision: π̂ = 1 / (1/π + Ω)
    // -------------------------------------------------------
    let mut total_volatility = tonic_volatility;

    if let Some(ref vol_parent_idxs) = network.edges[node_idx].volatility_parents {
        let vol_couplings = &network.attributes.vectors[node_idx].volatility_coupling_parents;

        for (i, &parent_idx) in vol_parent_idxs.iter().enumerate() {
            let parent_mean = network.attributes.states[parent_idx].mean;
            let kappa = vol_couplings.get(i).copied().unwrap_or(1.0);
            total_volatility += kappa * parent_mean;
        }
    }

    let predicted_volatility = (time_step * total_volatility.clamp(-80.0, 80.0).exp()).max(1e-128);
    let expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;

    // -------------------------------------------------------
    // 3. Store results
    // -------------------------------------------------------
    let is_input = network.edges[node_idx].value_children.is_none()
        && network.edges[node_idx].volatility_children.is_none();
    let has_volatility_parents = network.edges[node_idx].volatility_parents.is_some();

    let state = &mut network.attributes.states[node_idx];
    state.current_variance = 1.0 / precision;
    state.expected_mean = expected_mean;
    state.effective_precision = effective_precision;

    if !(is_input && !has_volatility_parents) {
        state.expected_precision = expected_precision;
    }
}
