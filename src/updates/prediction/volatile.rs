use crate::model::Network;

/// Prediction from a volatile state node
pub fn prediction_volatile_state_node(network: &mut Network, node_idx: usize, time_step: f64) {
    // Copy own scalar state
    let precision = network.attributes.states[node_idx].precision;
    let mean = network.attributes.states[node_idx].mean;
    let autoconnection_strength = network.attributes.states[node_idx].autoconnection_strength;
    let tonic_volatility = network.attributes.states[node_idx].tonic_volatility;
    let mean_vol = network.attributes.states[node_idx].mean_vol;
    let precision_vol = network.attributes.states[node_idx].precision_vol;
    let tonic_volatility_vol = network.attributes.states[node_idx].tonic_volatility_vol;
    let volatility_coupling_internal = network.attributes.states[node_idx].volatility_coupling_internal;

    // Store current variance for unbounded updates
    let current_variance = 1.0 / precision;

    // ===================================================================
    // 1. PREDICT VOLATILITY LEVEL (implicit internal state)
    // ===================================================================    
    let pvv_raw = time_step * tonic_volatility_vol.exp();
    let predicted_volatility_vol = if pvv_raw > 1e-128 { pvv_raw } else { f64::NAN };
    let expected_precision_vol = 1.0 / ((1.0 / precision_vol) + predicted_volatility_vol);
    let effective_precision_vol = predicted_volatility_vol * expected_precision_vol;

    // ===================================================================
    // 2. PREDICT VALUE LEVEL (external facing)
    // ===================================================================

    // --- 2a. Predict precision (depends on volatility level) ---
    let total_volatility = tonic_volatility + volatility_coupling_internal * mean_vol;
    let pv_raw = time_step * total_volatility.exp();
    let predicted_volatility = if pv_raw > 1e-128 { pv_raw } else { f64::NAN };
    let expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;

    // --- 2b. Predict mean (including value parents if any) ---
    let mut driftrate = 0.0;
    if let Some(ref vp_idxs) = network.edges[node_idx].value_parents {
        let couplings = &network.attributes.vectors[node_idx].value_coupling_parents;

        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let value_coupling_parent = couplings.get(i).copied().unwrap_or(1.0);
            let parent_value = match network.attributes.fn_ptrs[parent_idx].coupling_fn {
                Some(cf) => (cf.f)(parent_expected_mean),
                None => parent_expected_mean,
            };
            driftrate += value_coupling_parent * parent_value;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // Input/leaf override: a volatile-state node with no value children is an
    // observed input — it does not undergo a Gaussian random walk between
    // observations, so the tonic-volatility contribution to the value-level
    // expected precision is dropped (matches the continuous-node treatment in
    // `prediction_continuous_state_node`).
    let is_input = network.edges[node_idx].value_children.is_none();

    // Store all results
    let state = &mut network.attributes.states[node_idx];
    state.current_variance = current_variance;
    state.expected_mean_vol = mean_vol;
    state.expected_precision_vol = expected_precision_vol;
    state.effective_precision_vol = effective_precision_vol;
    state.expected_mean = expected_mean;
    if is_input {
        state.expected_precision = precision;
        state.effective_precision = 0.0;
    } else {
        state.expected_precision = expected_precision;
        state.effective_precision = effective_precision;
    }
}
