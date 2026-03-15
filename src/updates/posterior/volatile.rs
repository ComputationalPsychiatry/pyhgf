use crate::model::Network;

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Parametrised sigmoid: sigmoid(phi * (x - theta))
fn s_func(x: f64, theta: f64, phi: f64) -> f64 {
    sigmoid(phi * (x - theta))
}

/// Smoothed rectangular weighting function b
fn b_func(x: f64, theta_l: f64, phi_l: f64, theta_r: f64, phi_r: f64) -> f64 {
    s_func(x, theta_l, phi_l) * (1.0 - s_func(x, theta_r, phi_r))
}

// =============================================================================
// Value level updates
// =============================================================================

fn precision_update_value_level(network: &Network, node_idx: usize) -> f64 {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let mut posterior_precision = expected_precision;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_expected_mean = network.attributes.states[node_idx].expected_mean;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let parent_pos = network.edges[child_idx].value_parents.as_ref()
                .and_then(|vp| vp.iter().position(|&p| p == node_idx));

            let coupling_fn = parent_pos.and_then(|pos| {
                network.attributes.fn_ptrs[child_idx].value_coupling_fn_parents.get(pos).copied()
            });

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_expected_mean);
                    let g_second = (cf.d2f)(parent_expected_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            posterior_precision += child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term);
        }
    }

    posterior_precision
}

fn mean_update_value_level(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let mut value_pwpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_expected_mean = network.attributes.states[node_idx].expected_mean;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let parent_pos = network.edges[child_idx].value_parents.as_ref()
                .and_then(|vp| vp.iter().position(|&p| p == node_idx));

            let coupling_fn_prime = parent_pos
                .and_then(|pos| {
                    network.attributes.fn_ptrs[child_idx].value_coupling_fn_parents.get(pos).copied()
                })
                .map(|cf| (cf.df)(parent_expected_mean))
                .unwrap_or(1.0);

            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision) * child_vape;
        }
    }

    expected_mean + value_pwpe
}

// =============================================================================
// Volatility level updates
// =============================================================================

fn precision_update_volatility_level(network: &Network, node_idx: usize) -> f64 {
    let s = &network.attributes.states[node_idx];
    let expected_precision_vol = s.expected_precision_vol;
    let volatility_pe = s.volatility_prediction_error;
    let effective_precision = s.effective_precision;
    let volatility_coupling = s.volatility_coupling_internal;

    expected_precision_vol
        + 0.5 * (volatility_coupling * effective_precision).powi(2)
        + (volatility_coupling * effective_precision).powi(2) * volatility_pe
        - 0.5 * volatility_coupling.powi(2) * effective_precision * volatility_pe
}

fn mean_update_volatility_level(network: &Network, node_idx: usize, node_precision_vol: f64) -> f64 {
    let s = &network.attributes.states[node_idx];
    let expected_mean_vol = s.expected_mean_vol;
    let volatility_pe = s.volatility_prediction_error;
    let effective_precision = s.effective_precision;
    let volatility_coupling = s.volatility_coupling_internal;
    let observed = s.observed;

    let precision_weighted_pe =
        (volatility_coupling * effective_precision * volatility_pe)
        / (2.0 * node_precision_vol);

    expected_mean_vol + precision_weighted_pe * observed
}

// =============================================================================
// Prediction error recomputation
// =============================================================================

fn recompute_prediction_errors(network: &mut Network, node_idx: usize) {
    let mean = network.attributes.states[node_idx].mean;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let precision = network.attributes.states[node_idx].precision;
    let expected_precision = network.attributes.states[node_idx].expected_precision;

    let n_value_parents = network.edges[node_idx].value_parents.as_ref().map(|vp| vp.len());

    let mut value_pe = mean - expected_mean;
    if let Some(n) = n_value_parents {
        value_pe /= n as f64;
    }

    let volatility_pe =
        (expected_precision / precision)
        + expected_precision * value_pe.powi(2)
        - 1.0;

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_pe;
    state.volatility_prediction_error = volatility_pe;
}

// =============================================================================
// Standard posterior update
// =============================================================================

pub fn posterior_update_volatile_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    // 1. UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.states[node_idx].precision = precision_value;

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.states[node_idx].mean = mean_value;

    // 2. RECOMPUTE PREDICTION ERRORS
    recompute_prediction_errors(network, node_idx);

    // 3. UPDATE VOLATILITY LEVEL
    let precision_vol = precision_update_volatility_level(network, node_idx);
    network.attributes.states[node_idx].precision_vol = precision_vol;

    let mean_vol = mean_update_volatility_level(network, node_idx, precision_vol);
    network.attributes.states[node_idx].mean_vol = mean_vol;
}

// =============================================================================
// eHGF posterior update
// =============================================================================

pub fn posterior_update_volatile_state_node_ehgf(network: &mut Network, node_idx: usize, _time_step: f64) {
    // 1. UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.states[node_idx].precision = precision_value;

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.states[node_idx].mean = mean_value;

    // 2. RECOMPUTE PREDICTION ERRORS
    recompute_prediction_errors(network, node_idx);

    // 3. UPDATE VOLATILITY LEVEL (eHGF: mean first using expected precision)
    let expected_precision_vol = network.attributes.states[node_idx].expected_precision_vol;

    let mean_vol = mean_update_volatility_level(network, node_idx, expected_precision_vol);
    network.attributes.states[node_idx].mean_vol = mean_vol;

    let precision_vol = precision_update_volatility_level(network, node_idx);
    network.attributes.states[node_idx].precision_vol = precision_vol;
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

pub fn posterior_update_volatile_state_node_unbounded(network: &mut Network, node_idx: usize, _time_step: f64) {
    // 1. UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.states[node_idx].precision = precision_value;

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.states[node_idx].mean = mean_value;

    // 2. RECOMPUTE PREDICTION ERRORS
    recompute_prediction_errors(network, node_idx);

    // 3. UPDATE VOLATILITY LEVEL (unbounded)
    let (precision_vol, mean_vol) = unbounded_volatility_level_update(network, node_idx);
    network.attributes.states[node_idx].precision_vol = precision_vol;
    network.attributes.states[node_idx].mean_vol = mean_vol;
}

fn unbounded_volatility_level_update(network: &Network, node_idx: usize) -> (f64, f64) {
    let s = &network.attributes.states[node_idx];
    let expected_mean_vol = s.expected_mean_vol;
    let expected_precision_vol = s.expected_precision_vol;
    let volatility_coupling = s.volatility_coupling_internal;
    let tonic_volatility = s.tonic_volatility;
    let mean = s.mean;
    let expected_mean = s.expected_mean;
    let precision = s.precision;
    let previous_child_variance = s.current_variance.max(1e-128);

    let numerator = (1.0 / precision) + (mean - expected_mean).powi(2);

    // First quadratic approximation L1
    let x = volatility_coupling * expected_mean_vol + tonic_volatility;
    let w_child = sigmoid(x - previous_child_variance.ln());

    let exp_x_clamped = x.clamp(-80.0, 80.0).exp();
    let delta_child = numerator / (previous_child_variance + exp_x_clamped) - 1.0;

    let pi_l1 = expected_precision_vol
        + 0.5 * volatility_coupling.powi(2) * w_child * (1.0 - w_child);
    let mu_l1 = expected_mean_vol
        + (volatility_coupling * w_child / (2.0 * pi_l1)) * delta_child;

    // Second quadratic approximation L2
    let phi = (previous_child_variance * (2.0 + 3.0_f64.sqrt())).ln();
    let exp_kappa_phi = (volatility_coupling * phi + tonic_volatility).clamp(-80.0, 80.0).exp();
    let w_phi = exp_kappa_phi / (previous_child_variance + exp_kappa_phi);
    let delta_phi = numerator / (previous_child_variance + exp_kappa_phi) - 1.0;

    let pi_l2 = expected_precision_vol
        + 0.5 * volatility_coupling.powi(2) * w_phi * (w_phi + (2.0 * w_phi - 1.0) * delta_phi);
    let mu_hat_phi = ((2.0 * pi_l2 - 1.0) * phi + expected_mean_vol) / (2.0 * pi_l2);
    let mu_l2 = mu_hat_phi + (volatility_coupling * w_phi / (2.0 * pi_l2)) * delta_phi;

    // Full quadratic approximation
    let theta_l = (1.2 * numerator / (previous_child_variance * pi_l1)).sqrt();
    let weighting = b_func(expected_mean_vol, theta_l, 8.0, 0.0, 1.0);

    let posterior_precision = (1.0 - weighting) * pi_l1 + weighting * pi_l2;
    let posterior_mean = (1.0 - weighting) * mu_l1 + weighting * mu_l2;

    (posterior_precision, posterior_mean)
}
