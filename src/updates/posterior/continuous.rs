use crate::model::Network;

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Parametrised sigmoid: sigmoid(phi * (x - theta))
fn s(x: f64, theta: f64, phi: f64) -> f64 {
    sigmoid(phi * (x - theta))
}

/// Smoothed rectangular weighting function b
fn b(x: f64, theta_l: f64, phi_l: f64, theta_r: f64, phi_r: f64) -> f64 {
    s(x, theta_l, phi_l) * (1.0 - s(x, theta_r, phi_r))
}

// =============================================================================
// Shared building blocks for precision and mean updates
// =============================================================================

/// Compute the precision update contribution from value and volatility children.
fn precision_update_from_children(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = 0.0;

    // --- Value coupling ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let parent_pos = network.edges[child_idx].value_parents.as_ref()
                .and_then(|vp| vp.iter().position(|&p| p == node_idx));

            let coupling_fn = parent_pos.and_then(|pos| {
                network.attributes.fn_ptrs[child_idx].value_coupling_fn_parents.get(pos).copied()
            });

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            precision_wpe += (child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term)) * observed;
        }
    }

    // --- Volatility coupling ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths = &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            precision_wpe += (
                0.5 * (kappa * effective_precision).powi(2)
                + (kappa * effective_precision).powi(2) * volatility_pe
                - 0.5 * kappa.powi(2) * effective_precision * volatility_pe
            ) * observed;
        }
    }

    precision_wpe
}

/// Compute the mean update contribution from value and volatility children.
fn mean_update_from_children(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let mut value_pwpe = 0.0;
    let mut volatility_pwpe = 0.0;

    // --- Value coupling mean update ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

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
                .map(|cf| (cf.df)(parent_mean))
                .unwrap_or(1.0);

            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision) * child_vape;
        }
    }

    // --- Volatility coupling mean update ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths = &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            volatility_pwpe +=
                (kappa * effective_precision * volatility_pe) / (2.0 * node_precision) * observed;
        }
    }

    value_pwpe + volatility_pwpe
}

// =============================================================================
// Standard posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    let precision_wpe = precision_update_from_children(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe).max(1e-128);

    let mean_wpe = mean_update_from_children(network, node_idx, posterior_precision);
    let posterior_mean = expected_mean + mean_wpe;

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

// =============================================================================
// eHGF posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node_ehgf(network: &mut Network, node_idx: usize, _time_step: f64) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;

    let mean_wpe = mean_update_from_children(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    let precision_wpe = precision_update_from_children(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe).max(1e-128);
    network.attributes.states[node_idx].precision = posterior_precision;
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node_unbounded(network: &mut Network, node_idx: usize, _time_step: f64) {
    let volatility_child_idx = network.edges[node_idx]
        .volatility_children.as_ref()
        .expect("No volatility children found")[0];

    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let expected_precision = network.attributes.states[node_idx].expected_precision;

    let vol_coupling = network.attributes.vectors[node_idx]
        .volatility_coupling_children.get(0).copied().unwrap_or(1.0);

    let child_state = network.attributes.states[volatility_child_idx];
    let child_mean = child_state.mean;
    let child_precision = child_state.precision;
    let child_expected_mean = child_state.expected_mean;
    let child_tonic_volatility = child_state.tonic_volatility;
    let previous_child_variance = child_state.current_variance.max(1e-128);

    // First quadratic approximation L1
    let x = vol_coupling * expected_mean + child_tonic_volatility;
    let w_child = sigmoid(x - previous_child_variance.ln());

    let child_prediction_error_sq = (child_mean - child_expected_mean).powi(2);
    let numerator = (1.0 / child_precision) + child_prediction_error_sq;
    let exp_x_clamped = x.clamp(-80.0, 80.0).exp();
    let delta_child = numerator / (previous_child_variance + exp_x_clamped) - 1.0;

    let pi_l1 = expected_precision + 0.5 * vol_coupling.powi(2) * w_child * (1.0 - w_child);
    let mu_l1 = expected_mean + (vol_coupling * w_child / (2.0 * pi_l1)) * delta_child;

    // Second quadratic approximation L2
    let phi = (previous_child_variance * (2.0 + 3.0_f64.sqrt())).ln();
    let exp_kappa_phi = (vol_coupling * phi + child_tonic_volatility).clamp(-80.0, 80.0).exp();
    let w_phi = exp_kappa_phi / (previous_child_variance + exp_kappa_phi);
    let delta_phi = numerator / (previous_child_variance + exp_kappa_phi) - 1.0;

    let pi_l2 = expected_precision
        + 0.5 * vol_coupling.powi(2) * w_phi * (w_phi + (2.0 * w_phi - 1.0) * delta_phi);
    let mu_hat_phi = ((2.0 * pi_l2 - 1.0) * phi + expected_mean) / (2.0 * pi_l2);
    let mu_l2 = mu_hat_phi + (vol_coupling * w_phi / (2.0 * pi_l2)) * delta_phi;

    // Full quadratic approximation
    let theta_l = (1.2 * numerator / (previous_child_variance * pi_l1)).sqrt();
    let weighting = b(expected_mean, theta_l, 8.0, 0.0, 1.0);

    let posterior_precision = (1.0 - weighting) * pi_l1 + weighting * pi_l2;
    let posterior_mean = (1.0 - weighting) * mu_l1 + weighting * mu_l2;

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}
