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
// Shared building blocks for the value level (external facing)
// =============================================================================

/// Update the precision of the value level using value children's PEs.
fn precision_update_value_level(network: &Network, node_idx: usize) -> f64 {
    let expected_precision = *network.attributes.floats.get(&node_idx).unwrap()
        .get("expected_precision").expect("expected_precision not found");

    let mut posterior_precision = expected_precision;

    // Add contributions from value children
    if let Some(ref vc_idxs) = network.edges.get(&node_idx)
        .and_then(|e| e.value_children.clone())
    {
        let coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("value_coupling_children").cloned());

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_expected_precision = *network.attributes.floats.get(&child_idx)
                .expect("No floats for value child")
                .get("expected_precision")
                .expect("child expected_precision not found");
            let kappa = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);
            posterior_precision += kappa.powi(2) * child_expected_precision;
        }
    }

    posterior_precision
}

/// Update the mean of the value level using value children's PEs.
fn mean_update_value_level(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let expected_mean = *network.attributes.floats.get(&node_idx).unwrap()
        .get("expected_mean").expect("expected_mean not found");

    let mut value_pwpe = 0.0;

    // Add contributions from value children
    if let Some(ref vc_idxs) = network.edges.get(&node_idx)
        .and_then(|e| e.value_children.clone())
    {
        let coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("value_coupling_children").cloned());

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_floats = network.attributes.floats.get(&child_idx)
                .expect("No floats for value child");
            let child_expected_precision = *child_floats.get("expected_precision")
                .expect("child expected_precision not found");
            let child_vape = *child_floats.get("value_prediction_error")
                .expect("child value_prediction_error not found");
            let kappa = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

            value_pwpe += (kappa * child_expected_precision / node_precision) * child_vape;
        }
    }

    expected_mean + value_pwpe
}

// =============================================================================
// Shared building blocks for the volatility level (internal)
// =============================================================================

/// Standard precision update for the volatility level.
///
/// Uses the value level's volatility prediction error (internal coupling):
///     π_vol = π̂_vol + 0.5·(κ·γ)² + (κ·γ)²·Δ - 0.5·κ²·γ·Δ
fn precision_update_volatility_level(network: &Network, node_idx: usize) -> f64 {
    let floats = network.attributes.floats.get(&node_idx).unwrap();
    let expected_precision_vol = *floats.get("expected_precision_vol")
        .expect("expected_precision_vol not found");
    let volatility_pe = *floats.get("volatility_prediction_error")
        .expect("volatility_prediction_error not found");
    let effective_precision = *floats.get("effective_precision")
        .expect("effective_precision not found");
    let volatility_coupling = *floats.get("volatility_coupling_internal")
        .expect("volatility_coupling_internal not found");

    expected_precision_vol
        + 0.5 * (volatility_coupling * effective_precision).powi(2)
        + (volatility_coupling * effective_precision).powi(2) * volatility_pe
        - 0.5 * volatility_coupling.powi(2) * effective_precision * volatility_pe
}

/// Standard mean update for the volatility level.
///
/// Uses the value level's volatility prediction error (internal coupling):
///     μ_vol = μ̂_vol + (κ·γ·Δ) / (2·π_vol) · observed
fn mean_update_volatility_level(network: &Network, node_idx: usize, node_precision_vol: f64) -> f64 {
    let floats = network.attributes.floats.get(&node_idx).unwrap();
    let expected_mean_vol = *floats.get("expected_mean_vol")
        .expect("expected_mean_vol not found");
    let volatility_pe = *floats.get("volatility_prediction_error")
        .expect("volatility_prediction_error not found");
    let effective_precision = *floats.get("effective_precision")
        .expect("effective_precision not found");
    let volatility_coupling = *floats.get("volatility_coupling_internal")
        .expect("volatility_coupling_internal not found");
    let observed = *floats.get("observed").unwrap_or(&1.0);

    let precision_weighted_pe =
        (volatility_coupling * effective_precision * volatility_pe)
        / (2.0 * node_precision_vol);

    expected_mean_vol + precision_weighted_pe * observed
}

// =============================================================================
// Internal PE recomputation (between value and volatility level updates)
// =============================================================================

/// Recompute value and volatility prediction errors after the value level has been
/// updated. This is needed because the volatility PE depends on the updated value level.
fn recompute_prediction_errors(network: &mut Network, node_idx: usize) {
    let floats = network.attributes.floats.get(&node_idx).unwrap();
    let mean = *floats.get("mean").expect("mean not found");
    let expected_mean = *floats.get("expected_mean").expect("expected_mean not found");
    let precision = *floats.get("precision").expect("precision not found");
    let expected_precision = *floats.get("expected_precision")
        .expect("expected_precision not found");

    let n_value_parents = network.edges.get(&node_idx)
        .and_then(|e| e.value_parents.as_ref())
        .map(|vp| vp.len());

    let mut value_pe = mean - expected_mean;
    if let Some(n) = n_value_parents {
        value_pe /= n as f64;
    }

    let volatility_pe =
        (expected_precision / precision)
        + expected_precision * value_pe.powi(2)
        - 1.0;

    let floats_mut = network.attributes.floats.get_mut(&node_idx).unwrap();
    floats_mut.insert(String::from("value_prediction_error"), value_pe);
    floats_mut.insert(String::from("volatility_prediction_error"), volatility_pe);
}

// =============================================================================
// Standard posterior update
// =============================================================================

/// Standard posterior update for a volatile state node.
///
/// 1. Update value level: precision first, then mean (standard order)
/// 2. Recompute prediction errors
/// 3. Update volatility level: precision first, then mean (standard order)
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused).
pub fn posterior_update_volatile_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    // 1. UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("precision"), precision_value);

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("mean"), mean_value);

    // 2. RECOMPUTE PREDICTION ERRORS
    recompute_prediction_errors(network, node_idx);

    // 3. UPDATE VOLATILITY LEVEL (standard: precision first, then mean)
    let precision_vol = precision_update_volatility_level(network, node_idx);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("precision_vol"), precision_vol);

    let mean_vol = mean_update_volatility_level(network, node_idx, precision_vol);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("mean_vol"), mean_vol);
}

// =============================================================================
// eHGF posterior update
// =============================================================================

/// eHGF posterior update for a volatile state node.
///
/// 1. Update value level: precision first, then mean (standard order)
/// 2. Recompute prediction errors
/// 3. Update volatility level: mean first (using expected precision), then precision
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused).
pub fn posterior_update_volatile_state_node_ehgf(network: &mut Network, node_idx: usize, _time_step: f64) {
    // 1. UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("precision"), precision_value);

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("mean"), mean_value);

    // 2. RECOMPUTE PREDICTION ERRORS
    recompute_prediction_errors(network, node_idx);

    // 3. UPDATE VOLATILITY LEVEL (eHGF: mean first using expected precision, then precision)
    let expected_precision_vol = *network.attributes.floats.get(&node_idx).unwrap()
        .get("expected_precision_vol").expect("expected_precision_vol not found");

    let mean_vol = mean_update_volatility_level(network, node_idx, expected_precision_vol);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("mean_vol"), mean_vol);

    let precision_vol = precision_update_volatility_level(network, node_idx);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("precision_vol"), precision_vol);
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

/// Posterior update for a volatile state node using unbounded quadratic approximation.
///
/// 1. Update value level: precision first, then mean (standard order)
/// 2. Recompute prediction errors
/// 3. Update volatility level using the unbounded quadratic approximation
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused).
pub fn posterior_update_volatile_state_node_unbounded(network: &mut Network, node_idx: usize, _time_step: f64) {
    // 1. UPDATE VALUE LEVEL
    let precision_value = precision_update_value_level(network, node_idx);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("precision"), precision_value);

    let mean_value = mean_update_value_level(network, node_idx, precision_value);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("mean"), mean_value);

    // 2. RECOMPUTE PREDICTION ERRORS
    recompute_prediction_errors(network, node_idx);

    // 3. UPDATE VOLATILITY LEVEL (unbounded quadratic approximation)
    let (precision_vol, mean_vol) = unbounded_volatility_level_update(network, node_idx);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("precision_vol"), precision_vol);
    network.attributes.floats.get_mut(&node_idx).unwrap()
        .insert(String::from("mean_vol"), mean_vol);
}

/// Unbounded quadratic approximation for the internal volatility level.
///
/// Adapts the continuous unbounded update to operate on the implicit volatility
/// level within a volatile node. The "volatility child" is the value level of
/// the same node, and the "volatility parent" is the implicit volatility level.
fn unbounded_volatility_level_update(network: &Network, node_idx: usize) -> (f64, f64) {
    let floats = network.attributes.floats.get(&node_idx).unwrap();

    let expected_mean_vol = *floats.get("expected_mean_vol")
        .expect("expected_mean_vol not found");
    let expected_precision_vol = *floats.get("expected_precision_vol")
        .expect("expected_precision_vol not found");
    let volatility_coupling = *floats.get("volatility_coupling_internal")
        .expect("volatility_coupling_internal not found");
    let tonic_volatility = *floats.get("tonic_volatility")
        .expect("tonic_volatility not found");
    let mean = *floats.get("mean").expect("mean not found");
    let expected_mean = *floats.get("expected_mean").expect("expected_mean not found");
    let precision = *floats.get("precision").expect("precision not found");
    let previous_child_variance = (*floats.get("current_variance")
        .expect("current_variance not found"))
        .max(1e-128);

    let numerator = (1.0 / precision) + (mean - expected_mean).powi(2);

    // -----------------------------------------------------------------
    // First quadratic approximation L1
    // -----------------------------------------------------------------
    let x = volatility_coupling * expected_mean_vol + tonic_volatility;
    let w_child = sigmoid(x - previous_child_variance.ln());

    let exp_x_clamped = x.clamp(-80.0, 80.0).exp();
    let delta_child = numerator / (previous_child_variance + exp_x_clamped) - 1.0;

    let pi_l1 = expected_precision_vol
        + 0.5 * volatility_coupling.powi(2) * w_child * (1.0 - w_child);

    let mu_l1 = expected_mean_vol
        + (volatility_coupling * w_child / (2.0 * pi_l1)) * delta_child;

    // -----------------------------------------------------------------
    // Second quadratic approximation L2
    // -----------------------------------------------------------------
    let phi = (previous_child_variance * (2.0 + 3.0_f64.sqrt())).ln();

    let exp_kappa_phi = (volatility_coupling * phi + tonic_volatility).clamp(-80.0, 80.0).exp();
    let w_phi = exp_kappa_phi / (previous_child_variance + exp_kappa_phi);

    let delta_phi = numerator / (previous_child_variance + exp_kappa_phi) - 1.0;

    let pi_l2 = expected_precision_vol
        + 0.5 * volatility_coupling.powi(2) * w_phi * (w_phi + (2.0 * w_phi - 1.0) * delta_phi);

    let mu_hat_phi = ((2.0 * pi_l2 - 1.0) * phi + expected_mean_vol) / (2.0 * pi_l2);

    let mu_l2 = mu_hat_phi + (volatility_coupling * w_phi / (2.0 * pi_l2)) * delta_phi;

    // -----------------------------------------------------------------
    // Full quadratic approximation
    // -----------------------------------------------------------------
    let theta_l = (1.2 * numerator / (previous_child_variance * pi_l1)).sqrt();

    let weighting = b_func(expected_mean_vol, theta_l, 8.0, 0.0, 1.0);

    let posterior_precision = (1.0 - weighting) * pi_l1 + weighting * pi_l2;
    let posterior_mean = (1.0 - weighting) * mu_l1 + weighting * mu_l2;

    (posterior_precision, posterior_mean)
}
