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
///
/// Returns the total precision-weighted prediction error to be added to expected_precision.
fn precision_update_from_children(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = 0.0;

    // --- Value coupling: Σ π̂_child · (κ² · g'(μ)² − g''(μ) · δ_child) ---
    if let Some(ref vc_idxs) = network.edges.get(&node_idx)
        .and_then(|e| e.value_children.clone())
    {
        let coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("value_coupling_children").cloned());

        // g'(μ) and g''(μ) are evaluated at the parent's current mean.
        let parent_mean = *network.attributes.floats
            .get(&node_idx)
            .and_then(|f| f.get("mean"))
            .unwrap_or(&0.0);

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_floats = network.attributes.floats.get(&child_idx)
                .expect("No floats for value child");
            let child_expected_precision = *child_floats.get("expected_precision")
                .expect("child expected_precision not found");
            let kappa = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

            // Find the coupling function stored on the child (indexed by parent
            // position in the child's value_parents list).
            let parent_pos = network.edges.get(&child_idx)
                .and_then(|e| e.value_parents.as_ref())
                .and_then(|vp| vp.iter().position(|&p| p == node_idx));

            let coupling_fn = parent_pos.and_then(|pos| {
                network.attributes.fn_ptrs
                    .get(&child_idx)
                    .and_then(|fp| fp.get("value_coupling_fn_parents"))
                    .and_then(|fns| fns.get(pos).copied())
            });

            // g'(μ)² and g''(μ)·δ — for linear coupling these are 1 and 0.
            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = *child_floats.get("value_prediction_error")
                        .unwrap_or(&0.0);
                    (g_prime.powi(2), g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            // π̂_child · (κ² · g'(μ)² − g''(μ) · δ_child)
            precision_wpe += child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term);
        }
    }

    // --- Volatility coupling: 0.5·(κ·γ)² + (κ·γ)²·Δ - 0.5·κ²·γ·Δ ---
    if let Some(ref volc_idxs) = network.edges.get(&node_idx)
        .and_then(|e| e.volatility_children.clone())
    {
        let vol_coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("volatility_coupling_children").cloned());

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_floats = network.attributes.floats.get(&child_idx)
                .expect("No floats for volatility child");
            let effective_precision = *child_floats.get("effective_precision")
                .expect("child effective_precision not found");
            let volatility_pe = *child_floats.get("volatility_prediction_error")
                .expect("child volatility_prediction_error not found");
            let kappa = vol_coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

            precision_wpe +=
                0.5 * (kappa * effective_precision).powi(2)
                + (kappa * effective_precision).powi(2) * volatility_pe
                - 0.5 * kappa.powi(2) * effective_precision * volatility_pe;
        }
    }

    precision_wpe
}

/// Compute the mean update contribution from value and volatility children.
///
/// `node_precision` is the precision used as denominator (posterior for standard,
/// expected for eHGF).
///
/// Returns the total value + volatility precision-weighted prediction error.
fn mean_update_from_children(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let mut value_pwpe = 0.0;
    let mut volatility_pwpe = 0.0;

    // --- Value coupling mean update ---
    if let Some(ref vc_idxs) = network.edges.get(&node_idx)
        .and_then(|e| e.value_children.clone())
    {
        let coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("value_coupling_children").cloned());

        // The derivative g'(μ_parent) is evaluated at the parent's *current* mean.
        let parent_mean = *network.attributes.floats
            .get(&node_idx)
            .and_then(|f| f.get("mean"))
            .unwrap_or(&0.0);

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_floats = network.attributes.floats.get(&child_idx)
                .expect("No floats for value child");
            let child_expected_precision = *child_floats.get("expected_precision")
                .expect("child expected_precision not found");
            let child_vape = *child_floats.get("value_prediction_error")
                .expect("child value_prediction_error not found");
            let kappa = coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

            // Find the position of node_idx in this child's value_parents list so
            // we can retrieve the correct coupling function stored on the child.
            let parent_pos = network.edges.get(&child_idx)
                .and_then(|e| e.value_parents.as_ref())
                .and_then(|vp| vp.iter().position(|&p| p == node_idx));

            // g'(μ_parent): derivative of the coupling function evaluated at the
            // parent's mean.  Defaults to 1.0 (linear / identity).
            let coupling_fn_prime = parent_pos
                .and_then(|pos| {
                    network.attributes.fn_ptrs
                        .get(&child_idx)
                        .and_then(|fp| fp.get("value_coupling_fn_parents"))
                        .and_then(|fns| fns.get(pos).copied())
                })
                .map(|cf| (cf.df)(parent_mean))
                .unwrap_or(1.0);

            // (κ · g'(μ_parent) · π̂_child / π_node) · δ_child
            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision) * child_vape;
        }
    }

    // --- Volatility coupling mean update ---
    if let Some(ref volc_idxs) = network.edges.get(&node_idx)
        .and_then(|e| e.volatility_children.clone())
    {
        let vol_coupling_strengths = network.attributes.vectors
            .get(&node_idx)
            .and_then(|v| v.get("volatility_coupling_children").cloned());

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_floats = network.attributes.floats.get(&child_idx)
                .expect("No floats for volatility child");
            let effective_precision = *child_floats.get("effective_precision")
                .expect("child effective_precision not found");
            let volatility_pe = *child_floats.get("volatility_prediction_error")
                .expect("child volatility_prediction_error not found");
            let kappa = vol_coupling_strengths.as_ref().map(|cs| cs[i]).unwrap_or(1.0);

            volatility_pwpe +=
                (kappa * effective_precision * volatility_pe) / (2.0 * node_precision);
        }
    }

    value_pwpe + volatility_pwpe
}

// =============================================================================
// Standard posterior update
// =============================================================================

/// Standard posterior update from a continuous state node
///
/// 1. Update posterior precision
/// 2. Update posterior mean using the **posterior** precision
///
/// This is used for nodes that have only value children (no volatility children).
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused in this update).
pub fn posterior_update_continuous_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {

    let expected_precision = *network.attributes.floats.get(&node_idx)
        .expect("No floats for node")
        .get("expected_precision")
        .expect("expected_precision not found");
    let expected_mean = *network.attributes.floats.get(&node_idx)
        .expect("No floats for node")
        .get("expected_mean")
        .expect("expected_mean not found");

    // 1. Precision update
    let precision_wpe = precision_update_from_children(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe).max(1e-128);

    // 2. Mean update (using posterior precision)
    let mean_wpe = mean_update_from_children(network, node_idx, posterior_precision);
    let posterior_mean = expected_mean + mean_wpe;

    // Store results
    let floats_mut = network.attributes.floats.get_mut(&node_idx)
        .expect("No floats for node");
    floats_mut.insert(String::from("precision"), posterior_precision);
    floats_mut.insert(String::from("mean"), posterior_mean);
}

// =============================================================================
// eHGF posterior update
// =============================================================================

/// eHGF posterior update from a continuous state node
///
/// 1. Update posterior mean using the **expected** precision (anticipatory)
/// 2. Update posterior precision
///
/// This is the default for nodes that have volatility children.
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused in this update).
pub fn posterior_update_continuous_state_node_ehgf(network: &mut Network, node_idx: usize, _time_step: f64) {

    let expected_precision = *network.attributes.floats.get(&node_idx)
        .expect("No floats for node")
        .get("expected_precision")
        .expect("expected_precision not found");
    let expected_mean = *network.attributes.floats.get(&node_idx)
        .expect("No floats for node")
        .get("expected_mean")
        .expect("expected_mean not found");

    // 1. Mean update first (using expected precision as approximation)
    let mean_wpe = mean_update_from_children(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;

    // Store mean immediately (precision update may read updated child means)
    network.attributes.floats.get_mut(&node_idx)
        .expect("No floats for node")
        .insert(String::from("mean"), posterior_mean);

    // 2. Precision update
    let precision_wpe = precision_update_from_children(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe).max(1e-128);

    // Store precision
    network.attributes.floats.get_mut(&node_idx)
        .expect("No floats for node")
        .insert(String::from("precision"), posterior_precision);
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

/// Posterior update from a continuous state node using unbounded quadratic approximation
///
/// This function updates the posterior mean and precision of a continuous node
/// that has a volatility child, using two quadratic approximations (L1 and L2)
/// weighted by a smoothed rectangular function.
///
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// * `_time_step` - The time step (unused in this update).
///
/// # Returns
/// * `network` - The network after message passing.
pub fn posterior_update_continuous_state_node_unbounded(network: &mut Network, node_idx: usize, _time_step: f64) {

    // Get the first volatility child index
    let volatility_child_idx = network.edges.get(&node_idx)
        .expect("No edges found for node")
        .volatility_children.as_ref()
        .expect("No volatility children found")[0];

    // Read attributes from the node
    let node_floats = network.attributes.floats.get(&node_idx)
        .expect("No floats attributes found for node");
    let expected_mean = *node_floats.get("expected_mean").expect("expected_mean not found");
    let expected_precision = *node_floats.get("expected_precision")
        .expect("expected_precision not found");

    // Read volatility coupling strength from vector attributes
    let vol_coupling = network.attributes.vectors
        .get(&node_idx)
        .and_then(|v| v.get("volatility_coupling_children"))
        .map(|cs| cs[0])
        .unwrap_or(1.0);

    // Read attributes from the volatility child
    let child_floats = network.attributes.floats.get(&volatility_child_idx)
        .expect("No floats attributes found for volatility child");
    let child_mean = *child_floats.get("mean").expect("child mean not found");
    let child_precision = *child_floats.get("precision").expect("child precision not found");
    let child_expected_mean = *child_floats.get("expected_mean")
        .expect("child expected_mean not found");
    let child_tonic_volatility = *child_floats.get("tonic_volatility")
        .expect("child tonic_volatility not found");

    // Recover the variance of the child node at the previous time step
    let previous_child_variance = (*child_floats.get("current_variance")
        .expect("child current_variance not found"))
        .max(1e-128);

    // ---------------------------------------------------------------
    // First quadratic approximation L1
    // ---------------------------------------------------------------

    // x = κ · μ̂ + ω_child
    let x = vol_coupling * expected_mean + child_tonic_volatility;

    // Numerically stable form: w = sigmoid(x - log(v))
    let w_child = sigmoid(x - previous_child_variance.ln());

    // δ_child = ((1/π_child) + (μ_child - μ̂_child)²) / (v + exp(x)) - 1
    let child_prediction_error_sq = (child_mean - child_expected_mean).powi(2);
    let numerator = (1.0 / child_precision) + child_prediction_error_sq;
    let exp_x_clamped = x.clamp(-80.0, 80.0).exp();
    let delta_child = numerator / (previous_child_variance + exp_x_clamped) - 1.0;

    // π_L1 = π̂ + 0.5 · κ² · w · (1 - w)
    let pi_l1 = expected_precision + 0.5 * vol_coupling.powi(2) * w_child * (1.0 - w_child);

    // μ_L1 = μ̂ + (κ · w / (2 · π_L1)) · δ
    let mu_l1 = expected_mean + (vol_coupling * w_child / (2.0 * pi_l1)) * delta_child;

    // ---------------------------------------------------------------
    // Second quadratic approximation L2
    // ---------------------------------------------------------------

    // φ = log(v · (2 + √3))
    let phi = (previous_child_variance * (2.0 + 3.0_f64.sqrt())).ln();

    // w_φ = exp(κ·φ + ω) / (v + exp(κ·φ + ω))
    let exp_kappa_phi = (vol_coupling * phi + child_tonic_volatility).clamp(-80.0, 80.0).exp();
    let w_phi = exp_kappa_phi / (previous_child_variance + exp_kappa_phi);

    // δ_φ = numerator / (v + exp(κ·φ + ω)) - 1
    let delta_phi = numerator / (previous_child_variance + exp_kappa_phi) - 1.0;

    // π_L2 = π̂ + 0.5 · κ² · w_φ · (w_φ + (2·w_φ - 1) · δ_φ)
    let pi_l2 = expected_precision
        + 0.5 * vol_coupling.powi(2) * w_phi * (w_phi + (2.0 * w_phi - 1.0) * delta_phi);

    // μ̂_φ = ((2·π_L2 - 1)·φ + μ̂) / (2·π_L2)
    let mu_hat_phi = ((2.0 * pi_l2 - 1.0) * phi + expected_mean) / (2.0 * pi_l2);

    // μ_L2 = μ̂_φ + (κ · w_φ / (2 · π_L2)) · δ_φ
    let mu_l2 = mu_hat_phi + (vol_coupling * w_phi / (2.0 * pi_l2)) * delta_phi;

    // ---------------------------------------------------------------
    // Full quadratic approximation
    // ---------------------------------------------------------------

    // θ_l = sqrt(1.2 · numerator / (v · π_L1))
    let theta_l = (1.2 * numerator / (previous_child_variance * pi_l1)).sqrt();

    // Weighting of the two approximations using the smoothed rectangular function b
    let weighting = b(expected_mean, theta_l, 8.0, 0.0, 1.0);

    // Posterior precision and mean as weighted combination
    let posterior_precision = (1.0 - weighting) * pi_l1 + weighting * pi_l2;
    let posterior_mean = (1.0 - weighting) * mu_l1 + weighting * mu_l2;

    // Store the results
    let floats_mut = network.attributes.floats.get_mut(&node_idx)
        .expect("No floats attributes found for node");
    floats_mut.insert(String::from("precision"), posterior_precision);
    floats_mut.insert(String::from("mean"), posterior_mean);
}