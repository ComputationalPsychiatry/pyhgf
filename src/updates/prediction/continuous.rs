use crate::model::Network;

/// Prediction step for a continuous state node.
///
/// Computes the predicted mean μ̂, the conditional predicted precision π̂
/// (`conditional_expected_precision`), the marginal predicted precision π̃
/// (`expected_precision`), and the effective precision γ.
///
/// * π̂ = 1 / (1/π + Ω) — own variance plus volatility, no parent-uncertainty
///   bleed-through. Used by the parent's Schur-complement posterior-step
///   correction.
/// * π̃ = 1 / (1/π̂ + Σ_b (Δt · α · g'(μ̂_b))² / π̃_b) — inverse marginal
///   predictive variance, adding the first-order Laplace contribution from each
///   value parent (using the parent's marginal predicted precision π̃_b).
/// * Ω includes the moment-generating-function correction κ²/(2 π̂_vol) inside
///   the log-volatility exponent for each volatility parent.
pub fn prediction_continuous_state_node(network: &mut Network, node_idx: usize, time_step: f64) {
    // Copy own scalar state (f64 is Copy — no borrow held)
    let mean = network.attributes.states[node_idx].mean;
    let tonic_drift = network.attributes.states[node_idx].tonic_drift;
    let autoconnection_strength = network.attributes.states[node_idx].autoconnection_strength;
    let precision = network.attributes.states[node_idx].precision;
    let tonic_volatility = network.attributes.states[node_idx].tonic_volatility;

    // -------------------------------------------------------
    // 1. Predict the mean: μ̂ = λ · μ + Δt · driftrate.
    //    Also accumulate the piHGF Laplace value-coupling variance
    //        Σ_b (Δt · α · g'(μ̂_b))² / π̃_b
    //    using each parent's marginal predicted precision π̃_b
    //    (= `parent.expected_precision`). The constant-bias parent has infinite
    //    precision and contributes zero.
    // -------------------------------------------------------
    let mut driftrate = tonic_drift;
    let mut value_coupling_variance = 0.0_f64;

    if let Some(ref vp_idxs) = network.edges[node_idx].value_parents {
        let couplings = &network.attributes.vectors[node_idx].value_coupling_parents;

        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let parent_expected_precision =
                network.attributes.states[parent_idx].expected_precision;
            let psi = couplings.get(i).copied().unwrap_or(1.0);
            let (parent_value, g_prime) = match network.attributes.fn_ptrs[parent_idx].coupling_fn {
                Some(cf) => ((cf.f)(parent_expected_mean), (cf.df)(parent_expected_mean)),
                None => (parent_expected_mean, 1.0),
            };
            driftrate += psi * parent_value;
            // First-order Taylor expansion of g around μ̂_b yields a
            // (Δt · α · g'(μ̂_b))² / π̃_b contribution to the marginal
            // predictive variance of x_a. Vanishes as π̃_b → ∞.
            let coeff = time_step * psi * g_prime;
            value_coupling_variance += coeff * coeff / parent_expected_precision;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // -------------------------------------------------------
    // 2. Predict the two precisions:
    //        π̂ = 1 / (1/π + Ω)
    //        π̃ = 1 / (1/π̂ + value-coupling variance)
    //    Ω = Δt · exp(ω + Σ_j κ_j μ_j + Σ_j κ_j²/(2 π̂_vol_j)); the MGF correction
    //    κ²/(2 π̂_vol) marginalises over each volatility parent's Gaussian rather
    //    than collapsing it to a point estimate.
    // -------------------------------------------------------
    let mut total_volatility = tonic_volatility;

    if let Some(ref vol_parent_idxs) = network.edges[node_idx].volatility_parents {
        let vol_couplings = &network.attributes.vectors[node_idx].volatility_coupling_parents;

        for (i, &parent_idx) in vol_parent_idxs.iter().enumerate() {
            let parent_mean = network.attributes.states[parent_idx].mean;
            let parent_expected_precision =
                network.attributes.states[parent_idx].expected_precision;
            let kappa = vol_couplings.get(i).copied().unwrap_or(1.0);
            total_volatility += kappa * parent_mean;
            total_volatility += (kappa * kappa) / (2.0 * parent_expected_precision);
        }
    }

    let pv_raw = time_step * total_volatility.exp();
    let predicted_volatility = if pv_raw > 1e-128 { pv_raw } else { f64::NAN };
    // Conditional predicted precision π̂_a — own variance + volatility only,
    // WITHOUT the parent-uncertainty value-coupling term. The parent's posterior-step
    // Schur complement acts on this; the marginal would double-count parent uncertainty.
    let conditional_expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);
    let expected_precision =
        1.0 / ((1.0 / precision) + predicted_volatility + value_coupling_variance);
    // Effective precision γ — only the volatility-driven part enters γ, since
    // γ is consumed by the volatility-coupling posterior update.
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
        state.conditional_expected_precision = conditional_expected_precision;
    } else {
        // Leaf without random walk: conditional == marginal == prior precision.
        state.conditional_expected_precision = precision;
    }
}

/// Mean-field prediction step for a continuous state node.
pub fn prediction_continuous_state_node_mean_field(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let mean = network.attributes.states[node_idx].mean;
    let tonic_drift = network.attributes.states[node_idx].tonic_drift;
    let autoconnection_strength = network.attributes.states[node_idx].autoconnection_strength;
    let precision = network.attributes.states[node_idx].precision;
    let tonic_volatility = network.attributes.states[node_idx].tonic_volatility;

    let mut driftrate = tonic_drift;

    if let Some(ref vp_idxs) = network.edges[node_idx].value_parents {
        let couplings = &network.attributes.vectors[node_idx].value_coupling_parents;
        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let psi = couplings.get(i).copied().unwrap_or(1.0);
            let parent_value = match network.attributes.fn_ptrs[parent_idx].coupling_fn {
                Some(cf) => (cf.f)(parent_expected_mean),
                None => parent_expected_mean,
            };
            driftrate += psi * parent_value;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    let mut total_volatility = tonic_volatility;
    if let Some(ref vol_parent_idxs) = network.edges[node_idx].volatility_parents {
        let vol_couplings = &network.attributes.vectors[node_idx].volatility_coupling_parents;
        for (i, &parent_idx) in vol_parent_idxs.iter().enumerate() {
            let parent_mean = network.attributes.states[parent_idx].mean;
            let kappa = vol_couplings.get(i).copied().unwrap_or(1.0);
            total_volatility += kappa * parent_mean;
        }
    }

    let pv_raw = time_step * total_volatility.exp();
    let predicted_volatility = if pv_raw > 1e-128 { pv_raw } else { f64::NAN };
    let expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;

    let is_input = network.edges[node_idx].value_children.is_none()
        && network.edges[node_idx].volatility_children.is_none();
    let has_volatility_parents = network.edges[node_idx].volatility_parents.is_some();

    let state = &mut network.attributes.states[node_idx];
    state.current_variance = 1.0 / precision;
    state.expected_mean = expected_mean;
    state.effective_precision = effective_precision;

    if !(is_input && !has_volatility_parents) {
        state.expected_precision = expected_precision;
        state.conditional_expected_precision = expected_precision;
    } else {
        state.conditional_expected_precision = precision;
    }
}
