use crate::model::Network;

/// Prediction step for a volatile state node.
///
/// Predicts both the implicit volatility level (mean_vol, expected_precision_vol)
/// and the externally-facing value level. Two predicted precisions of the value
/// level are stored:
///
/// * π̂ = 1 / (1/π + Ω) — `conditional_expected_precision`. Own variance plus
///   volatility chain, without the parent-uncertainty bleed-through term. Used by
///   the parent's Schur-complement posterior-step correction.
/// * π̃ = 1 / (1/π̂ + Σ_b (Δt · α · g'(μ̂_b))² / π̃_b) — `expected_precision`,
///   the inverse marginal predictive variance, adding the first-order Laplace
///   contribution from each value parent (using the parent's marginal predicted
///   precision π̃_b).
///
/// Ω = Δt · exp(ω + κ · μ_vol + κ² / (2 π̂_vol)); the MGF correction
/// κ² / (2 π̂_vol) marginalises over the implicit volatility level's Gaussian
/// rather than collapsing it to a point estimate.
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

    // --- 2a. Predict mean (including value parents if any). Also accumulate the
    //         piHGF Laplace value-coupling variance
    //             Σ_b (Δt · α · g'(μ̂_b))² / π̃_b
    //         using each parent's marginal predicted precision π̃_b
    //         (= `parent.expected_precision`).
    let mut driftrate = 0.0;
    let mut value_coupling_variance = 0.0_f64;
    if let Some(ref vp_idxs) = network.edges[node_idx].value_parents {
        let couplings = &network.attributes.vectors[node_idx].value_coupling_parents;

        for (i, &parent_idx) in vp_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let parent_expected_precision =
                network.attributes.states[parent_idx].expected_precision;
            let value_coupling_parent = couplings.get(i).copied().unwrap_or(1.0);
            let (parent_value, g_prime) = match network.attributes.fn_ptrs[parent_idx].coupling_fn
            {
                Some(cf) => ((cf.f)(parent_expected_mean), (cf.df)(parent_expected_mean)),
                None => (parent_expected_mean, 1.0),
            };
            driftrate += value_coupling_parent * parent_value;
            let coeff = time_step * value_coupling_parent * g_prime;
            value_coupling_variance += coeff * coeff / parent_expected_precision;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // --- 2b. Predict precision (depends on volatility level). The implicit
    //         volatility level enters the conditional variance through
    //         exp(κ · x_vol); marginalising over the volatility level's
    //         Gaussian yields the closed-form moment-generating-function
    //         correction κ² / (2 · π̂_vol) inside the log-volatility exponent.
    let total_volatility = tonic_volatility
        + volatility_coupling_internal * mean_vol
        + (volatility_coupling_internal * volatility_coupling_internal)
            / (2.0 * expected_precision_vol);
    let pv_raw = time_step * total_volatility.exp();
    let predicted_volatility = if pv_raw > 1e-128 { pv_raw } else { f64::NAN };
    // Conditional predicted precision π̂_a — precision of x_a given its value
    // parents (own variance + volatility only), WITHOUT the parent-uncertainty
    // value-coupling term. This is what the parent's posterior-step Schur
    // complement acts on; substituting the marginal would double-count parent
    // uncertainty.
    let conditional_expected_precision = 1.0 / ((1.0 / precision) + predicted_volatility);
    let expected_precision =
        1.0 / ((1.0 / precision) + predicted_volatility + value_coupling_variance);
    // Effective precision γ — only the volatility-driven part enters γ, since
    // γ is consumed by the volatility-coupling posterior update.
    let effective_precision = predicted_volatility * expected_precision;

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
        // A leaf has no volatility random walk, so the conditional and marginal
        // predicted precisions coincide with the prior precision.
        state.conditional_expected_precision = precision;
        state.effective_precision = 0.0;
    } else {
        state.expected_precision = expected_precision;
        state.conditional_expected_precision = conditional_expected_precision;
        state.effective_precision = effective_precision;
    }
}
