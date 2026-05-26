use crate::model::Network;

/// Principal branch of the Lambert W function for z >= 0.
/// Solves w * exp(w) = z via 6 Halley iterations.
fn lambert_w0(z: f64) -> f64 {
    let mut w = (z + 1.0).ln();
    for _ in 0..6 {
        let ew = w.exp();
        let f = w * ew - z;
        let f1 = (w + 1.0) * ew;
        let f2 = (w + 2.0) * ew;
        w -= (2.0 * f * f1) / (2.0 * f1 * f1 - f * f2);
    }
    w
}

/// Compute value and volatility prediction errors for a volatile state node.
fn compute_volatile_prediction_errors(network: &mut Network, node_idx: usize) {
    let n_volatility_parents = network.edges[node_idx].volatility_parents.as_ref().map(|vp| vp.len());

    let mean = network.attributes.states[node_idx].mean;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let precision = network.attributes.states[node_idx].precision;
    let expected_precision = network.attributes.states[node_idx].expected_precision;

    // Value prediction error: δ = μ - μ̂
    let value_prediction_error = mean - expected_mean;
    
    // Volatility prediction error (internal coupling, no division)
    let mut volatility_prediction_error =
    (expected_precision / precision)
    + expected_precision * (mean - expected_mean).powi(2)
    - 1.0;
    if let Some(n) = n_volatility_parents {
        volatility_prediction_error /= n as f64;
    }

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_prediction_error;
    state.volatility_prediction_error = volatility_prediction_error;
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
// Standard: prediction error + volatility level posterior update
// =============================================================================

pub fn prediction_error_volatile_state_node(network: &mut Network, node_idx: usize, _time_step: f64) {
    compute_volatile_prediction_errors(network, node_idx);

    let precision_vol =
        precision_update_volatility_level(network, node_idx).min(network.max_posterior_precision);
    network.attributes.states[node_idx].precision_vol = precision_vol;

    let mean_vol = mean_update_volatility_level(network, node_idx, precision_vol);
    network.attributes.states[node_idx].mean_vol = mean_vol;
}

// =============================================================================
// eHGF: prediction error + volatility level posterior update
// =============================================================================

pub fn prediction_error_volatile_state_node_ehgf(network: &mut Network, node_idx: usize, _time_step: f64) {
    compute_volatile_prediction_errors(network, node_idx);

    let expected_precision_vol = network.attributes.states[node_idx].expected_precision_vol;

    let mean_vol = mean_update_volatility_level(network, node_idx, expected_precision_vol);
    network.attributes.states[node_idx].mean_vol = mean_vol;

    let precision_vol =
        precision_update_volatility_level(network, node_idx).min(network.max_posterior_precision);
    network.attributes.states[node_idx].precision_vol = precision_vol;
}

// =============================================================================
// Unbounded: prediction error + volatility level posterior update
// =============================================================================

pub fn prediction_error_volatile_state_node_unbounded(network: &mut Network, node_idx: usize, time_step: f64) {
    compute_volatile_prediction_errors(network, node_idx);

    let (precision_vol, mean_vol) = unbounded_volatility_level_update(network, node_idx, time_step);
    network.attributes.states[node_idx].precision_vol =
        precision_vol.min(network.max_posterior_precision);
    network.attributes.states[node_idx].mean_vol = mean_vol;
}

fn unbounded_volatility_level_update(network: &Network, node_idx: usize, time_step: f64) -> (f64, f64) {
    let s = &network.attributes.states[node_idx];
    let muhat_j = s.expected_mean_vol;
    let pihat_j = s.expected_precision_vol;
    let ka = s.volatility_coupling_internal;
    let om = s.tonic_volatility;
    let mean = s.mean;
    let expected_mean = s.expected_mean;
    let precision = s.precision;
    let expected_precision = s.expected_precision;
    let al_aux = s.current_variance.max(1e-128); // 1/pi_prev_jm1
    let be_aux = (1.0 / precision) + (mean - expected_mean).powi(2);

    // Canonical exponent at prediction: y = log(t_k) + ka*muhat_j + om
    let gamma_c = time_step.ln() + ka * muhat_j + om;

    // Recompute v and w using muhat_j. w is written as 1/(1 + al_aux/v) so
    // it stays finite when v_jm1 overflows to +inf (→ 1), matching Julia.
    let v_jm1 = gamma_c.exp();
    let w_jm1 = 1.0 / (1.0 + al_aux / v_jm1);
    // Volatility prediction error: da_jm1 = pihat * be_aux - 1, with pihat the
    // *marginal* predicted precision (expected_precision). Matches the standard /
    // eHGF volatility PE and the JAX/Python backends; the earlier
    // be_aux/(al_aux+v_jm1) - 1 form used a no-MGF/no-coupling conditional precision.
    let da_jm1 = expected_precision * be_aux - 1.0;

    // Expansion 1: quadratic at the prediction (prior mean)
    let pi1 = pihat_j + 0.5 * ka.powi(2) * w_jm1 * (1.0 - w_jm1);
    let mu1 = muhat_j + (ka * w_jm1 / (2.0 * pi1)) * da_jm1;

    // Expansion 2: quadratic at the Lambert W0 approximate mode.
    // W_arg is computed in log-space and capped at log(f64::MAX) to match the
    // MATLAB reference: W_arg = exp(min(log_W_arg, log(realmax))).
    let pihat_y = pihat_j / ka.powi(2);
    let log_w_arg = be_aux.ln() - (2.0 * pihat_y).ln() + 0.5 / pihat_y - gamma_c;
    let w_arg = log_w_arg.min(f64::MAX.ln()).exp();
    let v_w = lambert_w0(w_arg);
    let y_star = gamma_c + v_w - 0.5 / pihat_y;
    let x_star = (y_star - time_step.ln() - om) / ka;

    // Rearranged w/da formulas stay finite when s2 overflows (→ w=1, da=-1).
    let s2 = time_step * (ka * x_star + om).exp();
    let w2 = 1.0 / (1.0 + al_aux / s2);
    let da2 = be_aux / (al_aux + s2) - 1.0;

    let pi2_full = pihat_j + 0.5 * ka.powi(2) * w2 * (w2 + (2.0 * w2 - 1.0) * da2);
    let pi2_safe = if pi2_full <= 0.0 {
        pihat_j + 0.5 * ka.powi(2) * w2 * (1.0 - w2)
    } else {
        pi2_full
    };
    let mu2_safe = x_star + (0.5 * ka * w2 * da2 - pihat_j * (x_star - muhat_j)) / pi2_safe;

    // Fall back to Expansion 1 if Expansion 2 yields non-finite results —
    // matches MATLAB: "if ~isfinite(pi2) || ~isfinite(mu2), pi2 = pi1; mu2 = mu1".
    let exp2_finite = pi2_safe.is_finite() && mu2_safe.is_finite();
    let pi2 = if exp2_finite { pi2_safe } else { pi1 };
    let mu2 = if exp2_finite { mu2_safe } else { mu1 };

    // Variational energy-based softmax blend (direct form, matches MATLAB)
    let ey1 = time_step * (ka * mu1 + om).exp();
    let i1 = -0.5 * (al_aux + ey1).ln()
        - 0.5 * be_aux / (al_aux + ey1)
        - 0.5 * pihat_j * (mu1 - muhat_j).powi(2);

    let ey2 = time_step * (ka * mu2 + om).exp();
    let i2 = -0.5 * (al_aux + ey2).ln()
        - 0.5 * be_aux / (al_aux + ey2)
        - 0.5 * pihat_j * (mu2 - muhat_j).powi(2);

    let b = 1.0 / (1.0 + (i1 - i2).exp()); // sigmoid(i2 - i1)

    // Gaussian mixture moment matching
    let posterior_mean = (1.0 - b) * mu1 + b * mu2;
    let sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2).powi(2);
    let posterior_precision = 1.0 / sig2;

    (posterior_precision, posterior_mean)
}
