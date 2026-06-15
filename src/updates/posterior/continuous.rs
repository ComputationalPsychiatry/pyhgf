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

// =============================================================================
// Shared building blocks for precision and mean updates
// =============================================================================

/// Compute the precision update contribution from value and volatility children.
///
/// The value-coupling branch implements the posterior-step (smoothing) correction
/// of the relaxed HGF: the canonical child-precision factor is replaced by the
/// harmonic combination
///
/// ```text
/// π̂_a · π_y / (π̂_a + π_y),    π_y = π_a − π̃_a,
/// ```
///
/// where π̂_a is the child's conditional predicted precision
/// (`conditional_expected_precision`) and π̃_a its marginal predicted precision
/// (`expected_precision`). The same factor scales both the `(κ g')²` and the
/// `κ g'' · δ_a` terms. Reduces to the canonical formula when the child is fully
/// observed (π_y → ∞); returns no contribution when the child gained no bottom-up
/// information (π_y = 0).
///
/// Non-Gaussian children and Gaussian leaves fall back to the canonical
/// predicted-precision factor π̃_a — pyhgf keeps `precision = expected_precision`
/// for clamped observations, so the harmonic form would otherwise zero out their
/// contribution.
///
/// Volatility coupling is unchanged.
fn precision_update_from_children(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = 0.0;

    // --- Value coupling ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            // Effective child precision under the smoothing correction. The
            // Schur derivation assumes a Gaussian-Gaussian value-coupling edge,
            // so the correction only applies when the child carries a Gaussian
            // belief (continuous-state or volatile-state) AND is interior (has
            // children of its own). Binary, categorical, input/constant,
            // exponential-family, and Dirichlet children, plus any Gaussian
            // leaf, fall back to the canonical predicted-precision factor π̃_a
            // (paper's Limit 3, π_a → ∞).
            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior =
                matches!(child_node_type, "continuous-state" | "volatile-state")
                    && (network.edges[child_idx].value_children.is_some()
                        || network.edges[child_idx].volatility_children.is_some());
            let effective_child_precision = if child_is_gaussian_interior {
                // π_y = π_a − π̃_a; the Schur complement carries the *conditional*
                // predicted precision π̂_a (stored on the child). Using the
                // marginal would double-count parent uncertainty.
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * pi_y / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            precision_wpe += (effective_child_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    // --- Volatility coupling ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            precision_wpe += (0.5 * (kappa * effective_precision).powi(2)
                + (kappa * effective_precision).powi(2) * volatility_pe
                - 0.5 * kappa.powi(2) * effective_precision * volatility_pe)
                * observed;
        }
    }

    precision_wpe
}

/// Compute the mean update contribution from value and volatility children.
///
/// The value-coupling branch uses the joint-Gaussian (RTS-smoother) gain
///
/// ```text
/// g_a = π̂_a · π_a / (π̂_a + π_y),    π_y = π_a − π̃_a,
/// ```
///
/// accumulated across children and divided once by `node_precision` (π_b) — this is
/// what makes the multi-child mean exact rather than a sum of independent
/// single-child RTS gains. For leaves and non-Gaussian children π_y = 0 and g_a
/// collapses to π̃_a (= `child.expected_precision`), recovering the canonical gain.
fn mean_update_from_children(network: &Network, node_idx: usize, node_precision: f64) -> f64 {
    let mut value_pwpe = 0.0;
    let mut volatility_pwpe = 0.0;

    // --- Value coupling mean update ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_mean),
                None => 1.0,
            };

            // Joint-Gaussian (RTS-smoother) gain g_a = π̂_a · π_a / (π̂_a + π_y),
            // π_y = π_a − π̃_a; summed over children then divided once by node_precision
            // (exact multi-child marginal mean). Leaves / non-Gaussian children have
            // π_y = 0 so g_a collapses to the marginal, recovering the canonical gain.
            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior =
                matches!(child_node_type, "continuous-state" | "volatile-state")
                    && (network.edges[child_idx].value_children.is_some()
                        || network.edges[child_idx].volatility_children.is_some());
            let gain_precision = if child_is_gaussian_interior {
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * child_precision / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            value_pwpe +=
                (kappa * coupling_fn_prime * gain_precision / node_precision) * child_vape;
        }
    }

    // --- Volatility coupling mean update ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

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

pub fn posterior_update_continuous_state_node(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let max_posterior_precision = network.max_posterior_precision;

    let precision_wpe = precision_update_from_children(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);

    let mean_wpe = mean_update_from_children(network, node_idx, posterior_precision);
    let posterior_mean = expected_mean + mean_wpe;

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

// =============================================================================
// eHGF posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node_ehgf(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let max_posterior_precision = network.max_posterior_precision;

    let mean_wpe = mean_update_from_children(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    // eHGF safe precision update: recompute the effective precision from the
    // posterior mean and floor the volatility increment at zero.
    let precision_wpe = precision_update_from_children_ehgf(network, node_idx, time_step);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);
    network.attributes.states[node_idx].precision = posterior_precision;
}

// =============================================================================
// Unbounded posterior update
// =============================================================================

pub fn posterior_update_continuous_state_node_unbounded(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let volatility_child_idx = network.edges[node_idx]
        .volatility_children
        .as_ref()
        .expect("No volatility children found")[0];

    let volatility_coupling = network.attributes.vectors[node_idx]
        .volatility_coupling_children
        .get(0)
        .copied()
        .unwrap_or(1.0);

    let child_state = network.attributes.states[volatility_child_idx];
    let child_mean = child_state.mean;
    let child_precision = child_state.precision;
    let child_expected_mean = child_state.expected_mean;
    let tonic_volatility = child_state.tonic_volatility;
    let previous_variance = child_state.current_variance.max(1e-128); // previous-step variance (= 1 / precision at the previous step)
    let be_aux = (1.0 / child_precision) + (child_mean - child_expected_mean).powi(2);

    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let expected_precision = network.attributes.states[node_idx].expected_precision;

    // Canonical exponent at prediction: y = log(time_step) + volatility_coupling*expected_mean + tonic_volatility
    let gamma_c = time_step.ln() + volatility_coupling * expected_mean + tonic_volatility;

    // Expansion 1: quadratic at the prediction (prior mean).
    // w is written as 1/(1 + previous_variance/v) so it stays finite when v_jm1 overflows
    // to +inf (→ 1), matching Julia's rearrangement.
    let v_jm1 = gamma_c.exp();
    let w_jm1 = 1.0 / (1.0 + previous_variance / v_jm1);
    // Volatility prediction error: da_jm1 = pihat_child * be_aux - 1, with
    // pihat_child the volatility child's *marginal* predicted precision. Matches the
    // standard/eHGF volatility PE, the volatile-state node's fused update, and the
    // Python backend; the earlier be_aux/(previous_variance+v_jm1) - 1 form used a
    // no-MGF/no-coupling conditional precision.
    let da_jm1 = child_state.expected_precision * be_aux - 1.0;

    let pi1 = expected_precision + 0.5 * volatility_coupling.powi(2) * w_jm1 * (1.0 - w_jm1);
    let mu1 = expected_mean + (volatility_coupling * w_jm1 / (2.0 * pi1)) * da_jm1;

    // Expansion 2: quadratic at the Lambert W0 approximate mode.
    // W_arg is computed in log-space and capped at log(f64::MAX) to match the
    // MATLAB reference: W_arg = exp(min(log_W_arg, log(realmax))).
    let pihat_y = expected_precision / volatility_coupling.powi(2);
    let log_w_arg = be_aux.ln() - (2.0 * pihat_y).ln() + 0.5 / pihat_y - gamma_c;
    let w_arg = log_w_arg.min(f64::MAX.ln()).exp();
    let v_w = lambert_w0(w_arg);
    let y_star = gamma_c + v_w - 0.5 / pihat_y;
    let x_star = (y_star - time_step.ln() - tonic_volatility) / volatility_coupling;

    // Rearranged w/da formulas stay finite when s2 overflows (→ w=1, da=-1).
    let s2 = time_step * (volatility_coupling * x_star + tonic_volatility).exp();
    let w2 = 1.0 / (1.0 + previous_variance / s2);
    let da2 = be_aux / (previous_variance + s2) - 1.0;

    let pi2_full =
        expected_precision + 0.5 * volatility_coupling.powi(2) * w2 * (w2 + (2.0 * w2 - 1.0) * da2);
    let pi2_safe = if pi2_full <= 0.0 {
        expected_precision + 0.5 * volatility_coupling.powi(2) * w2 * (1.0 - w2)
    } else {
        pi2_full
    };
    let mu2_safe = x_star
        + (0.5 * volatility_coupling * w2 * da2 - expected_precision * (x_star - expected_mean))
            / pi2_safe;

    // Fall back to Expansion 1 if Expansion 2 yields non-finite results —
    // matches MATLAB: "if ~isfinite(pi2) || ~isfinite(mu2), pi2 = pi1; mu2 = mu1".
    let exp2_finite = pi2_safe.is_finite() && mu2_safe.is_finite();
    let pi2 = if exp2_finite { pi2_safe } else { pi1 };
    let mu2 = if exp2_finite { mu2_safe } else { mu1 };

    // Variational energy-based softmax blend (direct form, matches MATLAB)
    let ey1 = time_step * (volatility_coupling * mu1 + tonic_volatility).exp();
    let i1 = -0.5 * (previous_variance + ey1).ln()
        - 0.5 * be_aux / (previous_variance + ey1)
        - 0.5 * expected_precision * (mu1 - expected_mean).powi(2);

    let ey2 = time_step * (volatility_coupling * mu2 + tonic_volatility).exp();
    let i2 = -0.5 * (previous_variance + ey2).ln()
        - 0.5 * be_aux / (previous_variance + ey2)
        - 0.5 * expected_precision * (mu2 - expected_mean).powi(2);

    let b = 1.0 / (1.0 + (i1 - i2).exp()); // sigmoid(i2 - i1)

    // Gaussian mixture moment matching
    let posterior_mean = (1.0 - b) * mu1 + b * mu2;
    let sig2 = (1.0 - b) / pi1 + b / pi2 + b * (1.0 - b) * (mu1 - mu2).powi(2);
    let posterior_precision = (1.0 / sig2).min(network.max_posterior_precision);

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

// =============================================================================
// Mean-field (v0.2.11) building blocks
// =============================================================================

/// Mean-field precision update from children.
///
/// Uses `expected_precision` directly as the child-precision factor.
fn precision_update_from_children_mean_field(network: &Network, node_idx: usize) -> f64 {
    let mut precision_wpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            precision_wpe += (child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let effective_precision = child_state.effective_precision;
            let volatility_pe = child_state.volatility_prediction_error;
            let observed = child_state.observed;
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);

            precision_wpe += (0.5 * (kappa * effective_precision).powi(2)
                + (kappa * effective_precision).powi(2) * volatility_pe
                - 0.5 * kappa.powi(2) * effective_precision * volatility_pe)
                * observed;
        }
    }

    precision_wpe
}

/// Enhanced-HGF "safe" volatility-coupling precision increment for one child.
///
/// Recomputes the effective precision from the parent's just-updated posterior mean
/// (`mean`) and the elapsed time, then floors the increment at zero, matching the
/// enhanced HGF volatility update (TAPAS `hgf_volatility_update` `'ehgf'` branch). This
/// guarantees the posterior precision never drops below the predicted precision.
fn ehgf_volatility_increment(
    child_state: &crate::model::NodeState,
    volatility_coupling: f64,
    mean: f64,
    time_step: f64,
) -> f64 {
    // Child posterior variance at the previous step (σ = 1 / π).
    let previous_variance = child_state.current_variance;
    // Re-predict the child's volatility and precision from the parent posterior mean.
    let predicted_volatility =
        time_step * (volatility_coupling * mean + child_state.tonic_volatility).exp();
    let expected_precision = 1.0 / (previous_variance + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;
    let volatility_error_weight = (predicted_volatility - previous_variance) * expected_precision;
    let volatility_prediction_error = (1.0 / child_state.precision
        + (child_state.mean - child_state.expected_mean).powi(2))
        * expected_precision
        - 1.0;
    (0.5 * volatility_coupling.powi(2)
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error))
        .max(0.0)
        * child_state.observed
}

/// Enhanced-HGF precision update from children (relaxed value coupling).
fn precision_update_from_children_ehgf(network: &Network, node_idx: usize, time_step: f64) -> f64 {
    let mut precision_wpe = 0.0;

    // --- Value coupling (identical to the relaxed standard update) ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            let child_node_type = network.edges[child_idx].node_type.as_str();
            let child_is_gaussian_interior =
                matches!(child_node_type, "continuous-state" | "volatile-state")
                    && (network.edges[child_idx].value_children.is_some()
                        || network.edges[child_idx].volatility_children.is_some());
            let effective_child_precision = if child_is_gaussian_interior {
                let child_precision = child_state.precision;
                let pi_y = child_precision - child_expected_precision;
                let child_cond = child_state.conditional_expected_precision;
                child_cond * pi_y / (child_cond + pi_y)
            } else {
                child_expected_precision
            };

            precision_wpe += (effective_child_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    // --- Volatility coupling (eHGF safe update) ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);
            precision_wpe += ehgf_volatility_increment(child_state, kappa, parent_mean, time_step);
        }
    }

    precision_wpe
}

/// Enhanced-HGF precision update from children (mean-field value coupling).
fn precision_update_from_children_ehgf_mean_field(
    network: &Network,
    node_idx: usize,
    time_step: f64,
) -> f64 {
    let mut precision_wpe = 0.0;

    // --- Value coupling (identical to the mean-field standard update) ---
    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let observed = child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let (coupling_fn_prime_sq, coupling_fn_second_term) = match coupling_fn {
                Some(cf) => {
                    let g_prime = (cf.df)(parent_mean);
                    let g_second = (cf.d2f)(parent_mean);
                    let child_vape = child_state.value_prediction_error;
                    (g_prime.powi(2), kappa * g_second * child_vape)
                }
                None => (1.0, 0.0),
            };

            precision_wpe += (child_expected_precision
                * (kappa.powi(2) * coupling_fn_prime_sq - coupling_fn_second_term))
                * observed;
        }
    }

    // --- Volatility coupling (eHGF safe update) ---
    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;

        for (i, &child_idx) in volc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let kappa = vol_coupling_strengths.get(i).copied().unwrap_or(1.0);
            precision_wpe += ehgf_volatility_increment(child_state, kappa, parent_mean, time_step);
        }
    }

    precision_wpe
}

/// Mean-field mean update from children.
///
/// Uses `expected_precision` directly as the value-coupling gain.
fn mean_update_from_children_mean_field(
    network: &Network,
    node_idx: usize,
    node_precision: f64,
) -> f64 {
    let mut value_pwpe = 0.0;
    let mut volatility_pwpe = 0.0;

    if let Some(ref vc_idxs) = network.edges[node_idx].value_children {
        let coupling_strengths = &network.attributes.vectors[node_idx].value_coupling_children;
        let parent_mean = network.attributes.states[node_idx].mean;
        let coupling_fn = network.attributes.fn_ptrs[node_idx].coupling_fn;

        for (i, &child_idx) in vc_idxs.iter().enumerate() {
            let child_state = &network.attributes.states[child_idx];
            let child_expected_precision = child_state.expected_precision;
            let child_vape = child_state.value_prediction_error * child_state.observed;
            let kappa = coupling_strengths.get(i).copied().unwrap_or(1.0);

            let coupling_fn_prime = match coupling_fn {
                Some(cf) => (cf.df)(parent_mean),
                None => 1.0,
            };

            value_pwpe += (kappa * coupling_fn_prime * child_expected_precision / node_precision)
                * child_vape;
        }
    }

    if let Some(ref volc_idxs) = network.edges[node_idx].volatility_children {
        let vol_coupling_strengths =
            &network.attributes.vectors[node_idx].volatility_coupling_children;

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
// Mean-field posterior updates
// =============================================================================

pub fn posterior_update_continuous_state_node_mean_field(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let max_posterior_precision = network.max_posterior_precision;

    let precision_wpe = precision_update_from_children_mean_field(network, node_idx);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);

    let mean_wpe = mean_update_from_children_mean_field(network, node_idx, posterior_precision);
    let posterior_mean = expected_mean + mean_wpe;

    let state = &mut network.attributes.states[node_idx];
    state.precision = posterior_precision;
    state.mean = posterior_mean;
}

pub fn posterior_update_continuous_state_node_ehgf_mean_field(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let expected_precision = network.attributes.states[node_idx].expected_precision;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let max_posterior_precision = network.max_posterior_precision;

    let mean_wpe = mean_update_from_children_mean_field(network, node_idx, expected_precision);
    let posterior_mean = expected_mean + mean_wpe;
    network.attributes.states[node_idx].mean = posterior_mean;

    // eHGF safe precision update (mean-field value coupling).
    let precision_wpe =
        precision_update_from_children_ehgf_mean_field(network, node_idx, time_step);
    let posterior_precision = (expected_precision + precision_wpe)
        .max(1e-128)
        .min(max_posterior_precision);
    network.attributes.states[node_idx].precision = posterior_precision;
}
