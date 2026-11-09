use crate::model::network::Network;

/// Floor on the total predicted variance (carried + volatility + value-coupling):
/// at λ = 0 the carried term vanishes (precision can diverge as Ω → 0); at λ = 1
/// the sum already exceeds it. Mirrors `pyhgf.updates.prediction.continuous`.
pub const MIN_PREDICTED_VARIANCE: f64 = 1e-128;

/// Apply [`MIN_PREDICTED_VARIANCE`], propagating NaN: `f64::max` swallows NaN but
/// `jnp.maximum` propagates it, and the floor must keep the volatility guard's NaN
/// visible so the two backends agree.
#[inline]
pub(crate) fn floor_predicted_variance(variance: f64) -> f64 {
    if variance.is_nan() {
        f64::NAN
    } else {
        variance.max(MIN_PREDICTED_VARIANCE)
    }
}

/// Whether the node is an input node with a noise parent: a continuous leaf (no value
/// or volatility children) with at least one volatility parent. Following Weber et al.
/// (2026), a volatility parent of an input node is a noise parent: it sets the variance
/// of the observation noise. Mirrors `is_noise_child` in the JAX backend.
pub fn is_noise_child(network: &Network, node_idx: usize) -> bool {
    let edges = &network.edges[node_idx];
    edges.node_type.as_str() == "continuous-state"
        && edges.value_children.is_none()
        && edges.volatility_children.is_none()
        && edges.volatility_parents.is_some()
}

/// Predicted precision of the observation noise of an input node with noise parents
/// (Eq. 66 of Weber et al., 2026):
///
/// ```text
/// π̂_u = 1 / exp(ε_u + Σ_q κ_q μ̂_q),
/// ```
///
/// with the tonic observation noise ε_u given by the `tonic_volatility` of the input
/// node, which is 0 for an input node (a tonic noise is a shift of ε_u / κ of the mean
/// of a noise parent). The `precision` of the input node is the precision of the
/// observation noise without noise parent, and is not used here. The noise parents
/// enter at their predicted means, as point estimates, and the noise variance does not
/// scale with the time step, under both the relaxed and the mean-field predictions.
/// Mirrors `predict_noise_precision` in the JAX backend.
pub fn predict_noise_precision(network: &Network, node_idx: usize) -> f64 {
    let mut log_noise_variance = network.attributes.states[node_idx].tonic_volatility;
    if let Some(ref noise_parent_idxs) = network.edges[node_idx].volatility_parents {
        let couplings = &network.attributes.vectors[node_idx].volatility_coupling_parents;
        for (i, &parent_idx) in noise_parent_idxs.iter().enumerate() {
            let kappa = couplings.get(i).copied().unwrap_or(1.0);
            log_noise_variance += kappa * network.attributes.states[parent_idx].expected_mean;
        }
    }
    (-log_noise_variance).exp()
}

/// Overwrite the predicted precisions of an input node, which is observed exactly and
/// does not drift between observations: its `precision` without noise parent, the
/// noise precision of [`predict_noise_precision`] with noise parents, in which case the
/// effective precision is 1 (Eq. 73 of Weber et al., 2026). Other nodes are left as
/// they are.
fn store_input_prediction(network: &mut Network, node_idx: usize) {
    let edges = &network.edges[node_idx];
    if edges.value_children.is_some() || edges.volatility_children.is_some() {
        return;
    }
    let (precision, effective_precision) = if edges.volatility_parents.is_some() {
        (predict_noise_precision(network, node_idx), 1.0)
    } else {
        (network.attributes.states[node_idx].precision, 0.0)
    };
    let state = &mut network.attributes.states[node_idx];
    state.expected_precision = precision;
    state.conditional_expected_precision = precision;
    state.effective_precision = effective_precision;
}

/// Prediction step for a continuous state node.
///
/// Computes the predicted mean μ̂, the conditional predicted precision π̄
/// (`conditional_expected_precision`), the marginal predicted precision π̂
/// (`expected_precision`), and the effective precision γ.
///
/// * π̄ = 1 / (λ²/π + Ω): carried variance plus volatility, no parent-uncertainty
///   bleed-through. Used by the parent's Schur-complement posterior-step
///   relaxation.
/// * π̂ = 1 / (1/π̄ + Σ_b (Δt · α · g'(μ̂_b))² / π̂_b): inverse marginal
///   predictive variance, adding the first-order Taylor contribution from each
///   value parent (using the parent's marginal predicted precision π̂_b).
/// * Ω = Δt · exp(ω + Σ_j κ_j μ̂_vol_j + Σ_j κ_j²/(2 π̂_vol_j)): each volatility
///   parent enters through its predicted mean μ̂_vol and the
///   moment-generating-function term κ²/(2 π̂_vol), both from the current step
///   (parents are predicted before their children).
///
/// The carried variance takes λ²: scaling the belief by λ scales its variance by
/// λ². Only the node's own variance takes the factor; the drift is added after the
/// autoregression, so the value-coupling term is unscaled. At λ = 1 this is canonical.
/// An input node takes the precision of its observation noise instead, from its noise
/// parents if it has any (see `store_input_prediction`).
pub fn prediction_continuous_state_node(network: &mut Network, node_idx: usize, time_step: f64) {
    // Copy own scalar state (f64 is Copy — no borrow held)
    let mean = network.attributes.states[node_idx].mean;
    let tonic_drift = network.attributes.states[node_idx].tonic_drift;
    let autoconnection_strength = network.attributes.states[node_idx].autoconnection_strength;
    let precision = network.attributes.states[node_idx].precision;
    let tonic_volatility = network.attributes.states[node_idx].tonic_volatility;

    // -------------------------------------------------------
    // 1. Predict the mean: μ̂ = λ · μ + Δt · driftrate.
    //    Also accumulate the Taylor value-coupling variance
    //        Σ_b (Δt · α · g'(μ̂_b))² / π̂_b
    //    using each parent's marginal predicted precision π̂_b
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
            let alpha = couplings.get(i).copied().unwrap_or(1.0);
            let (parent_value, g_prime) = match network.attributes.fn_ptrs[parent_idx].coupling_fn {
                Some(cf) => ((cf.f)(parent_expected_mean), (cf.df)(parent_expected_mean)),
                None => (parent_expected_mean, 1.0),
            };
            driftrate += alpha * parent_value;
            // First-order Taylor expansion of g around μ̂_b yields a
            // (Δt · α · g'(μ̂_b))² / π̂_b contribution to the marginal
            // predictive variance of x_a. Vanishes as π̂_b → ∞.
            let coeff = time_step * alpha * g_prime;
            value_coupling_variance += coeff * coeff / parent_expected_precision;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    // -------------------------------------------------------
    // 2. Predict the two precisions:
    //        π̄ = 1 / (λ²/π + Ω)
    //        π̂ = 1 / (1/π̄ + value-coupling variance)
    //    Ω = Δt · exp(ω + Σ_j κ_j μ̂_j + Σ_j κ_j²/(2 π̂_vol_j)); the MGF term
    //    κ²/(2 π̂_vol) marginalises over each volatility parent's Gaussian rather
    //    than collapsing it to a point estimate.
    // -------------------------------------------------------
    let mut total_volatility = tonic_volatility;

    if let Some(ref vol_parent_idxs) = network.edges[node_idx].volatility_parents {
        let vol_couplings = &network.attributes.vectors[node_idx].volatility_coupling_parents;

        for (i, &parent_idx) in vol_parent_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let parent_expected_precision =
                network.attributes.states[parent_idx].expected_precision;
            let kappa = vol_couplings.get(i).copied().unwrap_or(1.0);
            total_volatility += kappa * parent_expected_mean;
            total_volatility += (kappa * kappa) / (2.0 * parent_expected_precision);
        }
    }

    let pv_raw = time_step * total_volatility.exp();
    let predicted_volatility = if pv_raw > 1e-128 { pv_raw } else { f64::NAN };
    // Carried variance: scaling the belief by λ scales its variance by λ².
    let carried_variance = (autoconnection_strength * autoconnection_strength) / precision;
    // Conditional predicted precision π̄_a: carried variance + volatility only,
    // WITHOUT the parent-uncertainty value-coupling term. The parent's posterior-step
    // Schur complement acts on this; the marginal would double-count parent uncertainty.
    let conditional_expected_precision =
        1.0 / floor_predicted_variance(carried_variance + predicted_volatility);
    let expected_precision = 1.0
        / floor_predicted_variance(
            carried_variance + predicted_volatility + value_coupling_variance,
        );
    // Effective precision γ — only the volatility-driven part enters γ, since
    // γ is consumed by the volatility-coupling posterior update.
    let effective_precision = predicted_volatility * expected_precision;

    // -------------------------------------------------------
    // 3. Store results
    // -------------------------------------------------------
    let state = &mut network.attributes.states[node_idx];
    // Store carried variance for the volatility-coupling posteriors (unbounded,
    // eHGF), which re-predict from it — same quantity the prediction step uses.
    state.current_variance = carried_variance;
    state.expected_mean = expected_mean;
    state.effective_precision = effective_precision;
    state.expected_precision = expected_precision;
    state.conditional_expected_precision = conditional_expected_precision;

    // An input node takes the precision of its observation noise instead.
    store_input_prediction(network, node_idx);
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
            let alpha = couplings.get(i).copied().unwrap_or(1.0);
            let parent_value = match network.attributes.fn_ptrs[parent_idx].coupling_fn {
                Some(cf) => (cf.f)(parent_expected_mean),
                None => parent_expected_mean,
            };
            driftrate += alpha * parent_value;
        }
    }

    let expected_mean = autoconnection_strength * mean + time_step * driftrate;

    let mut total_volatility = tonic_volatility;
    if let Some(ref vol_parent_idxs) = network.edges[node_idx].volatility_parents {
        let vol_couplings = &network.attributes.vectors[node_idx].volatility_coupling_parents;
        for (i, &parent_idx) in vol_parent_idxs.iter().enumerate() {
            let parent_expected_mean = network.attributes.states[parent_idx].expected_mean;
            let kappa = vol_couplings.get(i).copied().unwrap_or(1.0);
            total_volatility += kappa * parent_expected_mean;
        }
    }

    let pv_raw = time_step * total_volatility.exp();
    let predicted_volatility = if pv_raw > 1e-128 { pv_raw } else { f64::NAN };
    // Carried variance: scaling the belief by λ scales its variance by λ².
    let carried_variance = (autoconnection_strength * autoconnection_strength) / precision;
    let expected_precision =
        1.0 / floor_predicted_variance(carried_variance + predicted_volatility);
    let effective_precision = predicted_volatility * expected_precision;

    let state = &mut network.attributes.states[node_idx];
    state.current_variance = carried_variance;
    state.expected_mean = expected_mean;
    state.effective_precision = effective_precision;
    state.expected_precision = expected_precision;
    state.conditional_expected_precision = expected_precision;

    // An input node takes the precision of its observation noise instead.
    store_input_prediction(network, node_idx);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_precision_matches_hand_computation() {
        // λ = 0.2, π = 1, Ω = 0.1: carried variance 0.04, predicted variance 0.14,
        // predicted precision 7.142857. Dropping λ² would give 1/1.1 = 0.909091.
        let carried = (0.2_f64 * 0.2) / 1.0;
        let variance = floor_predicted_variance(carried + 0.1);
        assert!((variance - 0.14).abs() < 1e-15);
        assert!((1.0 / variance - 7.142_857_142_857_143).abs() < 1e-12);
    }

    #[test]
    fn the_variance_floor_propagates_nan() {
        // `f64::max` returns the other operand when one is NaN, which would swallow
        // the NaN the volatility underflow guard produces. `jnp.maximum` on the JAX
        // side propagates it, so the floor has to as well.
        assert!(floor_predicted_variance(f64::NAN).is_nan());
        assert_eq!(floor_predicted_variance(0.0), MIN_PREDICTED_VARIANCE);
        // Above the floor the value passes through unchanged.
        assert_eq!(floor_predicted_variance(0.14), 0.14);
    }
}
