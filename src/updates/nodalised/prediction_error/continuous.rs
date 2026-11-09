use crate::model::network::Network;
use crate::updates::nodalised::posterior::continuous::has_observations;
use crate::updates::nodalised::prediction::continuous::is_noise_child;

/// Prediction error from a continuous state node.
///
/// A node with value or volatility children also records whether at least one of
/// them was observed at this step, in its `observed` flag. Its parents then treat it
/// as unobserved after a step without observations: they receive no prediction error
/// from it and keep their predicted precision. The flag of a node without children is
/// the one set by the observation step (mirrors `continuous_node_prediction_error` in
/// the JAX backend). An input node with noise parents sends them the noise prediction
/// error of [`noise_prediction_error_continuous_input_node`] instead of a volatility
/// prediction error.
pub fn prediction_error_continuous_state_node(
    network: &mut Network,
    node_idx: usize,
    _time_step: f64,
) {
    let mean = network.attributes.states[node_idx].mean;
    let expected_mean = network.attributes.states[node_idx].expected_mean;
    let precision = network.attributes.states[node_idx].precision;
    let expected_precision = network.attributes.states[node_idx].expected_precision;

    // Value prediction error: δ = μ - μ̂
    let value_prediction_error = mean - expected_mean;

    // A node with children is observed when at least one of its children was.
    let edges = &network.edges[node_idx];
    let observed = if edges.value_children.is_some() || edges.volatility_children.is_some() {
        if has_observations(network, node_idx) {
            1.0
        } else {
            0.0
        }
    } else {
        network.attributes.states[node_idx].observed
    };
    let is_noise_child = is_noise_child(network, node_idx);

    let state = &mut network.attributes.states[node_idx];
    state.value_prediction_error = value_prediction_error;
    state.observed = observed;
    // Volatility prediction error: Δ = (π̂ / π) + π̂ · δ² - 1. Every volatility parent
    // of the node receives the full Δ, as in the unbounded update.
    if !is_noise_child {
        state.volatility_prediction_error = (expected_precision / precision)
            + expected_precision * (mean - expected_mean).powi(2)
            - 1.0;
    }
}

/// Noise prediction error of an input node with noise parents (Eq. 72 of Weber et al.,
/// 2026):
///
/// ```text
/// ε_u = π̂_u / π_p + π̂_u (u − μ_p)² − 1,
/// ```
///
/// with π̂_u the precision of the observation noise and (μ_p, π_p) the posterior of the
/// value parent p, so that this step runs after the update of the value parents and
/// before the update of the noise parents, which read it with the effective precision
/// γ_u = 1. With several value parents b, coupling strengths α_b, coupling functions
/// g_b and time step t, the expected squared noise is taken to first order under all
/// posteriors:
///
/// ```text
/// ε_u = π̂_u (Σ_b (t α_b g_b'(μ_b))² / π_b + (u − m_u)²) − 1,
/// ```
///
/// with m_u the predicted mean of u with every value parent moved from μ̂_b to μ_b.
/// Mirrors `continuous_node_noise_prediction_error` in the JAX backend.
pub fn noise_prediction_error_continuous_input_node(
    network: &mut Network,
    node_idx: usize,
    time_step: f64,
) {
    let mut posterior_mean = network.attributes.states[node_idx].expected_mean;
    let mut posterior_variance = 0.0;

    if let Some(ref value_parent_idxs) = network.edges[node_idx].value_parents {
        let couplings = &network.attributes.vectors[node_idx].value_coupling_parents;
        for (i, &parent_idx) in value_parent_idxs.iter().enumerate() {
            let alpha = couplings.get(i).copied().unwrap_or(1.0);
            let parent = &network.attributes.states[parent_idx];
            let (shift, g_prime) = match network.attributes.fn_ptrs[parent_idx].coupling_fn {
                Some(cf) => (
                    (cf.f)(parent.mean) - (cf.f)(parent.expected_mean),
                    (cf.df)(parent.mean),
                ),
                None => (parent.mean - parent.expected_mean, 1.0),
            };
            posterior_mean += time_step * alpha * shift;
            posterior_variance += (time_step * alpha * g_prime).powi(2) / parent.precision;
        }
    }

    let state = &mut network.attributes.states[node_idx];
    state.volatility_prediction_error = state.expected_precision
        * (posterior_variance + (state.mean - posterior_mean).powi(2))
        - 1.0;
}
